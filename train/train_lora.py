"""
LoRA fine-tuning script for Qwen3.5 VLM (vision-language) models.

Qwen3.5 models are native multimodal — they accept text **and** images.
This script loads the full VLM (``Qwen3_5ForConditionalGeneration``) with
``AutoProcessor`` so that rendered images from Blender trajectories are
processed through the vision encoder and fused with the text tokens.

Single GPU:
    python train/train_lora.py

Multi-GPU (torchrun, all GPUs):
    torchrun --nproc_per_node=8 train/train_lora.py

Multi-GPU (torchrun, specific GPUs e.g. 0-3):
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train/train_lora.py

Multi-GPU (accelerate):
    accelerate launch --num_processes=4 train/train_lora.py

Change model variant via --model_name_or_path, e.g.:
    --model_name_or_path Qwen/Qwen3.5-4B
    --model_name_or_path Qwen/Qwen3.5-8B
    --model_name_or_path Qwen/Qwen3.5-14B
"""

import argparse
import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.distributed as dist
from datasets import Dataset

from blender_dataset import (
    DEFAULT_TRAJECTORIES_JSONL,
    load_blender_trajectories_dataset,
    vlm_data_collator,
)
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoProcessor,
    DataCollatorForLanguageModeling,
    Qwen3_5ForConditionalGeneration,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

SCRIPT_DIR = Path(__file__).resolve().parent

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ScriptArgs:
    model_name_or_path: str = field(default="Qwen/Qwen3.5-4B")
    output_dir: str = field(default_factory=lambda: str(SCRIPT_DIR / "output" / "lora-blender-4b-mm"))
    # Data: default Blender trajectories JSONL; use --dummy_dataset for smoke tests
    trajectories_jsonl: str = field(default_factory=lambda: str(DEFAULT_TRAJECTORIES_JSONL))
    dummy_dataset: bool = field(default=False)
    trajectories_only_success: bool = field(default=True)
    trajectories_split: str | None = field(default=None)  # e.g. "train" to match prompt_meta.split
    # LoRA
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    # Training
    num_train_epochs: int = field(default=3)
    per_device_train_batch_size: int = field(default=2)
    gradient_accumulation_steps: int = field(default=4)
    learning_rate: float = field(default=2e-4)
    max_seq_length: int = field(default=8192)
    logging_steps: int = field(default=10)
    save_steps: int = field(default=100)
    fp16: bool = field(default=False)
    bf16: bool = field(default=True)
    gradient_checkpointing: bool = field(default=False)


def parse_args() -> ScriptArgs:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for Qwen3.5 models")
    defaults = ScriptArgs()
    parser.add_argument("--model_name_or_path", default=defaults.model_name_or_path)
    parser.add_argument("--output_dir", default=defaults.output_dir)
    parser.add_argument("--lora_r", type=int, default=defaults.lora_r)
    parser.add_argument("--lora_alpha", type=int, default=defaults.lora_alpha)
    parser.add_argument("--lora_dropout", type=float, default=defaults.lora_dropout)
    parser.add_argument("--num_train_epochs", type=int, default=defaults.num_train_epochs)
    parser.add_argument("--per_device_train_batch_size", type=int, default=defaults.per_device_train_batch_size)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=defaults.gradient_accumulation_steps)
    parser.add_argument("--learning_rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--max_seq_length", type=int, default=defaults.max_seq_length)
    parser.add_argument("--logging_steps", type=int, default=defaults.logging_steps)
    parser.add_argument("--save_steps", type=int, default=defaults.save_steps)
    parser.add_argument("--fp16", action="store_true", default=defaults.fp16)
    parser.add_argument("--bf16", action="store_true", default=defaults.bf16)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=defaults.gradient_checkpointing)
    parser.add_argument(
        "--trajectories_jsonl",
        default=defaults.trajectories_jsonl,
        help=f"Path to trajectories JSONL (default: {DEFAULT_TRAJECTORIES_JSONL})",
    )
    parser.add_argument(
        "--dummy_dataset",
        action="store_true",
        help="Use a small built-in text corpus instead of trajectories JSONL.",
    )
    parser.add_argument(
        "--trajectories_only_success",
        action=argparse.BooleanOptionalAction,
        default=defaults.trajectories_only_success,
        help="Keep only successful trajectories (default: true).",
    )
    parser.add_argument(
        "--trajectories_split",
        type=str,
        default=defaults.trajectories_split,
        help='If set, keep only rows with prompt_meta.split equal to this (e.g. "train").',
    )
    namespace = parser.parse_args()
    return ScriptArgs(**vars(namespace))


# ---------------------------------------------------------------------------
# Dummy dataset
# ---------------------------------------------------------------------------

DUMMY_SAMPLES = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning models learn patterns from data.",
    "Paris is the capital of France and a global cultural hub.",
    "Transformers use self-attention to model long-range dependencies.",
    "The mitochondria is the powerhouse of the cell.",
    "Neural networks are inspired by biological brain structures.",
    "Reinforcement learning agents maximize cumulative reward over time.",
    "Python is a popular language for scientific computing and AI.",
    "The speed of light in vacuum is approximately 299,792,458 m/s.",
    "Large language models are pre-trained on vast corpora of text.",
] * 20  # repeat to have a non-trivial number of samples


def make_dummy_dataset(tokenizer, max_length: int) -> Dataset:
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    ds = Dataset.from_dict({"text": DUMMY_SAMPLES})
    # load_from_cache_file=False avoids Arrow cache file-lock races when
    # multiple DDP ranks hit Dataset.map() simultaneously.
    ds = ds.map(tokenize, batched=True, remove_columns=["text"], load_from_cache_file=False)
    ds = ds.with_format("torch")
    return ds


# ---------------------------------------------------------------------------
# LoRA target modules per Qwen3.5 architecture
# ---------------------------------------------------------------------------

QWEN35_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def build_lora_config(args: ScriptArgs) -> LoraConfig:
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=QWEN35_LORA_TARGET_MODULES,
        bias="none",
    )


# ---------------------------------------------------------------------------
# Observability callback
# ---------------------------------------------------------------------------

class TrainingMonitor(TrainerCallback):
    """Logs loss, perplexity, learning rate, throughput, and GPU memory each
    logging interval, and prints an epoch summary at the end of every epoch."""

    def __init__(self):
        self._step_start: float = 0.0
        self._tokens_per_step: int = 0
        self._epoch_losses: list[float] = []

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        logger.info("=" * 60)
        logger.info("Training started")
        logger.info("  total steps     : %d", state.max_steps)
        logger.info("  epochs          : %s", args.num_train_epochs)
        logger.info("  batch / device  : %d", args.per_device_train_batch_size)
        logger.info("  grad accum steps: %d", args.gradient_accumulation_steps)
        logger.info("  learning rate   : %g", args.learning_rate)
        logger.info("=" * 60)
        self._step_start = time.perf_counter()

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self._step_start = time.perf_counter()

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs: dict, **kwargs):
        # Only log on the main process in DDP.
        if not state.is_world_process_zero:
            return

        loss = logs.get("loss")
        lr = logs.get("learning_rate")
        epoch = logs.get("epoch")
        grad_norm = logs.get("grad_norm")

        if loss is not None:
            ppl = math.exp(loss) if loss < 20 else float("inf")
            self._epoch_losses.append(loss)

            elapsed = time.perf_counter() - self._step_start
            steps_per_sec = args.logging_steps / max(elapsed, 1e-6)

            gpu_mem = ""
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                gpu_mem = f"  gpu_mem: {allocated:.2f}/{reserved:.2f} GB (alloc/reserved)"

            logger.info(
                "step %5d | epoch %s | loss: %.4f | ppl: %.2f | lr: %.2e | steps/s: %.2f%s%s",
                state.global_step,
                f"{epoch:.2f}" if epoch is not None else "?",
                loss,
                ppl,
                lr if lr is not None else float("nan"),
                steps_per_sec,
                f" | grad_norm: {grad_norm:.4f}" if grad_norm is not None else "",
                gpu_mem,
            )

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if not state.is_world_process_zero or not self._epoch_losses:
            return

        avg_loss = sum(self._epoch_losses) / len(self._epoch_losses)
        avg_ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
        logger.info("-" * 60)
        logger.info(
            "Epoch %d summary — avg loss: %.4f | avg ppl: %.2f",
            round(state.epoch) if state.epoch else 0,
            avg_loss,
            avg_ppl,
        )
        logger.info("-" * 60)
        self._epoch_losses.clear()

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if not state.is_world_process_zero:
            return
        logger.info("Training complete — best loss: %.4f (step %d)",
                    state.best_metric or float("nan"), state.global_step)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    cuda_available = torch.cuda.is_available()
    num_gpus = torch.cuda.device_count() if cuda_available else 0

    # Abort early if launched in distributed mode without CUDA.
    # Without GPUs, each rank loads the full model into CPU RAM; with 8 ranks
    # that means 8× model size in RAM, which triggers the OOM killer.
    if dist.is_initialized() and not cuda_available:
        raise RuntimeError(
            "Distributed training requires CUDA, but no GPU is available. "
            "Check your NVIDIA driver (run `nvidia-smi`). "
            "To train on CPU use a single process: `python train/train_lora.py`."
        )

    # Override bf16/fp16 flags when no GPU is present (single-process CPU run).
    if not cuda_available:
        if args.bf16 or args.fp16:
            logger.warning("No CUDA device found — disabling bf16/fp16, falling back to CPU + fp32.")
        args.bf16 = False
        args.fp16 = False

    logger.info("Model : %s", args.model_name_or_path)
    logger.info("Output: %s", args.output_dir)
    logger.info("GPUs  : %d visible", num_gpus)
    if cuda_available:
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            logger.info("  GPU %d: %s  (%.1f GB)", i, props.name, props.total_memory / 1024**3)

    # --- Processor (tokenizer + image processor) -----------------------------
    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # --- Dataset -------------------------------------------------------------
    if args.dummy_dataset:
        train_dataset = make_dummy_dataset(processor.tokenizer, max_length=args.max_seq_length)
    else:
        path = Path(args.trajectories_jsonl)
        if not path.is_file():
            raise FileNotFoundError(
                f"Trajectories JSONL not found: {path}. "
                "Pass --trajectories_jsonl or use --dummy_dataset for a smoke test."
            )
        train_dataset = load_blender_trajectories_dataset(
            processor,
            jsonl_path=path,
            max_length=args.max_seq_length,
            only_success=args.trajectories_only_success,
            split_filter=args.trajectories_split,
        )
    logger.info("Dataset size: %d samples", len(train_dataset))

    # Sync all ranks after dataset creation before touching the model.
    if dist.is_initialized():
        dist.barrier()

    # --- Model (full VLM: vision encoder + language decoder) -----------------
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32),
        trust_remote_code=True,
    )
    model.enable_input_require_grads()  # required for gradient checkpointing with PEFT

    # --- Apply LoRA ----------------------------------------------------------
    lora_config = build_lora_config(args)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # Disable KV-cache — incompatible with gradient computation during training.
    model.config.use_cache = False

    # Sync all ranks after model + LoRA setup so every rank enters Trainer
    # at the same time and the DDP weight broadcast doesn't stall.
    if dist.is_initialized():
        dist.barrier()

    # --- Training arguments --------------------------------------------------
    # LOCAL_RANK is read automatically from the environment by accelerate
    # (which Trainer uses as its distributed backend). Do not pass local_rank
    # explicitly — that conflicts with accelerate's own detection and can cause
    # a hang at DDP initialization.
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        fp16=args.fp16,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to="none",
        remove_unused_columns=False,
        # With standard LoRA all adapter params (q/k/v/o/gate/up/down_proj) are
        # used in every forward pass, so find_unused_parameters=False is safe.
        # Setting True adds a collective op after every backward that scans PEFT's
        # module hierarchy and can deadlock in multi-GPU runs.
        ddp_find_unused_parameters=False,
    )

    # --- Trainer -------------------------------------------------------------
    # Blender dataset yields variable-size vision tensors that need the VLM
    # collator; the dummy dataset is text-only so the standard LM collator works.
    if args.dummy_dataset:
        collator = DataCollatorForLanguageModeling(tokenizer=processor.tokenizer, mlm=False)
    else:
        collator = vlm_data_collator

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        callbacks=[TrainingMonitor()],
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    logger.info("Saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
