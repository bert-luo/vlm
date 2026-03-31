# Blender Code Generation — Qwen 3.5 VLM Fine-Tuning

Fine-tune a Qwen 3.5 vision-language model to generate Blender Python (`bpy`) scripts from scene descriptions, using multi-turn trajectories distilled from Claude.

## Project Structure

```
vlm/
├── data/
│   ├── generate_trajectories.py   # Claude distillation: prompt → multi-turn bpy trajectory
│   ├── prompts.jsonl              # 200 scene prompts with category, domain, tags, difficulty, split
│   ├── DATA_PLAN.md               # Prompt taxonomy and coverage plan
│   ├── trajectory/
│   │   ├── env.py                 # BlenderEnvironment: write_file + render tool implementations
│   │   ├── formatter.py           # Anthropic thinking → Qwen <think> block reformat
│   │   ├── writer.py              # JSONL writer (strips base64 images from stored records)
│   │   └── prompts.py             # Prompt loader with filtering (category, domain, tags, split)
│   └── trajectories/              # Generated output: trajectories.jsonl + artifacts/
├── train/
│   ├── train_lora.py              # LoRA SFT training script
│   └── blender_dataset.py         # JSONL + render PNGs → multimodal PyTorch Dataset
├── eval/
│   ├── generate.py                # Inference: trajectory mode (Claude) or simple mode (any endpoint)
│   └── serve_lora_generate.sh     # Spin up vLLM with optional LoRA, run generate.py against it
└── pyproject.toml                 # Root deps: transformers, peft, torch, anthropic, openai
```

## Design Decisions

### 1. Distillation from Claude Multi-Turn Trajectories

The core data generation strategy is **distilling agentic trajectories from Claude**, not collecting single-shot prompt-to-code pairs.

Each trajectory is a multi-turn conversation where Claude:
1. Reads a scene description
2. Writes a bpy script via the `write_file` tool
3. Renders it headlessly via the `render` tool
4. Inspects the rendered image (returned as base64 PNG)
5. Diagnoses errors from Blender tracebacks or visual issues
6. Iterates until the scene looks correct

This captures far richer training signal than one-shot examples. The model learns not just "write code" but the full loop of write → execute → inspect → debug. Even though the student model doesn't use tools at inference time, the multi-turn structure teaches it to reason about common failure modes (missing materials, wrong coordinates, deprecated Blender APIs) during its initial code generation.

Claude's extended thinking is enabled (default 8K token budget) so the teacher model reasons deeply about spatial layout, material choices, and error diagnosis before acting.

### 2. Qwen 3.5 VLM as the Student — Full Multimodal, LoRA on Language Only

The student model is **Qwen 3.5** (default 4B, also tested at 8B/14B) loaded as a full VLM via `Qwen3_5ForConditionalGeneration` + `AutoProcessor`. This means the vision encoder is active during training and receives the actual rendered images from each trajectory.

LoRA adapters target only the **language model's attention and MLP projections** (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`). The vision encoder stays frozen. The rationale: the pre-trained vision encoder already understands 3D renders, lighting, and spatial layout well enough. The task-specific knowledge (Blender API patterns, bpy idioms, scene construction strategies) lives in the language side. Adapting only the LM stack is also far more parameter-efficient — LoRA with r=16, alpha=32 typically adds <1% trainable parameters.

### 3. Thinking Format: Anthropic → Qwen Consolidation

Claude's native output interleaves `thinking` blocks between tool calls across multiple turns. Qwen 3.5's chat format expects a single `<think>...</think>` block at the start of a response.

The formatter (`trajectory/formatter.py`) reconciles this:
- **First assistant turn**: all of Claude's thinking blocks are merged into one `<think>` prefix
- **Subsequent turns**: thinking is stripped entirely

This loses the per-turn reasoning from later turns, but it matches Qwen's expected inference behavior and avoids confusing the model with a format it was never pre-trained on. The first-turn reasoning — which contains the most important planning about scene layout, material choices, and construction strategy — is preserved.

### 4. Assistant-Only Loss Masking

Only tokens inside **assistant turns** contribute to the training loss. System prompts, user messages, tool results, padding tokens, and vision-pad tokens are all masked with `IGNORE_INDEX = -100`.

The label builder scans for `<|im_start|>assistant\n` markers and unmasks everything from there to `<|im_end|>`, skipping pad and `<|image_pad|>` tokens even within assistant spans. This means the model learns to:
- Generate `<think>` reasoning
- Write `<tool_call>` blocks with correct JSON
- Produce well-formed bpy code

without wasting capacity memorizing the system prompt or parroting back tool outputs.

### 5. Post-Processor Truncation for Vision-Token Integrity

The dataset processes text and images through the Qwen processor **without truncation first**, then truncates manually to `max_seq_length` afterward. This ordering is critical: the processor expands each image into a variable number of `<|image_pad|>` tokens based on resolution and the vision encoder's patch grid. Truncating before processing would break the alignment between the number of image-pad tokens and the actual `pixel_values` / `image_grid_thw` tensors.

When truncation does remove image-pad tokens (long multi-turn trajectories with many renders), the dataset trims `pixel_values` and `image_grid_thw` to match, counting remaining `<|image_pad|>` tokens and keeping only as many complete images as fit within the token budget.

### 6. Blender Environment with Safety Gates

The `BlenderEnvironment` in `trajectory/env.py` implements two tools:

- **`write_file`**: Validates the script with `ast.parse()` before writing to disk. This catches syntax errors early, saving an expensive Blender subprocess invocation. Filenames are restricted to the working directory (no path traversal) and `.py` is enforced.
- **`render`**: Runs `blender --background --python <script>` with a 120-second timeout. On success, returns the rendered PNG as base64 for Claude to inspect. On failure, extracts the most relevant traceback lines (searching backward for the last `Traceback (most recent call last)` block).

Each trajectory gets its own working directory under `artifacts/<trajectory_id>/`, with renders numbered sequentially (`render_001.png`, `render_002.png`, ...). This makes it straightforward to trace the iteration history for any trajectory.

### 7. Constrained Blender Subset

The system prompt constrains scripts to a strict Blender 5.1 subset:
- **Primitives only** (cube, plane, cylinder, sphere, cone) — no external assets or `.blend` imports
- **EEVEE renderer** at 800×600 — fast headless rendering, good enough for training signal
- **AREA lights**, **Principled BSDF** materials with only Base Color, Roughness, Metallic
- **Explicit exclusion** of deprecated EEVEE properties (`use_bloom`, `use_ssr`, etc.) that cause `AttributeError` in Blender 5.1

This dramatically reduces the surface area for runtime errors and makes trajectories more likely to succeed. It also means the fine-tuned model produces scripts that are portable across Blender 5.1 installations without asset dependencies.

### 8. JSONL Without Inline Images

Trajectory records in `trajectories.jsonl` store the message history but **strip base64 image data** to keep file sizes manageable. The rendered PNGs live as files in `artifacts/<trajectory_id>/render_*.png` and are loaded at training time by the dataset class. This separation means:
- The JSONL stays small and human-inspectable
- Artifacts can be managed independently (e.g. rsync only PNGs for a specific run)
- Reproducibility requires keeping both the JSONL and the artifacts directory

### 9. Training Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| LoRA rank (r) | 16 | Good capacity/efficiency balance for code generation tasks |
| LoRA alpha | 32 | 2× rank, standard effective learning rate scaling |
| Dropout | 0.05 | Light regularization; dataset is small so some regularization helps |
| Learning rate | 2e-4 | Standard for LoRA fine-tuning |
| Scheduler | Cosine with 10 warmup steps | Fast warmup (small dataset), smooth decay |
| Batch size | 2 × 4 gradient accumulation = 8 effective | Balances GPU memory with training stability |
| Max sequence length | 8192 | Multi-turn trajectories with images easily exceed 4K tokens |
| Epochs | 3 | Multiple passes over a small distillation dataset |
| Precision | bf16 | Halves memory vs fp32; better dynamic range than fp16 for training |
| KV cache | Disabled | Incompatible with gradient computation during training |

### 10. Evaluation: Text-Only via vLLM

The evaluation script (`serve_lora_generate.sh`) serves the base model + LoRA adapter through **vLLM** and runs `generate.py` against it in **simple mode** — one-shot text generation with no image inputs and no tool use.

This means evaluation tests a different capability than training: "given only a text prompt, can the LoRA produce a runnable bpy script?" rather than "given a text prompt and render feedback images, can it iteratively refine?" This is intentional — at deployment time the model generates code without access to a Blender process, so the evaluation matches the deployment setting. The multimodal training signal is used to teach the model what good renders look like so it can "imagine" the output while writing code.

### 11. Prompt Taxonomy

The 200 prompts in `prompts.jsonl` are organized across 7 dimensions (domain, scene scale, functional role, object density, lighting, structural complexity, artistic style) and 18+ categories covering gaming (fantasy, sci-fi, post-apocalyptic, horror, stylized) and robotics (manipulation, navigation, inspection, collaborative). Each prompt has a `split` field for train/test separation.

### 12. Parallelism and Resumption in Data Generation

Trajectory generation supports `--workers N` with a thread pool and an optional `--rpm` token-bucket rate limiter to avoid 429s from the Anthropic API. Runs are resumable via `--resume --run-dir <path>` — the writer checks which prompts already have records in the output JSONL and skips them.

## Quick Start

```bash
# Generate training data (requires ANTHROPIC_API_KEY and blender on PATH)
uv run data/generate_trajectories.py --limit 10

# Train (single GPU)
python train/train_lora.py --trajectories_jsonl data/trajectories/run_*/trajectories.jsonl

# Train (multi-GPU)
torchrun --nproc_per_node=4 train/train_lora.py

# Evaluate with vLLM
CUDA_VISIBLE_DEVICES=0 ./eval/serve_lora_generate.sh train/output/lora-blender-4b-mm/checkpoint-40 "cozy coffee shop"

# Smoke test (no data needed)
python train/train_lora.py --dummy_dataset
```
