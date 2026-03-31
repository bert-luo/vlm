"""
Blender trajectory JSONL → PyTorch Dataset for Qwen 3.5 VLM LoRA training.

Pipeline
--------
1. Read trajectories JSONL, filter by success / split.
2. Convert Anthropic-style ``messages_qwen`` content blocks to HF message dicts,
   resolving stripped image placeholders to render PNGs from the artifacts directory.
3. On each ``__getitem__``: format with ``processor.apply_chat_template()``, encode
   via ``processor()`` (tokenizes text **and** extracts vision features).
4. Build **labels** that mask non-assistant tokens (``-100``).
   Only assistant-generated content (including reasoning inside ``<think>``)
   contributes to the training loss.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from PIL import Image
from torch.utils.data import Dataset as TorchDataset

if TYPE_CHECKING:
    from transformers import AutoProcessor

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DATA_DIR = _PROJECT_ROOT / "data"
if str(_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_DIR))

from trajectory.formatter import _blocks_to_text

DEFAULT_TRAJECTORIES_JSONL = _PROJECT_ROOT / "data" / "trajectories" / "trajectories.jsonl"
DEFAULT_ARTIFACTS_DIR = _PROJECT_ROOT / "data" / "trajectories" / "artifacts"


# ---------------------------------------------------------------------------
# Anthropic → HF message conversion (multimodal)
# ---------------------------------------------------------------------------

def _messages_qwen_to_hf(
    messages_qwen: list[dict],
    render_pngs: list[Path],
) -> tuple[list[dict], list[Image.Image]]:
    """Convert Anthropic-style content-block messages to HF chat dicts.

    Image placeholders inside ``tool_result`` blocks are resolved to PIL Images
    loaded from the corresponding render PNGs (matched in filename order).

    Images are placed in a follow-up ``user`` message so that the Qwen VL chat
    template can insert ``<|vision_start|>…<|vision_end|>`` tokens reliably
    (the template always supports image content blocks in user turns).

    Returns:
        ``(hf_messages, pil_images)`` — messages ready for
        ``processor.apply_chat_template`` and the PIL images in conversation
        order (passed to ``processor()``).
    """
    hf: list[dict] = []
    images: list[Image.Image] = []
    render_idx = 0

    for msg in messages_qwen:
        role = msg["role"]
        content = msg.get("content", "")

        if role == "system":
            hf.append({
                "role": "system",
                "content": content if isinstance(content, str) else _blocks_to_text(content),
            })

        elif role == "user":
            if isinstance(content, str):
                hf.append({"role": "user", "content": content})
            elif isinstance(content, list):
                has_tool_result = any(b.get("type") == "tool_result" for b in content)
                if has_tool_result:
                    for block in content:
                        btype = block.get("type")
                        if btype == "tool_result":
                            inner = block.get("content", "")
                            if isinstance(inner, list):
                                text_parts: list[str] = []
                                inner_has_image = False
                                for ib in inner:
                                    ibt = ib.get("type")
                                    if ibt == "text":
                                        text_parts.append(ib["text"])
                                    elif ibt == "image":
                                        inner_has_image = True
                                inner_text = "\n".join(text_parts) if text_parts else ""
                                hf.append({"role": "tool", "content": inner_text})
                                if inner_has_image and render_idx < len(render_pngs):
                                    img = Image.open(render_pngs[render_idx]).convert("RGB")
                                    images.append(img)
                                    hf.append({"role": "user", "content": [{"type": "image"}]})
                                    render_idx += 1
                            elif isinstance(inner, str):
                                hf.append({"role": "tool", "content": inner})
                            else:
                                hf.append({"role": "tool", "content": str(inner)})
                        elif btype == "text":
                            hf.append({"role": "user", "content": block["text"]})
                else:
                    parts: list[dict] = []
                    for block in content:
                        btype = block.get("type")
                        if btype == "text":
                            parts.append({"type": "text", "text": block["text"]})
                        elif btype == "image":
                            if render_idx < len(render_pngs):
                                img = Image.open(render_pngs[render_idx]).convert("RGB")
                                images.append(img)
                                parts.append({"type": "image"})
                                render_idx += 1
                    if parts:
                        hf.append({"role": "user", "content": parts})
            else:
                hf.append({"role": "user", "content": str(content)})

        elif role == "assistant":
            text = _blocks_to_text(content) if isinstance(content, list) else content
            hf.append({"role": "assistant", "content": text})

    return hf, images


# ---------------------------------------------------------------------------
# Label builder — mask everything except assistant content
# ---------------------------------------------------------------------------

def _build_labels(
    input_ids: list[int],
    tokenizer,
    *,
    _cache: dict = {},
) -> list[int]:
    """Return labels with ``IGNORE_INDEX`` for non-assistant positions.

    Supervised region per assistant turn:
      tokens *after* ``<|im_start|>assistant\\n`` up to and including ``<|im_end|>``.
    Everything else (system, user, tool, padding, vision-pad) is masked.
    """
    if not _cache:
        _cache["im_start"] = tokenizer.convert_tokens_to_ids("<|im_start|>")
        _cache["im_end"] = tokenizer.convert_tokens_to_ids("<|im_end|>")
        _cache["pad"] = tokenizer.pad_token_id
        _cache["header"] = tokenizer.encode("assistant\n", add_special_tokens=False)
        _cache["hlen"] = len(_cache["header"])
        raw = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        _cache["image_pad"] = raw if isinstance(raw, int) and raw != tokenizer.unk_token_id else None

    im_start: int = _cache["im_start"]
    im_end: int = _cache["im_end"]
    pad: int | None = _cache["pad"]
    header: list[int] = _cache["header"]
    hlen: int = _cache["hlen"]
    image_pad: int | None = _cache["image_pad"]

    n = len(input_ids)
    labels = [IGNORE_INDEX] * n

    i = 0
    while i < n:
        if input_ids[i] == im_start:
            header_end = i + 1 + hlen
            if header_end <= n and input_ids[i + 1 : header_end] == header:
                j = header_end
                while j < n and input_ids[j] != im_end:
                    j += 1
                sup_end = min(j + 1, n)  # include <|im_end|>
                for k in range(header_end, sup_end):
                    tok = input_ids[k]
                    if tok == pad:
                        continue
                    if image_pad is not None and tok == image_pad:
                        continue
                    labels[k] = tok
                i = sup_end
                continue
        i += 1

    return labels


# ---------------------------------------------------------------------------
# JSONL reader + filter
# ---------------------------------------------------------------------------

def _read_and_filter(
    path: Path,
    *,
    only_success: bool,
    split_filter: str | None,
) -> tuple[list[tuple[list[dict], str]], int]:
    """Return ``(rows, total)`` where each row is ``(messages_qwen, trajectory_id)``."""
    rows: list[tuple[list[dict], str]] = []
    total = 0
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Bad JSON on line {lineno} of {path}") from exc
            if only_success and not obj.get("success", False):
                continue
            if split_filter is not None:
                meta = obj.get("prompt_meta") or {}
                if meta.get("split") != split_filter:
                    continue
            mq = obj.get("messages_qwen")
            if not mq:
                continue
            rows.append((mq, obj["id"]))
    return rows, total


# ---------------------------------------------------------------------------
# PyTorch Dataset — lazy multimodal processing
# ---------------------------------------------------------------------------

class BlenderTrajectoryDataset(TorchDataset):
    """Loads and processes trajectories on each access.

    Each ``__getitem__`` call:
      1. Loads render PNGs for the trajectory.
      2. Builds HF messages with image content blocks.
      3. Applies the chat template and runs the processor (tokenises text +
         extracts vision features for all images in the conversation).
      4. Builds assistant-only labels.

    The resulting dicts always contain ``input_ids``, ``attention_mask``,
    ``labels`` and, when images are present, ``pixel_values`` and
    ``image_grid_thw``.
    """

    def __init__(
        self,
        rows: list[tuple[list[dict], str]],
        processor: "AutoProcessor",
        artifacts_dir: Path,
        max_length: int,
    ):
        self.rows = rows
        self.processor = processor
        self.artifacts_dir = artifacts_dir
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        mq, traj_id = self.rows[idx]
        traj_art = self.artifacts_dir / traj_id
        render_pngs = sorted(traj_art.glob("render_*.png")) if traj_art.is_dir() else []

        hf_msgs, pil_images = _messages_qwen_to_hf(mq, render_pngs)

        text: str = self.processor.apply_chat_template(
            hf_msgs, tokenize=False, add_generation_prompt=False,
        )

        # Process WITHOUT truncation — the processor validates that image-pad
        # token counts match between text and input_ids, so truncation must
        # happen *after* the processor expands vision tokens.
        proc_kwargs: dict = {"text": [text], "return_tensors": "pt"}
        if pil_images:
            proc_kwargs["images"] = pil_images

        enc = self.processor(**proc_kwargs)

        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        # --- Manual truncation + padding to self.max_length ------------------
        seq_len = input_ids.size(0)
        pad_id = self.processor.tokenizer.pad_token_id or 0

        if seq_len > self.max_length:
            input_ids = input_ids[: self.max_length]
            attention_mask = attention_mask[: self.max_length]
        elif seq_len < self.max_length:
            pad_len = self.max_length - seq_len
            input_ids = torch.cat([input_ids, input_ids.new_full((pad_len,), pad_id)])
            attention_mask = torch.cat([attention_mask, attention_mask.new_zeros(pad_len)])

        # mm_token_type_ids — same truncate/pad treatment
        mm_token_type_ids: torch.Tensor | None = None
        if "mm_token_type_ids" in enc:
            mm_token_type_ids = enc["mm_token_type_ids"].squeeze(0)
            if seq_len > self.max_length:
                mm_token_type_ids = mm_token_type_ids[: self.max_length]
            elif seq_len < self.max_length:
                pad_len = self.max_length - seq_len
                mm_token_type_ids = torch.cat([mm_token_type_ids, mm_token_type_ids.new_zeros(pad_len)])

        # --- Reconcile vision tensors after truncation -----------------------
        # If truncation removed image-pad tokens, we must drop the
        # corresponding rows from pixel_values / image_grid_thw so the model
        # doesn't try to consume pixels for tokens that no longer exist.
        pixel_values: torch.Tensor | None = None
        image_grid_thw: torch.Tensor | None = None

        if "pixel_values" in enc and "image_grid_thw" in enc:
            image_pad_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
            n_remaining = int((input_ids == image_pad_id).sum().item())

            if n_remaining > 0:
                all_grid = enc["image_grid_thw"]          # (n_images, 3)
                all_pv = enc["pixel_values"]              # (total_patches, C)
                merge_sq = self.processor.image_processor.merge_size ** 2

                kept, tok_budget = 0, 0
                for g in all_grid:
                    toks = int(g.prod().item() // merge_sq)
                    if tok_budget + toks <= n_remaining:
                        tok_budget += toks
                        kept += 1
                    else:
                        break

                if kept > 0:
                    image_grid_thw = all_grid[:kept]
                    kept_patches = sum(int(g.prod().item()) for g in image_grid_thw)
                    pixel_values = all_pv[:kept_patches]

        # --- Labels ----------------------------------------------------------
        labels = torch.tensor(
            _build_labels(input_ids.tolist(), self.processor.tokenizer),
            dtype=torch.long,
        )

        result: dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        if mm_token_type_ids is not None:
            result["mm_token_type_ids"] = mm_token_type_ids
        if pixel_values is not None:
            result["pixel_values"] = pixel_values
            result["image_grid_thw"] = image_grid_thw

        return result


# ---------------------------------------------------------------------------
# Data collator — handles variable-size vision tensors across a batch
# ---------------------------------------------------------------------------

def vlm_data_collator(features: list[dict]) -> dict[str, torch.Tensor]:
    """Collate that stacks fixed-size text tensors and concatenates vision tensors."""
    batch: dict[str, torch.Tensor] = {
        "input_ids": torch.stack([f["input_ids"] for f in features]),
        "attention_mask": torch.stack([f["attention_mask"] for f in features]),
        "labels": torch.stack([f["labels"] for f in features]),
    }

    if "mm_token_type_ids" in features[0]:
        batch["mm_token_type_ids"] = torch.stack([f["mm_token_type_ids"] for f in features])

    pv = [f["pixel_values"] for f in features if "pixel_values" in f]
    if pv:
        batch["pixel_values"] = torch.cat(pv, dim=0)

    gt = [f["image_grid_thw"] for f in features if "image_grid_thw" in f]
    if gt:
        batch["image_grid_thw"] = torch.cat(gt, dim=0)

    return batch


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def load_blender_trajectories_dataset(
    processor: "AutoProcessor",
    *,
    jsonl_path: str | Path | None = None,
    artifacts_dir: str | Path | None = None,
    max_length: int = 512,
    only_success: bool = True,
    split_filter: str | None = None,
) -> BlenderTrajectoryDataset:
    """Load trajectory metadata and return a Dataset that processes on access.

    Args:
        processor: Qwen 3.5 ``AutoProcessor`` (bundles tokenizer + image processor).
        jsonl_path: Path to the JSONL; defaults to the project trajectories file.
        artifacts_dir: Root of per-trajectory artifact directories containing
            render PNGs.  Defaults to ``data/trajectories/artifacts/``.
        max_length: Max token length (truncation + padding).  Multi-turn
            trajectories with images easily exceed 4 K tokens — consider
            setting this to at least 4096.
        only_success: Keep only rows with ``success == True``.
        split_filter: If set, keep rows whose ``prompt_meta.split`` matches.

    Returns:
        :class:`BlenderTrajectoryDataset` yielding dicts with ``input_ids``,
        ``attention_mask``, ``labels``, and (when images are present)
        ``pixel_values`` / ``image_grid_thw``.
    """
    path = Path(jsonl_path) if jsonl_path is not None else DEFAULT_TRAJECTORIES_JSONL
    art = Path(artifacts_dir) if artifacts_dir is not None else DEFAULT_ARTIFACTS_DIR

    if not path.is_file():
        raise FileNotFoundError(f"Trajectories file not found: {path}")

    rows, total = _read_and_filter(path, only_success=only_success, split_filter=split_filter)
    logger.info(
        "Blender trajectories: kept %d / %d rows (only_success=%s, split_filter=%s)",
        len(rows), total, only_success, split_filter,
    )
    if not rows:
        raise ValueError(
            "No rows left after filtering; check JSONL path, success flags, and split_filter."
        )

    return BlenderTrajectoryDataset(
        rows=rows,
        processor=processor,
        artifacts_dir=art,
        max_length=max_length,
    )
