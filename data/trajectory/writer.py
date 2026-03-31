"""
writer.py — Trajectory dataclass and JSONL writer.

Trajectories are written one-per-line to a JSONL file.
Base64 image data is stripped from stored messages (images are saved as files);
the final_image_path field points to the on-disk PNG.
"""

import copy
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class Trajectory:
    # Identity
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    prompt: str = ""
    prompt_meta: dict = field(default_factory=dict)
    model: str = ""

    # Outcome
    success: bool = False           # True if at least one render succeeded
    num_render_attempts: int = 0
    num_assistant_turns: int = 0

    # Messages — two representations:
    #   raw:  Anthropic content-block format (thinking blocks preserved as-is)
    #   qwen: thinking reformatted to Qwen 3.5 style (<think> prefix in turn 1)
    messages_raw: list = field(default_factory=list)
    messages_qwen: list = field(default_factory=list)

    # Artifacts
    final_script: str = ""          # Last script written by the model
    final_image_path: str | None = None   # Path to last successful render PNG

    # Book-keeping
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    error: str | None = None        # Set if the trajectory ended due to an exception


class TrajectoryWriter:
    """
    Append trajectories to a JSONL file.

    Usage:
        with TrajectoryWriter(Path("out.jsonl")) as writer:
            writer.write(traj)
    """

    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = None

    def __enter__(self) -> "TrajectoryWriter":
        self._file = self.output_path.open("a", encoding="utf-8")
        return self

    def __exit__(self, *args) -> None:
        if self._file:
            self._file.close()
            self._file = None

    def write(self, traj: Trajectory) -> None:
        record = asdict(traj)
        record["messages_raw"] = _strip_inline_images(record["messages_raw"])
        record["messages_qwen"] = _strip_inline_images(record["messages_qwen"])
        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._file.flush()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _strip_inline_images(messages: list[dict]) -> list[dict]:
    """
    Replace base64 image payloads with a lightweight placeholder.

    Images are saved to disk by the environment; inline base64 in JSONL
    bloats files by ~1 MB per render and isn't needed for SFT training.
    """
    result = []
    for msg in messages:
        msg = copy.deepcopy(msg)
        content = msg.get("content")
        if isinstance(content, list):
            msg["content"] = [_strip_image_block(b) for b in content]
        result.append(msg)
    return result


def _strip_image_block(block: dict) -> dict:
    btype = block.get("type")
    if btype == "image":
        src = block.get("source", {})
        if src.get("type") == "base64":
            return {"type": "image", "source": {"type": "file_ref", "note": "base64 stripped; see final_image_path"}}
        return block
    if btype == "tool_result":
        # Images may be nested inside tool_result.content
        inner = block.get("content")
        if isinstance(inner, list):
            return {**block, "content": [_strip_image_block(b) for b in inner]}
    return block
