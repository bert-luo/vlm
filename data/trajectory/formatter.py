"""
formatter.py — Convert Anthropic-style trajectories to Qwen 3.5 training format.

Claude 4.6 produces interleaved thinking blocks between tool calls.
Qwen 3.5 style expects ALL reasoning in a single <think> block at the start of the
first assistant turn, with no thinking in subsequent turns.

Transformations applied:
  - Turn 1 assistant: collect all thinking blocks → single <think>...</think> prefix
  - Turn 2+ assistant: strip all thinking blocks
  - Everything else (text, tool_use, tool_result): pass through unchanged

The "raw" format preserves Anthropic content blocks as plain dicts.
The "qwen" format applies the thinking reformat and is ready for SFT serialization.
"""

import copy
import json


def reformat_to_qwen_style(messages: list[dict]) -> list[dict]:
    """
    Reformat a full trajectory (Anthropic content-block format) to Qwen 3.5 style.

    Args:
        messages: List of message dicts with role and content (content blocks as dicts).

    Returns:
        New list with thinking reformatted. Does not mutate input.
    """
    result = []
    assistant_index = 0

    for msg in messages:
        if msg["role"] != "assistant":
            result.append(copy.deepcopy(msg))
            continue

        content = msg.get("content", [])

        if isinstance(content, str):
            # Rare: simple string content, no thinking to handle
            result.append(copy.deepcopy(msg))
            assistant_index += 1
            continue

        thinking_texts = [b["thinking"] for b in content if b.get("type") == "thinking"]
        non_thinking = [b for b in content if b.get("type") != "thinking"]

        if assistant_index == 0 and thinking_texts:
            # First assistant turn: merge all thinking into a single <think> prefix block
            combined = "\n\n".join(thinking_texts)
            think_block = {"type": "text", "text": f"<think>\n{combined}\n</think>"}
            new_content = [think_block] + non_thinking
        else:
            # Later turns: drop thinking entirely (Qwen 3.5 style)
            new_content = non_thinking

        result.append({**msg, "content": new_content})
        assistant_index += 1

    return result


def serialize_messages_to_text(messages: list[dict]) -> str:
    """
    Serialize a trajectory to a human-readable / training-compatible text string
    using Qwen 3.5 chat template conventions.

    Tool calls are rendered as JSON inside <tool_call> tags.
    Tool results are rendered as <tool_response> inside a 'tool' role turn.

    This is useful for debugging and for feeding into a tokenizer's apply_chat_template.
    """
    parts = []

    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")

        if role == "system":
            text = content if isinstance(content, str) else _blocks_to_text(content)
            parts.append(f"<|im_start|>system\n{text}<|im_end|>")

        elif role == "user":
            if isinstance(content, list):
                # May be tool results
                text = _user_content_to_text(content)
            else:
                text = content
            parts.append(f"<|im_start|>user\n{text}<|im_end|>")

        elif role == "assistant":
            text = _blocks_to_text(content) if isinstance(content, list) else content
            parts.append(f"<|im_start|>assistant\n{text}<|im_end|>")

    return "\n".join(parts)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _blocks_to_text(blocks: list[dict]) -> str:
    """Convert a list of content blocks to text, handling tool_use serialization."""
    parts = []
    for block in blocks:
        btype = block.get("type")
        if btype == "text":
            parts.append(block["text"])
        elif btype == "thinking":
            # Should already be reformatted; include as-is if somehow present
            parts.append(f"<think>\n{block['thinking']}\n</think>")
        elif btype == "tool_use":
            call = json.dumps({"name": block["name"], "arguments": block.get("input", {})},
                              ensure_ascii=False)
            parts.append(f"<tool_call>\n{call}\n</tool_call>")
        elif btype == "image":
            parts.append("[rendered image]")
        # Skip unknown block types
    return "\n".join(parts)


def _user_content_to_text(content: list[dict] | str) -> str:
    """Handle user turns that may contain tool results."""
    if isinstance(content, str):
        return content

    parts = []
    for block in content:
        btype = block.get("type")
        if btype == "tool_result":
            inner = block.get("content", "")
            if isinstance(inner, str):
                inner_text = inner
            elif isinstance(inner, list):
                inner_text = _blocks_to_text(inner)
            else:
                inner_text = str(inner)
            parts.append(f"<tool_response>\n{inner_text}\n</tool_response>")
        elif btype == "text":
            parts.append(block["text"])
        elif btype == "image":
            parts.append("[image]")
    return "\n".join(parts)
