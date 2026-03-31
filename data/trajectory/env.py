"""
BlenderEnvironment — agentic tool environment for bpy script generation.

Provides two tools:
  write_file(path, content) — write a Python script to the working directory
  render(path)              — run Blender headlessly; return traceback or rendered PNG
"""

import ast
import base64
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


# Anthropic tool schemas for the two actions
TOOLS: list[dict] = [
    {
        "name": "write_file",
        "description": (
            "Write a Python bpy script to the working directory. "
            "Returns a syntax error immediately if the code is invalid Python, "
            "saving a render attempt."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Filename only (e.g. 'scene.py'). No directory components.",
                },
                "content": {
                    "type": "string",
                    "description": "Complete Python script content.",
                },
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "render",
        "description": (
            "Run Blender 5.1 headlessly on a previously written script. "
            "On failure returns the traceback so you can fix the script. "
            "On success returns the rendered PNG image so you can inspect quality."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Filename of the script to render (must have been written with write_file first).",
                },
            },
            "required": ["path"],
        },
    },
]


@dataclass
class ToolResult:
    """Wraps a tool execution result in Anthropic content-block format."""
    tool_use_id: str
    # Content blocks: list of {"type": "text", "text": ...} or {"type": "image", ...}
    content: list[dict] = field(default_factory=list)
    # Convenience flags for the orchestrator
    is_render_success: bool = False
    is_render_failure: bool = False


class BlenderEnvironment:
    """
    Manages a temporary working directory and executes tool calls.
    One instance per trajectory.
    """

    TOOLS = TOOLS

    def __init__(self, work_dir: Path, render_timeout: int = 120):
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.render_timeout = render_timeout
        self._render_count = 0

    def execute(self, tool_name: str, tool_input: dict, tool_use_id: str) -> ToolResult:
        if tool_name == "write_file":
            return self._write_file(
                path=tool_input.get("path", "scene.py"),
                content=tool_input.get("content", ""),
                tool_use_id=tool_use_id,
            )
        elif tool_name == "render":
            return self._render(
                path=tool_input.get("path", "scene.py"),
                tool_use_id=tool_use_id,
            )
        else:
            return ToolResult(
                tool_use_id=tool_use_id,
                content=[{"type": "text", "text": f"Unknown tool: {tool_name!r}. Use 'write_file' or 'render'."}],
            )

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    def _write_file(self, path: str, content: str, tool_use_id: str) -> ToolResult:
        # Reject directory traversal — filename only
        filename = Path(path).name
        if not filename.endswith(".py"):
            filename = filename + ".py"

        # Syntax check before touching disk
        try:
            ast.parse(content)
        except SyntaxError as exc:
            return ToolResult(
                tool_use_id=tool_use_id,
                content=[{
                    "type": "text",
                    "text": (
                        f"Python syntax error — file NOT written.\n"
                        f"Error: {exc}\n"
                        f"Fix the syntax error and call write_file again."
                    ),
                }],
            )

        dest = self.work_dir / filename
        dest.write_text(content, encoding="utf-8")

        return ToolResult(
            tool_use_id=tool_use_id,
            content=[{"type": "text", "text": f"File written: {filename} ({len(content)} bytes)"}],
        )

    def _render(self, path: str, tool_use_id: str) -> ToolResult:
        filename = Path(path).name
        script_path = self.work_dir / filename

        if not script_path.exists():
            return ToolResult(
                tool_use_id=tool_use_id,
                content=[{
                    "type": "text",
                    "text": (
                        f"Error: {filename!r} does not exist in the working directory. "
                        "Call write_file first."
                    ),
                }],
            )

        self._render_count += 1
        output_png = self.work_dir / f"render_{self._render_count:03d}.png"

        cmd = [
            "blender", "--background",
            "--python", str(script_path),
            "--",
            "--output", str(output_png),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.render_timeout,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                tool_use_id=tool_use_id,
                content=[{"type": "text", "text": f"Render timed out after {self.render_timeout}s."}],
                is_render_failure=True,
            )

        if output_png.exists():
            img_bytes = output_png.read_bytes()
            img_b64 = base64.standard_b64encode(img_bytes).decode()
            return ToolResult(
                tool_use_id=tool_use_id,
                content=[
                    {
                        "type": "text",
                        "text": (
                            f"Render succeeded ({len(img_bytes) // 1024} KB). "
                            "Inspect the image below for quality. "
                            "If the scene looks correct, you are done. "
                            "If there are visual issues, fix and re-render."
                        ),
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": img_b64,
                        },
                    },
                ],
                is_render_success=True,
            )

        # Failure — extract the most useful error lines
        combined = (result.stdout or "") + "\n" + (result.stderr or "")
        error_text = _extract_blender_error(combined)

        return ToolResult(
            tool_use_id=tool_use_id,
            content=[{
                "type": "text",
                "text": f"Render failed. Traceback:\n\n{error_text}",
            }],
            is_render_failure=True,
        )


def _extract_blender_error(output: str, max_lines: int = 50) -> str:
    """Pull the most relevant lines from Blender's combined stdout+stderr."""
    lines = output.splitlines()

    # Find the last Traceback block
    traceback_start = -1
    for i, line in enumerate(lines):
        if "Traceback (most recent call last)" in line:
            traceback_start = i

    if traceback_start >= 0:
        relevant = lines[traceback_start:]
        return "\n".join(relevant[:max_lines])

    # Fallback: lines containing error keywords
    error_keywords = ("Error:", "error:", "Exception", "KeyError", "AttributeError",
                      "TypeError", "NameError", "RuntimeError", "line ")
    error_lines = [l for l in lines if any(k in l for k in error_keywords)]
    if error_lines:
        return "\n".join(error_lines[-max_lines:])

    # Last resort: tail of output
    return "\n".join(lines[-max_lines:])
