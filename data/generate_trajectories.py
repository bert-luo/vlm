"""
generate_trajectories.py — Distil Claude 4.6 multi-turn bpy trajectories for SFT.

Each trajectory demonstrates:
  - Multi-turn agentic tool use (write_file → render → fix → re-render)
  - Iterative error correction via real Blender tracebacks
  - Visual quality checking on successful renders
  - Reasoning captured in Qwen 3.5 style (<think> block up-front, stripped in later turns)

Each run creates its own timestamped output directory under data/trajectories/
containing trajectories.jsonl and an artifacts/ subdirectory.

Usage:
    # All prompts, default model (creates data/trajectories/run_<timestamp>/)
    uv run data/generate_trajectories.py

    # Filter + limit
    uv run data/generate_trajectories.py --category gaming_fantasy --limit 10

    # Specific model, more thinking budget
    uv run data/generate_trajectories.py --model claude-opus-4-6 --thinking-budget 12000

    # Parallel workers
    uv run data/generate_trajectories.py --workers 4

    # Resume a previous run
    uv run data/generate_trajectories.py --resume --run-dir data/trajectories/run_20260330_143000
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import anthropic
from dotenv import load_dotenv

HERE = Path(__file__).parent.resolve()
load_dotenv(HERE.parent / ".env")

from trajectory import (
    BlenderEnvironment,
    PromptEntry,
    PromptLoader,
    Trajectory,
    TrajectoryWriter,
    reformat_to_qwen_style,
)

# ------------------------------------------------------------------
# System prompt
# ------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert Blender 5.1 Python (bpy) scripter. Your task is to generate a \
complete, working Blender scene script from a scene description, then verify it \
renders correctly.

You have two tools:
  write_file(path, content) — write a Python script to the working directory
  render(path)              — run Blender headlessly; returns a traceback on \
failure, or the rendered PNG on success

Workflow:
1. Write a complete bpy script with write_file.
2. Render it with render.
3. If the render fails, study the traceback, fix the root cause, and try again.
4. If the render succeeds, inspect the image. If the scene looks correct and \
complete, stop. If there are obvious visual errors, fix and re-render.

IMPORTANT — reasoning style:
At the start of EVERY response, open with a <think> block. Think through what you \
observe (error, image, or initial prompt) and what action to take next. Finish the \
<think> block before making any tool calls or writing a final statement. Do not \
interleave thinking and tool calls within a single response.

Blender 5.1 rules (follow exactly):
1. Imports: bpy, math, os, sys — nothing else.
2. Parse --output <path> from sys.argv after the "--" separator.
3. Clear scene: bpy.ops.object.select_all(action="SELECT") then \
bpy.ops.object.delete().
4. Build the scene using ONLY Blender primitives: cube, plane, cylinder, sphere, cone.
   No external assets, no .blend file imports.
5. Materials:
   - mat.use_nodes = True
   - Access Principled BSDF via mat.node_tree.nodes["Principled BSDF"]
   - Only set "Base Color", "Roughness", "Metallic" inputs.
   - DO NOT use "Emission" as a direct Principled BSDF input (Blender 5.1 removed it).
     Use a separate Emission shader node if needed.
   - Never use mat.diffuse_color.
6. Lighting: add at least one AREA light (bpy.ops.object.light_add(type="AREA")).
7. Camera: position with a clear isometric-ish view; assign to \
bpy.context.scene.camera.
8. Render setup:
   - scene.render.engine = "BLENDER_EEVEE"
   - Resolution 800×600, PNG format
   - scene.render.filepath = os.path.abspath(output_path)
   - World background: access bpy.data.worlds["World"].node_tree and set the \
Background node's Color input. Do NOT call world.use_nodes = True.
   - Do NOT touch any other EEVEE settings (use_bloom, use_ssr, use_gtao, \
use_volumetric_lights, etc.). These properties do not exist in Blender 5.1's \
SceneEEVEE and will raise AttributeError.
9. Call bpy.ops.render.render(write_still=True).
10. Wrap everything in main(); call main() at the bottom.
11. Keep the script under 200 lines. Use helper functions for repeated patterns."""


# ------------------------------------------------------------------
# Rate limiting
# ------------------------------------------------------------------

class RateLimiter:
    """
    Thread-safe token-bucket rate limiter.

    Each call to acquire() blocks until a token is available, ensuring
    no more than `rpm` API requests are issued per minute across all threads.
    """

    def __init__(self, rpm: float):
        self._rate = rpm / 60.0          # tokens replenished per second
        self._tokens = self._rate        # start with one token's worth
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens += (now - self._last) * self._rate
                self._tokens = min(self._tokens, self._rate)  # cap at 1 token burst
                self._last = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait = (1.0 - self._tokens) / self._rate
            time.sleep(wait)


_NO_LIMIT = None  # sentinel: no rate limiting


# ------------------------------------------------------------------
# Core: single trajectory
# ------------------------------------------------------------------

def run_trajectory(
    prompt_entry: PromptEntry,
    client: anthropic.Anthropic,
    model: str,
    artifacts_dir: Path,
    max_turns: int = 8,
    thinking_budget: int = 8000,
    verbose: bool = False,
    rate_limiter: RateLimiter | None = None,
) -> Trajectory:
    """
    Run one agentic trajectory for a single prompt.

    Returns a Trajectory with raw and Qwen-reformatted messages.
    """
    traj = Trajectory(
        prompt=prompt_entry.prompt,
        prompt_meta=prompt_entry.meta(),
        model=model,
    )

    # Per-trajectory working directory (keeps scripts + renders for inspection)
    work_dir = artifacts_dir / traj.id
    env = BlenderEnvironment(work_dir)

    # Anthropic API messages (does not include system; that goes in the API call)
    api_messages: list[dict] = [
        {"role": "user", "content": f"Scene description: {prompt_entry.prompt}"},
    ]

    # Full trajectory including system (for storage)
    stored_messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *api_messages,
    ]

    _log(f"[{traj.id[:8]}] Starting: {prompt_entry.prompt[:60]!r}", verbose or True)

    try:
        for turn in range(max_turns):
            response = _call_claude(client, model, api_messages, thinking_budget, rate_limiter)

            # Serialise content blocks to plain dicts
            content_blocks = _serialise_content(response.content)

            assistant_msg = {"role": "assistant", "content": content_blocks}
            api_messages.append(assistant_msg)
            stored_messages.append(assistant_msg)
            traj.num_assistant_turns += 1

            tool_uses = [b for b in content_blocks if b.get("type") == "tool_use"]

            if response.stop_reason == "end_turn" or not tool_uses:
                _log(f"[{traj.id[:8]}] Model finished after {turn + 1} turn(s).", verbose)
                break

            # Execute all tool calls in this turn
            tool_result_blocks = []
            for tu in tool_uses:
                result = env.execute(tu["name"], tu["input"], tu["id"])
                _log(
                    f"[{traj.id[:8]}] Tool {tu['name']!r} → "
                    f"{'OK' if result.is_render_success else ('FAIL' if result.is_render_failure else 'done')}",
                    verbose,
                )

                if tu["name"] == "write_file" and not result.content[0]["text"].startswith("Python syntax"):
                    # Track last written script content from the tool call input
                    traj.final_script = tu["input"].get("content", "")

                if result.is_render_success:
                    traj.success = True
                    traj.num_render_attempts += 1
                    # Record path to the rendered PNG
                    render_pngs = sorted(work_dir.glob("render_*.png"))
                    if render_pngs:
                        traj.final_image_path = str(render_pngs[-1])
                elif result.is_render_failure:
                    traj.num_render_attempts += 1

                tool_result_blocks.append({
                    "type": "tool_result",
                    "tool_use_id": result.tool_use_id,
                    "content": result.content,
                })

            tool_msg = {"role": "user", "content": tool_result_blocks}
            api_messages.append(tool_msg)
            stored_messages.append(tool_msg)

    except Exception as exc:
        traj.error = f"{type(exc).__name__}: {exc}"
        _log(f"[{traj.id[:8]}] ERROR: {traj.error}", verbose or True)

    traj.messages_raw = stored_messages
    traj.messages_qwen = reformat_to_qwen_style(stored_messages)

    quality = "SUCCESS" if traj.success else ("ERROR" if traj.error else "NO_RENDER")
    _log(
        f"[{traj.id[:8]}] Done — {quality}, "
        f"{traj.num_assistant_turns} assistant turns, "
        f"{traj.num_render_attempts} render attempt(s).",
        verbose or True,
    )

    return traj


# ------------------------------------------------------------------
# Anthropic API call
# ------------------------------------------------------------------

def _call_claude(
    client: anthropic.Anthropic,
    model: str,
    messages: list[dict],
    thinking_budget: int,
    rate_limiter: RateLimiter | None = None,
    max_retries: int = 3,
) -> anthropic.types.Message:
    """Call Claude with extended thinking enabled. Retries on transient errors."""
    kwargs: dict = {
        "model": model,
        "max_tokens": max(thinking_budget + 8192, 16384),
        "system": SYSTEM_PROMPT,
        "tools": BlenderEnvironment.TOOLS,
        "messages": messages,
    }
    if thinking_budget > 0:
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}

    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        if rate_limiter is not None:
            rate_limiter.acquire()
        try:
            return client.messages.create(**kwargs)
        except anthropic.RateLimitError as exc:
            wait = 30 * attempt
            print(f"  [rate limit] waiting {wait}s (attempt {attempt}/{max_retries})…",
                  flush=True)
            time.sleep(wait)
            last_exc = exc
        except anthropic.APIStatusError as exc:
            if exc.status_code >= 500 and attempt < max_retries:
                time.sleep(5 * attempt)
                last_exc = exc
            else:
                raise

    raise last_exc  # type: ignore[misc]


# ------------------------------------------------------------------
# Content serialisation
# ------------------------------------------------------------------

def _serialise_content(content) -> list[dict]:
    """Convert Anthropic SDK content block objects to plain dicts."""
    blocks = []
    for block in content:
        btype = block.type
        if btype == "thinking":
            blocks.append({
                "type": "thinking",
                "thinking": block.thinking,
                # Keep signature for API round-trips (needed by Anthropic multi-turn)
                "signature": getattr(block, "signature", ""),
            })
        elif btype == "text":
            blocks.append({"type": "text", "text": block.text})
        elif btype == "tool_use":
            blocks.append({
                "type": "tool_use",
                "id": block.id,
                "name": block.name,
                "input": block.input,
            })
        else:
            # Preserve unknown types as-is
            blocks.append({"type": btype, "raw": str(block)})
    return blocks


# ------------------------------------------------------------------
# Resumption
# ------------------------------------------------------------------

def _load_done_prompts(output_path: Path) -> set[str]:
    """Return set of prompt strings already written to the output file."""
    done = set()
    if not output_path.exists():
        return done
    with output_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done.add(rec.get("prompt", ""))
            except json.JSONDecodeError:
                pass
    return done


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate Claude 4.6 multi-turn bpy trajectories for SFT distillation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input
    p.add_argument(
        "--prompts", default=str(HERE / "prompts.jsonl"),
        help="Path to prompts JSONL file.",
    )
    p.add_argument("--category", default=None, help="Filter by category.")
    p.add_argument("--domain", default=None, help="Filter by domain.")
    p.add_argument("--tags", nargs="+", default=None, help="Filter: prompts must have ALL these tags.")
    p.add_argument("--train", action="store_true", help='Only use prompts with "split": "train".')
    p.add_argument("--max-difficulty", type=int, default=None, help="Skip prompts harder than this.")
    p.add_argument("--min-difficulty", type=int, default=None, help="Skip prompts easier than this.")
    p.add_argument("--limit", type=int, default=None, help="Max number of trajectories to generate.")
    p.add_argument("--shuffle", action="store_true", help="Shuffle prompt order before filtering.")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for --shuffle.")

    # Model
    p.add_argument("--model", default="claude-sonnet-4-6", help="Anthropic model ID.")
    p.add_argument(
        "--thinking-budget", type=int, default=8000,
        help="Token budget for extended thinking (0 to disable).",
    )
    p.add_argument("--max-turns", type=int, default=8, help="Max agentic turns per trajectory.")

    # Output
    p.add_argument(
        "--run-dir", default=None,
        help="Directory for this run's output. "
             "Defaults to data/trajectories/run_<YYYYMMDD_HHMMSS>/. "
             "The directory will contain trajectories.jsonl and artifacts/.",
    )
    p.add_argument("--resume", action="store_true",
                   help="Resume an existing --run-dir (skip already-written prompts).")
    p.add_argument("--keep-artifacts", action="store_true",
                   help="Keep per-trajectory work dirs (scripts + PNGs). "
                        "Default: keep only on success.")

    # Execution
    p.add_argument("--workers", type=int, default=1,
                   help="Parallel worker threads.")
    p.add_argument(
        "--rpm", type=float, default=0,
        help="Max Anthropic API requests per minute across all workers (0 = unlimited). "
             "Each agentic turn counts as one request. "
             "Recommended when --workers > 1 to avoid 429s.",
    )
    p.add_argument("--verbose", action="store_true", help="Extra logging.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print prompts that would be processed, then exit.")

    return p


def main() -> None:
    args = build_parser().parse_args()

    # Validate Blender is available
    import shutil as _shutil
    if not _shutil.which("blender"):
        print("ERROR: 'blender' not found in PATH. Install Blender and ensure it's on PATH.",
              file=sys.stderr)
        sys.exit(1)

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        if args.resume:
            print("ERROR: --resume requires --run-dir pointing to an existing run.",
                  file=sys.stderr)
            sys.exit(1)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = HERE / "trajectories" / f"run_{timestamp}"

    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / "trajectories.jsonl"
    artifacts_dir = run_dir / "artifacts"

    loader = PromptLoader(
        path=Path(args.prompts),
        category=args.category,
        domain=args.domain,
        tags=args.tags,
        split="train" if args.train else None,
        max_difficulty=args.max_difficulty,
        min_difficulty=args.min_difficulty,
        limit=args.limit,
        shuffle=args.shuffle,
        seed=args.seed,
    )

    prompts = list(loader)

    if args.resume:
        done = _load_done_prompts(output_path)
        before = len(prompts)
        prompts = [p for p in prompts if p.prompt not in done]
        print(f"[resume] Skipping {before - len(prompts)} already-done prompt(s).")

    if args.dry_run:
        print(f"Would generate {len(prompts)} trajectory/ies with model={args.model!r}:")
        for i, p in enumerate(prompts, 1):
            print(f"  {i:3d}. [{p.category}/{p.difficulty:+d}] {p.prompt[:80]}")
        return

    rate_limiter: RateLimiter | None = None
    if args.rpm > 0:
        rate_limiter = RateLimiter(rpm=args.rpm)
    elif args.workers > 1:
        print(
            f"WARNING: --workers {args.workers} with no --rpm set. "
            "All workers will issue API calls simultaneously and may hit rate limits. "
            "Consider e.g. --rpm 50.",
            file=sys.stderr,
        )

    rpm_note = f", rpm={args.rpm}" if args.rpm > 0 else ""
    print(f"Generating {len(prompts)} trajectory/ies — model={args.model!r}, "
          f"workers={args.workers}{rpm_note}, run_dir={run_dir}")

    client = anthropic.Anthropic(api_key=api_key)

    def _process(entry: PromptEntry) -> Trajectory:
        return run_trajectory(
            prompt_entry=entry,
            client=client,
            model=args.model,
            artifacts_dir=artifacts_dir,
            max_turns=args.max_turns,
            thinking_budget=args.thinking_budget,
            verbose=args.verbose,
            rate_limiter=rate_limiter,
        )

    with TrajectoryWriter(output_path) as writer:
        if args.workers == 1:
            for entry in prompts:
                traj = _process(entry)
                writer.write(traj)
                _maybe_clean(traj, artifacts_dir, args.keep_artifacts)
        else:
            futures = {}
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                for entry in prompts:
                    fut = pool.submit(_process, entry)
                    futures[fut] = entry

                for fut in as_completed(futures):
                    try:
                        traj = fut.result()
                        writer.write(traj)
                        _maybe_clean(traj, artifacts_dir, args.keep_artifacts)
                    except Exception as exc:
                        entry = futures[fut]
                        print(f"[FATAL] {entry.prompt[:60]!r}: {exc}", file=sys.stderr)

    print(f"Done. Run output in {run_dir}")


def _maybe_clean(traj: Trajectory, artifacts_dir: Path, keep: bool) -> None:
    """Remove per-trajectory work dir if not keeping artifacts and render failed."""
    if keep:
        return
    work_dir = artifacts_dir / traj.id
    if not traj.success and work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)


def _log(msg: str, enabled: bool) -> None:
    if enabled:
        print(msg, flush=True)


if __name__ == "__main__":
    main()
