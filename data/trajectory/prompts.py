"""
prompts.py — Load and filter prompts for trajectory generation.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass
class PromptEntry:
    prompt: str
    category: str = ""
    domain: str = ""
    tags: list[str] = field(default_factory=list)
    difficulty: int = 0
    split: str = ""

    def meta(self) -> dict:
        return {
            "category": self.category,
            "domain": self.domain,
            "tags": self.tags,
            "difficulty": self.difficulty,
            "split": self.split,
        }


class PromptLoader:
    """
    Iterate over prompts from a JSONL file with optional filtering.

    Each line must have at minimum a "prompt" key.
    Optional keys: category, domain, tags, difficulty.

    Args:
        path:           Path to prompts JSONL file.
        category:       Only yield prompts matching this category (exact).
        domain:         Only yield prompts matching this domain (exact).
        tags:           Only yield prompts that have ALL of these tags.
        max_difficulty: Skip prompts with difficulty > this value.
        min_difficulty: Skip prompts with difficulty < this value.
        limit:          Stop after yielding this many prompts.
        shuffle:        Randomise order before filtering/limiting.
        seed:           Random seed (only used when shuffle=True).
    """

    def __init__(
        self,
        path: Path,
        category: str | None = None,
        domain: str | None = None,
        tags: list[str] | None = None,
        split: str | None = None,
        max_difficulty: int | None = None,
        min_difficulty: int | None = None,
        limit: int | None = None,
        shuffle: bool = False,
        seed: int = 42,
    ):
        self.path = Path(path)
        self.category = category
        self.domain = domain
        self.tags = tags or []
        self.split = split
        self.max_difficulty = max_difficulty
        self.min_difficulty = min_difficulty
        self.limit = limit
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[PromptEntry]:
        entries = self._load_all()

        if self.shuffle:
            import random
            rng = random.Random(self.seed)
            rng.shuffle(entries)

        count = 0
        for entry in entries:
            if self.category and entry.category != self.category:
                continue
            if self.domain and entry.domain != self.domain:
                continue
            if self.tags and not all(t in entry.tags for t in self.tags):
                continue
            if self.split and entry.split != self.split:
                continue
            if self.max_difficulty is not None and entry.difficulty > self.max_difficulty:
                continue
            if self.min_difficulty is not None and entry.difficulty < self.min_difficulty:
                continue

            yield entry
            count += 1
            if self.limit is not None and count >= self.limit:
                break

    def __len__(self) -> int:
        """Count matching prompts (loads full file)."""
        return sum(1 for _ in self)

    def _load_all(self) -> list[PromptEntry]:
        entries = []
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                entries.append(PromptEntry(
                    prompt=data["prompt"],
                    category=data.get("category", ""),
                    domain=data.get("domain", ""),
                    tags=data.get("tags", []),
                    difficulty=data.get("difficulty", 0),
                    split=data.get("split", ""),
                ))
        return entries
