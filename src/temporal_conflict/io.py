"""JSONL I/O for PTC instances and result records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

from temporal_conflict.schema import PTCInstance


def load_jsonl(path: Path | str) -> Iterator[PTCInstance]:
    """Yield PTCInstance objects from a JSONL file."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield PTCInstance.from_dict(json.loads(line))


def load_directory(
    root: Path | str,
    pattern: str = "*.jsonl",
    require_passed_filter: bool = True,
) -> list[PTCInstance]:
    """Load all matching JSONL files under `root` into a flat list.

    By default drops records with `passed_filter == False`. Records that
    omit the field (older JSONL formats) are kept.
    """
    root = Path(root)
    out: list[PTCInstance] = []
    for path in sorted(root.glob(pattern)):
        for inst in load_jsonl(path):
            if require_passed_filter and inst.passed_filter is False:
                continue
            out.append(inst)
    return out


def write_jsonl(path: Path | str, records: Iterable[dict]) -> None:
    """Write an iterable of JSON-serializable dicts as JSONL."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
