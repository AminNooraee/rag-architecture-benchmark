"""Simple atomic JSON/JSONL writers used by the dataset builder."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from rag_benchmark.schemas.strict_json import strict_json_dumps


def _atomic_write(path: Path, data: str) -> None:
    dir_path = path.parent
    fd, tmp_path = tempfile.mkstemp(dir=dir_path, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def write_json(path: str | Path, obj: Any) -> None:
    # Strict JSON serialization without NaN/Infinity, sorted keys for deterministic output
    json_str = strict_json_dumps(obj, indent=2)
    _atomic_write(Path(path), json_str + "\n")


def write_jsonl(path: str | Path, items: Iterable[Any]) -> None:
    # Serialize each item with strict JSON and sorted keys
    lines = []
    for item in items:
        data = item.model_dump(mode="json") if hasattr(item, "model_dump") else item
        lines.append(strict_json_dumps(data))
    _atomic_write(Path(path), "\n".join(lines) + "\n")
