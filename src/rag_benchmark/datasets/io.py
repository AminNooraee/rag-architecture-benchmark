"""Simple atomic JSON/JSONL writers used by the dataset builder."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any


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
    json_str = json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)
    _atomic_write(Path(path), json_str + "\n")


def write_jsonl(path: str | Path, items: Iterable[Any]) -> None:
    lines = []
    for item in items:
        data = item.model_dump(mode="json") if hasattr(item, "model_dump") else item
        lines.append(json.dumps(data, ensure_ascii=False, sort_keys=True))
    _atomic_write(Path(path), "\n".join(lines) + "\n")
