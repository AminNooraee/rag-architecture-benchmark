"""Tests for dataset I/O utilities."""

from pathlib import Path
import json

from rag_benchmark.datasets.io import write_json, write_jsonl


def test_write_json(tmp_path: Path) -> None:
    data = {"key": "value", "list": [1, 2, 3]}
    p = tmp_path / "out.json"
    write_json(p, data)
    loaded = json.loads(p.read_text(encoding="utf-8"))
    assert loaded == data


def test_write_jsonl_preserves_persian(tmp_path: Path) -> None:
    persian = "متن فارسی با نیم‌فاصله\u200c و نقاط."
    items = [{"text": persian}]
    p = tmp_path / "out.jsonl"
    write_jsonl(p, items)
    line = p.read_text(encoding="utf-8").strip()
    obj = json.loads(line)
    assert obj["text"] == persian
