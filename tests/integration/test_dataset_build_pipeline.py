"""Integration test for the full dataset build pipeline."""

import json
from pathlib import Path

import yaml

from rag_benchmark.cli.dataset import build_from_config


def test_build_pipeline(tmp_path: Path) -> None:
    cfg_path = Path("tests/fixtures/dataset_config.yaml")
    cfg_text = cfg_path.read_text(encoding="utf-8")
    cfg_text = cfg_text.replace("{tmp_output}", str(tmp_path / "out"))
    cfg = yaml.safe_load(cfg_text)
    build_from_config(cfg)
    out_dir = Path(cfg["output"]["directory"])
    for name in ["documents.jsonl", "questions.jsonl", "manifest.json", "statistics.json"]:
        assert (out_dir / name).is_file()
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    # Expected counts: 1 pquad doc + 2 wikipedia docs = 3 documents
    assert manifest["document_count"] == 3
    # Expected questions: 2 pquad + 1 mhqa = 3
    assert manifest["question_count"] == 3
    # Verify Persian text survived in at least one document
    docs = []
    for line in (out_dir / "documents.jsonl").read_text(encoding="utf-8").splitlines():
        docs.append(json.loads(line))
    assert any("متن پاراگراف" in d["text"] for d in docs)
