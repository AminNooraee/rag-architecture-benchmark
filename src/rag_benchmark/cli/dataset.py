"""CLI for building benchmark datasets.

Only the ``build`` sub‑command is required for the test suite.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

from rag_benchmark.datasets.builder import BenchmarkDatasetBuilder
from rag_benchmark.datasets.custom_wiki_questions import CustomWikiQuestionLoader
from rag_benchmark.datasets.persian_mhqa import PersianMHQALoader
from rag_benchmark.datasets.pquad import PQuADLoader
from rag_benchmark.datasets.wikipedia import WikipediaLoader


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("Configuration file must contain a mapping")
    return data


def build_from_config(cfg: dict[str, Any]) -> None:
    output_dir = Path(cfg.get("output", {}).get("directory", "data/processed/dataset"))
    builder = BenchmarkDatasetBuilder()
    sources_cfg = cfg.get("sources", {})
    pquad_cfg = sources_cfg.get("pquad", {})
    if pquad_cfg.get("enabled", False):
        builder.add_source(PQuADLoader(pquad_cfg["path"], pquad_cfg.get("version", "")))
    mhqa_cfg = sources_cfg.get("persian_mhqa", {})
    if mhqa_cfg.get("enabled", False):
        builder.add_source(PersianMHQALoader(mhqa_cfg["path"], mhqa_cfg.get("version", "")))
    wiki_cfg = sources_cfg.get("wikipedia", {})
    if wiki_cfg.get("enabled", False):
        builder.add_source(WikipediaLoader(wiki_cfg["path"], wiki_cfg.get("version", "")))
    custom_q_cfg = sources_cfg.get("custom_wikipedia_questions", {})
    if custom_q_cfg.get("enabled", False):
        builder.add_source(
            CustomWikiQuestionLoader(
                custom_q_cfg["path"],
                custom_q_cfg.get("version", ""),
            )
        )
    builder.build(output_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag-benchmark-dataset")
    subparsers = parser.add_subparsers(dest="cmd", required=True)
    build_parser = subparsers.add_parser("build", help="Build dataset from configuration file")
    build_parser.add_argument("--config", required=True, help="Path to YAML configuration")
    args = parser.parse_args(argv)
    if args.cmd == "build":
        cfg = _load_config(Path(args.config))
        build_from_config(cfg)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
