"""CLI for building benchmark datasets.

Only the ``build`` sub‑command is required for the test suite.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rag_benchmark.datasets.builder import BenchmarkDatasetBuilder
from rag_benchmark.datasets.config import RootConfig
from rag_benchmark.datasets.custom_wiki_questions import CustomWikiQuestionLoader
from rag_benchmark.datasets.persian_mhqa import PersianMHQALoader
from rag_benchmark.datasets.pquad import PQuADLoader
from rag_benchmark.datasets.wikipedia import WikipediaLoader

from typing import Any, TYPE_CHECKING


def _load_config(path: Path) -> RootConfig:
    # Load YAML with duplicate‑key detection and parse into typed config
    from rag_benchmark.datasets.config import load_config

    return load_config(path)


def build_from_config(cfg: RootConfig | dict[str, Any]) -> None:
    if not isinstance(cfg, RootConfig):
        cfg = RootConfig.parse_obj(cfg)
    # Use typed configuration to drive builder construction
    output_dir = cfg.output.directory
    builder = BenchmarkDatasetBuilder(
        dataset_name=cfg.dataset.name,
        dataset_version=cfg.dataset.version,
    )
    sources_cfg = cfg.sources
    if sources_cfg.pquad and sources_cfg.pquad.enabled:
        assert sources_cfg.pquad.path is not None
        builder.add_source(
            PQuADLoader(
                sources_cfg.pquad.path,
                sources_cfg.pquad.version,
            )
        )
    if sources_cfg.persian_mhqa and sources_cfg.persian_mhqa.enabled:
        assert sources_cfg.persian_mhqa.path is not None
        builder.add_source(
            PersianMHQALoader(
                sources_cfg.persian_mhqa.path,
                sources_cfg.persian_mhqa.version,
            )
        )
    if sources_cfg.wikipedia and sources_cfg.wikipedia.enabled:
        assert sources_cfg.wikipedia.path is not None
        builder.add_source(
            WikipediaLoader(
                sources_cfg.wikipedia.path,
                sources_cfg.wikipedia.version,
            )
        )
    if sources_cfg.custom_wikipedia_questions and sources_cfg.custom_wikipedia_questions.enabled:
        assert sources_cfg.custom_wikipedia_questions.path is not None
        builder.add_source(
            CustomWikiQuestionLoader(
                sources_cfg.custom_wikipedia_questions.path,
                sources_cfg.custom_wikipedia_questions.version,
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
