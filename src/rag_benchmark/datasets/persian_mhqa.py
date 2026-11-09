"""PersianMHQA loader.

Loads a minimal JSON fixture with multi‑hop QA entries and maps them to the
canonical :class:`BenchmarkQuestion` model.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from rag_benchmark.datasets.base import DatasetLoader
from rag_benchmark.datasets.ids import deterministic_id
from rag_benchmark.schemas.document import BenchmarkDocument
from rag_benchmark.schemas.question import (
    BenchmarkQuestion,
    DifficultyLevel,
    QuestionScope,
    QuestionType,
)


class PersianMHQALoader(DatasetLoader):
    def __init__(self, source_path: str | Path, source_version: str = "") -> None:
        self._path = Path(source_path)
        if not self._path.is_file():
            raise FileNotFoundError(f"PersianMHQA source file not found: {self._path}")
        self._source_version = source_version

    @property
    def source_name(self) -> str:
        return "persian_mhqa"

    @property
    def source_version(self) -> str:
        return self._source_version

    def _load_raw(self) -> dict[str, Any]:
        with self._path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("PersianMHQA source file does not contain a JSON object")
        return data

    def load_documents(self) -> Iterable[BenchmarkDocument]:
        return []

    def load_questions(self) -> Iterable[BenchmarkQuestion]:
        raw = self._load_raw()
        for entry in raw.get("questions", []):
            native_id = entry.get("question_id")
            qid = deterministic_id("pmhqa-q", native_id)
            src_type = entry.get("type", "")
            # Determine question type
            if src_type == "multi_hop":
                qtype = QuestionType.MULTI_HOP
            else:
                qtype = QuestionType.DIRECT_FACTUAL
            hops = int(entry.get("reasoning_hops", 1))
            # Map supporting document placeholders to deterministic Wikipedia IDs
            raw_supporting = entry.get("supporting_documents", [])
            supporting = [deterministic_id("wiki-doc", doc_id) for doc_id in raw_supporting]
            yield BenchmarkQuestion(
                question_id=qid,
                question=entry.get("question", ""),
                question_type=qtype,
                difficulty=DifficultyLevel.MEDIUM,
                scope=QuestionScope.MULTI_DOCUMENT,
                reasoning_hops=hops,
                gold_answer=entry.get("answer", ""),
                acceptable_answers=[],
                supporting_document_ids=supporting,
                supporting_evidence=[],
                answerable=True,
                dataset_source=self.source_name,
                query_metadata={},
                metadata={"native_id": native_id},
            )
