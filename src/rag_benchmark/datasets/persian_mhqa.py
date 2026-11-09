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
        if "questions" not in raw:
            raise ValueError(
                f"PersianMHQA source at {self._path} is missing the required 'questions' collection"
            )
        questions = raw["questions"]
        if not isinstance(questions, list):
            raise ValueError(f"'questions' must be a list in PersianMHQA source at {self._path}")
        for idx, entry in enumerate(questions):
            if not isinstance(entry, dict):
                raise ValueError(f"Malformed question record at index {idx} in PersianMHQA source")
            native_id = entry.get("question_id")
            if not native_id:
                raise ValueError(f"Missing question_id in PersianMHQA source at index {idx}")
            qid = deterministic_id("pmhqa-q", native_id)
            question_text = entry.get("question", "")
            if not question_text:
                raise ValueError(
                    f"Missing question text in PersianMHQA source for "
                    f"question_id '{native_id}' at index {idx}"
                )
            src_type = entry.get("type", "")
            # Determine question type
            if src_type == "multi_hop":
                qtype = QuestionType.MULTI_HOP
            else:
                qtype = QuestionType.DIRECT_FACTUAL
            hops_raw = entry.get("reasoning_hops", 1)
            if isinstance(hops_raw, bool) or not isinstance(hops_raw, int):
                raise ValueError(
                    f"'reasoning_hops' must be an integer in PersianMHQA source for "
                    f"question_id '{native_id}' at index {idx}"
                )
            if hops_raw < 1:
                raise ValueError(
                    f"'reasoning_hops' must be at least 1 in PersianMHQA source for "
                    f"question_id '{native_id}' at index {idx}"
                )
            hops = hops_raw
            # Map supporting document placeholders to deterministic Wikipedia IDs
            raw_supporting = entry.get("supporting_documents", [])
            if not isinstance(raw_supporting, list):
                raise ValueError(
                    f"supporting_documents must be a list in PersianMHQA source for "
                    f"question_id '{native_id}' at index {idx}"
                )
            for sd_idx, doc_id in enumerate(raw_supporting):
                if not isinstance(doc_id, str) or not doc_id:
                    raise ValueError(
                        f"supporting_documents entry {sd_idx} must be a non-empty string in "
                        f"PersianMHQA source for question_id '{native_id}' at index {idx}"
                    )
            supporting = [deterministic_id("wiki-doc", doc_id) for doc_id in raw_supporting]
            yield BenchmarkQuestion(
                question_id=qid,
                question=question_text,
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
