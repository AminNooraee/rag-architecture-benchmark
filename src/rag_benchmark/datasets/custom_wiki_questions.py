"""Custom Wikipedia benchmark question loader.

Loads manually authored question records from a local JSONL file.  The format must
match the canonical :class:`BenchmarkQuestion` fields.  No LLM generation is
performed.
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


class CustomWikiQuestionLoader(DatasetLoader):
    def __init__(self, source_path: str | Path, source_version: str = "") -> None:
        self._path = Path(source_path)
        if not self._path.is_file():
            raise FileNotFoundError(f"Custom questions file not found: {self._path}")
        self._source_version = source_version

    @property
    def source_name(self) -> str:
        return "wikipedia_custom_questions"

    @property
    def source_version(self) -> str:
        return self._source_version

    def _iter_lines(self) -> Iterable[dict[str, Any]]:
        with self._path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

    def load_documents(self) -> Iterable[BenchmarkDocument]:
        return []

    def load_questions(self) -> Iterable[BenchmarkQuestion]:
        for obj in self._iter_lines():
            # Validate required fields minimally; Pydantic will enforce further
            qid_raw = obj.get("question_id")
            if not qid_raw:
                continue
            qid = deterministic_id("custom-q", qid_raw)
            # Map any supporting document placeholders to Wikipedia IDs if needed
            raw_support = obj.get("supporting_document_ids", [])
            supporting = [
                deterministic_id("wiki-doc", doc_id)
                if not doc_id.startswith("wiki-doc-")
                else doc_id
                for doc_id in raw_support
            ]
            yield BenchmarkQuestion(
                question_id=qid,
                question=obj.get("question", ""),
                question_type=QuestionType(obj.get("question_type", "direct_factual")),
                difficulty=DifficultyLevel(obj.get("difficulty", "medium")),
                scope=QuestionScope(obj.get("scope", "local")),
                reasoning_hops=int(obj.get("reasoning_hops", 1)),
                gold_answer=obj.get("gold_answer", ""),
                acceptable_answers=obj.get("acceptable_answers", []),
                supporting_document_ids=supporting,
                supporting_evidence=obj.get("supporting_evidence", []),
                answerable=bool(obj.get("answerable", True)),
                dataset_source=self.source_name,
                query_metadata=obj.get("query_metadata", {}),
                metadata=obj.get("metadata", {}),
            )
