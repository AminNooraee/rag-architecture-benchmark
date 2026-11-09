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

    def _iter_lines(self) -> Iterable[tuple[int, dict[str, Any]]]:
        with self._path.open("r", encoding="utf-8") as f:
            for line_no, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Invalid JSON at line {line_no} in {self._path}: {e.msg}"
                    ) from None
                if not isinstance(obj, dict):
                    raise ValueError(
                        f"JSON record at line {line_no} in {self._path} "
                        "does not contain an object mapping"
                    )
                yield line_no, obj

    def load_documents(self) -> Iterable[BenchmarkDocument]:
        return []

    def load_questions(self) -> Iterable[BenchmarkQuestion]:
        for line_no, obj in self._iter_lines():
            # Validate required fields; raise on malformed records
            qid_raw = obj.get("question_id")
            if not qid_raw:
                raise ValueError(
                    f"Missing 'question_id' in custom Wiki questions source at line {line_no}"
                )
            question_text = obj.get("question", "")
            if not question_text:
                raise ValueError(
                    f"Missing 'question' for question_id '{qid_raw}' "
                    f"in custom Wiki questions source at line {line_no}"
                )
            qid = deterministic_id("custom-q", qid_raw)
            # Validate supporting_document_ids type
            raw_support = obj.get("supporting_document_ids", [])
            if not isinstance(raw_support, list):
                raise ValueError(
                    f"'supporting_document_ids' must be a list for "
                    f"question_id '{qid_raw}' at line {line_no}"
                )
            for sd_idx, doc_id in enumerate(raw_support):
                if not isinstance(doc_id, str) or not doc_id:
                    raise ValueError(
                        f"'supporting_document_ids' entry {sd_idx} must be a non-empty string "
                        f"for question_id '{qid_raw}' at line {line_no}"
                    )
            supporting = [
                deterministic_id("wiki-doc", doc_id)
                if not doc_id.startswith("wiki-doc-")
                else doc_id
                for doc_id in raw_support
            ]
            hops_raw = obj.get("reasoning_hops", 1)
            if isinstance(hops_raw, bool) or not isinstance(hops_raw, int):
                raise ValueError(
                    f"'reasoning_hops' must be an integer for question_id '{qid_raw}' "
                    f"at line {line_no}"
                )
            yield BenchmarkQuestion(
                question_id=qid,
                question=question_text,
                question_type=QuestionType(obj.get("question_type", "direct_factual")),
                difficulty=DifficultyLevel(obj.get("difficulty", "medium")),
                scope=QuestionScope(obj.get("scope", "local")),
                reasoning_hops=hops_raw,
                gold_answer=obj.get("gold_answer", ""),
                acceptable_answers=obj.get("acceptable_answers", []),
                supporting_document_ids=supporting,
                supporting_evidence=obj.get("supporting_evidence", []),
                answerable=bool(obj.get("answerable", True)),
                dataset_source=self.source_name,
                query_metadata=obj.get("query_metadata", {}),
                metadata=obj.get("metadata", {}),
            )
