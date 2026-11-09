"""PQuAD dataset loader.

Parses a SQuAD‑style Persian QA JSON file and yields canonical
:class:`BenchmarkDocument` and :class:`BenchmarkQuestion` objects.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from rag_benchmark.datasets.base import DatasetLoader
from rag_benchmark.datasets.ids import deterministic_id
from rag_benchmark.schemas.document import BenchmarkDocument, DocumentSection
from rag_benchmark.schemas.question import (
    BenchmarkQuestion,
    DifficultyLevel,
    QuestionScope,
    QuestionType,
)


class PQuADLoader(DatasetLoader):
    """Load PQuAD JSON files into the benchmark models.

    Parameters
    ----------
    source_path:
        Path to the raw ``pquad.json`` file.
    source_version:
        Optional version string.
    """

    def __init__(self, source_path: str | Path, source_version: str = "") -> None:
        self._path = Path(source_path)
        if not self._path.is_file():
            raise FileNotFoundError(f"PQuAD source file not found: {self._path}")
        self._source_version = source_version

    @property
    def source_name(self) -> str:
        return "pquad"

    @property
    def source_version(self) -> str:
        return self._source_version

    def _load_raw(self) -> dict[str, Any]:
        with self._path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("PQuAD source file does not contain a JSON object")
        if "data" not in data:
            raise ValueError(
                f"PQuAD source at {self._path} is missing the required 'data' collection"
            )
        if not isinstance(data["data"], list):
            raise ValueError(f"'data' must be a list in PQuAD source at {self._path}")
        return data

    def load_documents(self) -> Iterable[BenchmarkDocument]:
        raw = self._load_raw()
        for article in raw["data"]:
            title = article.get("title", "")
            doc_id = deterministic_id("pquad-doc", title)
            sections: list[DocumentSection] = []
            for para_idx, para in enumerate(article.get("paragraphs", [])):
                ctx = para.get("context", "")
                sec_id = deterministic_id("pquad-sec", doc_id, str(para_idx))
                sections.append(
                    DocumentSection(
                        section_id=sec_id,
                        heading="",
                        text=ctx,
                        order=para_idx,
                    )
                )
            yield BenchmarkDocument(
                document_id=doc_id,
                title=title or "(no title)",
                text="\n\n".join([s.text for s in sections]),
                language="fa",
                source=self.source_name,
                url=None,
                sections=sections,
                metadata={"native": "pquad"},
            )

    def load_questions(self) -> Iterable[BenchmarkQuestion]:
        raw = self._load_raw()
        for article in raw["data"]:
            title = article.get("title", "")
            doc_id = deterministic_id("pquad-doc", title)
            for para_idx, para in enumerate(article.get("paragraphs", [])):
                for qa in para.get("qas", []):
                    qid = qa.get("id")
                    if not qid:
                        raise ValueError(
                            f"Missing question id in PQuAD source at article '{title}' "
                            f"paragraph index {para_idx}"
                        )
                    question_text = qa.get("question", "")
                    if not question_text:
                        raise ValueError(
                            f"Missing question text for question id '{qid}' in PQuAD source"
                        )
                    is_impossible = qa.get("is_impossible", False)
                    # Determine question type
                    if is_impossible:
                        qtype = QuestionType.UNANSWERABLE
                    else:
                        qtype = QuestionType.DIRECT_FACTUAL
                    answers = qa.get("answers", [])
                    if not isinstance(answers, list):
                        raise ValueError(
                            f"'answers' must be a list for question id '{qid}' in PQuAD source "
                            f"at article '{title}' paragraph index {para_idx}"
                        )
                    for ans in answers:
                        if not isinstance(ans, dict) or "text" not in ans:
                            raise ValueError(
                                f"Malformed answer entry for question id '{qid}' in PQuAD source "
                                f"at article '{title}' paragraph index {para_idx}"
                            )
                    gold = answers[0]["text"] if answers else ""
                    accept = [a["text"] for a in answers[1:]] if len(answers) > 1 else []
                    qid_canonical = deterministic_id("pquad-q", qid)
                    yield BenchmarkQuestion(
                        question_id=qid_canonical,
                        question=question_text,
                        question_type=qtype,
                        difficulty=DifficultyLevel.MEDIUM,
                        scope=QuestionScope.LOCAL,
                        reasoning_hops=1,
                        gold_answer=gold,
                        acceptable_answers=accept,
                        supporting_document_ids=[doc_id] if not is_impossible else [],
                        supporting_evidence=[],
                        answerable=not is_impossible,
                        dataset_source=self.source_name,
                        query_metadata={},
                        metadata={"native_id": qid},
                    )
