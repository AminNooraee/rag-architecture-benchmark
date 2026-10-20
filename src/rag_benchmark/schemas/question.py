"""Canonical benchmark question schema and adapter query input."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from rag_benchmark.schemas.document import _check_json_serializable
from rag_benchmark.schemas.evidence import RetrievedEvidence


class QuestionType(StrEnum):
    """Categorization of questions by benchmark query complexity."""

    DIRECT_FACTUAL = "direct_factual"
    DETAILED_FACTUAL = "detailed_factual"
    MULTI_HOP = "multi_hop"
    RELATIONSHIP = "relationship"
    CROSS_DOCUMENT = "cross_document"
    GLOBAL_SUMMARY = "global_summary"
    HIERARCHICAL = "hierarchical"
    UNANSWERABLE = "unanswerable"


class QuestionScope(StrEnum):
    """Information boundary scope required to answer the question."""

    LOCAL = "local"
    MULTI_DOCUMENT = "multi_document"
    GLOBAL = "global"


class DifficultyLevel(StrEnum):
    """Question difficulty classification."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class AdapterQueryInput(BaseModel):
    """Input payload delivered to a RAG adapter's query() method.

    CRITICAL ARCHITECTURAL FAIRNESS RULE:
    The adapter's normal query() invocation must NEVER receive gold answers,
    acceptable answers, or ground-truth supporting evidence.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    question_id: str = Field(
        ...,
        min_length=1,
        description="Unique question identifier.",
    )
    question: str = Field(
        ...,
        min_length=1,
        description="The Persian or multilingual question prompt.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Safe query-time metadata (ground-truth excluded).",
    )

    @field_validator("question_id")
    @classmethod
    def validate_question_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("question_id cannot be empty or whitespace only")
        return v

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("question cannot be empty or whitespace only")
        return v

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, v: dict[str, Any]) -> dict[str, Any]:
        return _check_json_serializable(v, "metadata")


class BenchmarkQuestion(BaseModel):
    """Full ground-truth evaluation model for a benchmark question.

    Used by the dataset loader and evaluation metrics engine.
    Never passed directly to an adapter's query() execution path.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    question_id: str = Field(
        ...,
        min_length=1,
        description="Unique identifier for the benchmark question.",
    )
    question: str = Field(
        ...,
        min_length=1,
        description="The benchmark question text in Persian Unicode.",
    )
    question_type: QuestionType = Field(
        ...,
        description="Type of question according to retrieval and reasoning requirements.",
    )
    difficulty: DifficultyLevel | str = Field(
        default=DifficultyLevel.MEDIUM,
        description="Subjective or calibrated difficulty level.",
    )
    scope: QuestionScope = Field(
        ...,
        description="Scope of information needed to answer (local, multi_document, global).",
    )
    reasoning_hops: int = Field(
        default=1,
        ge=0,
        description="Number of logical reasoning hops required (>= 0).",
    )
    gold_answer: str = Field(
        default="",
        description="Ground-truth reference answer in Persian.",
    )
    acceptable_answers: list[str] = Field(
        default_factory=list,
        description="Alternative acceptable variations of the answer.",
    )
    supporting_document_ids: list[str] = Field(
        default_factory=list,
        description="List of document IDs containing necessary ground-truth facts.",
    )
    supporting_evidence: list[RetrievedEvidence] = Field(
        default_factory=list,
        description="Structured ground-truth evidence snippets or entities.",
    )
    answerable: bool = Field(
        default=True,
        description="Whether the question is answerable given the corpus.",
    )
    dataset_source: str = Field(
        default="",
        description="Source dataset or benchmark subset identifier.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Evaluation metadata (e.g. human annotator ID, license).",
    )

    @field_validator("question_id")
    @classmethod
    def validate_question_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("question_id cannot be empty or whitespace only")
        return v

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("question cannot be empty or whitespace only")
        return v

    @field_validator("reasoning_hops")
    @classmethod
    def validate_reasoning_hops(cls, v: int) -> int:
        if v < 0:
            raise ValueError("reasoning_hops must be non-negative (>= 0)")
        return v

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, v: dict[str, Any]) -> dict[str, Any]:
        return _check_json_serializable(v, "metadata")

    def to_query_input(self) -> AdapterQueryInput:
        """Strip evaluation ground-truth and generate adapter-safe query input.

        This guarantees that gold_answer, acceptable_answers, and supporting
        evidence cannot accidentally leak into the RAG adapter execution.
        """
        # Filter out ground-truth or evaluation keys from metadata if present
        safe_meta = {
            k: v
            for k, v in self.metadata.items()
            if not k.startswith("eval_") and k not in {"gold", "ground_truth", "answers"}
        }
        return AdapterQueryInput(
            question_id=self.question_id,
            question=self.question,
            metadata=safe_meta,
        )
