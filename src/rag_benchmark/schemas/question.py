"""Canonical benchmark question schema and adapter query input."""

from __future__ import annotations

import copy
from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from rag_benchmark.errors.models import GroundTruthLeakageError
from rag_benchmark.schemas.evidence import RetrievedEvidence
from rag_benchmark.schemas.strict_json import validate_strict_json_dict

# Prohibited ground-truth fields that must NEVER appear in query_metadata
PROHIBITED_GROUND_TRUTH_KEYS = {
    "gold_answer",
    "reference_answer",
    "acceptable_answers",
    "ground_truth",
    "gold_evidence",
    "supporting_facts",
    "supporting_document_ids",
    "expected_entities",
    "answer_key",
    "reference_documents",
}


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

    model_config = ConfigDict(frozen=True, extra="forbid", validate_assignment=True)

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
        description="Strict JSON-native query-time metadata (ground-truth strictly excluded).",
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
        return validate_strict_json_dict(v, "metadata")


class BenchmarkQuestion(BaseModel):
    """Full ground-truth evaluation model for a benchmark question.

    Used by the dataset loader and evaluation metrics engine.
    Never passed directly to an adapter's query() execution path.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", validate_assignment=True)

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
        description=(
            "Structured ground-truth evidence snippets or entities. For unanswerable "
            "questions, may optionally contain background or distractor corpus passages."
        ),
    )
    answerable: bool = Field(
        default=True,
        description="Whether the question is answerable given the corpus.",
    )
    dataset_source: str = Field(
        default="",
        description="Source dataset or benchmark subset identifier.",
    )
    query_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Explicitly allowlisted metadata safe to pass to RAG adapters at query time.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Evaluation-only metadata (annotators, metrics). Never sent to adapter.",
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

    @field_validator("query_metadata")
    @classmethod
    def validate_query_metadata(cls, v: dict[str, Any]) -> dict[str, Any]:
        return validate_strict_json_dict(v, "query_metadata")

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, v: dict[str, Any]) -> dict[str, Any]:
        return validate_strict_json_dict(v, "metadata")

    @model_validator(mode="after")
    def validate_question_semantics(self) -> Self:
        """Validate logical coherence across benchmark question fields."""
        # Unanswerable question semantics
        if self.question_type == QuestionType.UNANSWERABLE:
            if self.answerable:
                raise ValueError("Questions of type 'unanswerable' must have answerable=False")

        if not self.answerable:
            if self.gold_answer and self.gold_answer.strip():
                raise ValueError("Questions with answerable=False must have an empty gold_answer")

        # Multi-hop question semantics
        if self.question_type == QuestionType.MULTI_HOP:
            if self.reasoning_hops < 2:
                raise ValueError(
                    f"MULTI_HOP requires reasoning_hops >= 2, got {self.reasoning_hops}"
                )

        return self

    def to_query_input(self) -> AdapterQueryInput:
        """Generate adapter-safe query input using explicit query_metadata allowlist.

        CRITICAL ARCHITECTURAL CONTRACT:
        Evaluation metadata (self.metadata), gold answers, and supporting evidence
        are strictly isolated and never forwarded.
        """
        # Validate that query_metadata does not contain forbidden ground-truth keys
        for key in self.query_metadata:
            norm_key = key.strip().lower().replace("-", "_")
            if norm_key in PROHIBITED_GROUND_TRUTH_KEYS:
                raise GroundTruthLeakageError(
                    f"Forbidden ground-truth key '{key}' detected in query_metadata. "
                    "Evaluation ground-truth must not be passed to RAG adapters."
                )

        return AdapterQueryInput(
            question_id=self.question_id,
            question=self.question,
            metadata=copy.deepcopy(self.query_metadata),
        )
