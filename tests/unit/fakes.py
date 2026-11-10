"""Deterministic offline fake providers for Traditional Vector RAG tests.

Plain typed classes (no pytest fixtures) so they can be imported from both
unit and integration test modules.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence

from rag_benchmark.adapters.traditional.providers import LLMGenerationResult
from rag_benchmark.config.models import GenerationConfig


class FakeEmbeddingProvider:
    """Offline embedding provider with fixed and hash-derived vectors.

    Known texts map to the vectors supplied in ``mapping``; unknown texts
    derive stable vectors from SHA-256, so retrieval order is deterministic.
    """

    def __init__(
        self,
        dimension: int = 8,
        mapping: dict[str, Sequence[float]] | None = None,
        fail_documents: Exception | None = None,
        fail_query: Exception | None = None,
    ) -> None:
        self._dimension = dimension
        converted = {}
        for text, vector in (mapping or {}).items():
            converted[text] = tuple(float(value) for value in vector)
        self._mapping = converted
        self._fail_documents = fail_documents
        self._fail_query = fail_query
        self.documents_batches: list[list[str]] = []
        self.query_texts: list[str] = []

    def _vector_for(self, text: str) -> tuple[float, ...]:
        if text in self._mapping:
            return self._mapping[text]
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values = [digest[i % len(digest)] / 255.0 - 0.5 for i in range(self._dimension)]
        if all(value == 0.0 for value in values):
            values[0] = 0.5
        return tuple(values)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if self._fail_documents is not None:
            raise self._fail_documents
        self.documents_batches.append(list(texts))
        return [list(self._vector_for(text)) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        if self._fail_query is not None:
            raise self._fail_query
        self.query_texts.append(text)
        return list(self._vector_for(text))


class FakeLLMProvider:
    """Offline LLM provider with deterministic answers and full call capture."""

    def __init__(
        self,
        answer_for: Callable[[str, tuple[str, ...]], str] | None = None,
        raise_error: Exception | None = None,
        report_usage: bool = False,
        input_tokens: int = 10,
        output_tokens: int = 5,
    ) -> None:
        self._answer_for = answer_for
        self._raise_error = raise_error
        self._report_usage = report_usage
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        self.call_count = 0
        self.calls: list[tuple[str, tuple[str, ...], GenerationConfig]] = []

    @staticmethod
    def default_answer(question: str, context: tuple[str, ...]) -> str:
        if not context:
            return "The provided context is insufficient to answer the question."
        return f"Answer derived from {len(context)} passage(s) for: {question}"

    def generate(
        self, question: str, context: Sequence[str], config: GenerationConfig
    ) -> LLMGenerationResult:
        self.call_count += 1
        self.calls.append((question, tuple(context), config))
        if self._raise_error is not None:
            raise self._raise_error
        if self._answer_for is not None:
            answer = self._answer_for(question, tuple(context))
        else:
            answer = self.default_answer(question, tuple(context))
        if not self._report_usage:
            return LLMGenerationResult(answer=answer)
        return LLMGenerationResult(
            answer=answer, input_tokens=self._input_tokens, output_tokens=self._output_tokens
        )
