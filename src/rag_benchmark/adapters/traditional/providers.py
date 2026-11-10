"""Minimal typed provider abstractions for embeddings and LLM generation.

The adapter depends only on the two small protocols below, so the benchmark
can run fully offline with deterministic fake providers. An OpenAI-compatible
HTTP implementation is provided using only the Python standard library; API
keys are never stored in configuration or source and are resolved from
environment variables at wiring time.
"""

from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from rag_benchmark.config.models import GenerationConfig
from rag_benchmark.errors.models import ConfigurationError
from rag_benchmark.security.sanitizer import sanitize_url

LLM_SYSTEM_INSTRUCTION = (
    "You are a Persian retrieval-augmented question answering assistant. "
    "Answer the user's question using ONLY the numbered context passages provided. "
    "Preserve Persian text verbatim. "
    "If the provided context is insufficient to answer the question, state clearly "
    "that the provided context is insufficient."
)


@dataclass(frozen=True)
class LLMGenerationResult:
    """Outcome of a single LLM generation call.

    Token counts are optional: providers that cannot report them leave them
    as None and the adapter records zeros rather than fabricating counts.
    """

    answer: str
    input_tokens: int | None = None
    output_tokens: int | None = None


class EmbeddingProvider(Protocol):
    """Minimal embedding provider contract (finite numeric vectors)."""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of texts, returning one vector per input text in order."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text."""
        ...


class LLMProvider(Protocol):
    """Minimal LLM generation contract (question + retrieved context)."""

    def generate(
        self,
        question: str,
        context: Sequence[str],
        config: GenerationConfig,
    ) -> LLMGenerationResult:
        """Generate an answer from the question and retrieved context passages."""
        ...


def build_llm_messages(question: str, context: Sequence[str]) -> list[dict[str, str]]:
    """Build the fixed two-message chat payload for the baseline prompt.

    The prompt contains only: a stable system instruction, the retrieved
    context passages in retrieval-rank order, and the user question. No
    rewriting, HyDE, or ground-truth material is introduced.
    """
    if context:
        context_block = "\n\n".join(
            f"[{rank}] {text}" for rank, text in enumerate(context, start=1)
        )
    else:
        context_block = "(no context)"
    user_message = f"Context:\n{context_block}\n\nQuestion:\n{question}"
    return [
        {"role": "system", "content": LLM_SYSTEM_INSTRUCTION},
        {"role": "user", "content": user_message},
    ]


def resolve_api_key_from_env(env_var: str) -> str:
    """Read an API key from the environment, failing clearly when absent."""
    value = os.environ.get(env_var, "")
    if not value.strip():
        raise ConfigurationError(
            f"Environment variable '{env_var}' is not set or is empty. "
            "API keys must be provided via environment variables only."
        )
    return value.strip()


def _post_json(
    url: str,
    payload: dict[str, object],
    api_key: str,
    timeout_seconds: float,
) -> dict[str, object]:
    """POST a JSON payload and parse the JSON object response (stdlib only)."""
    data = json.dumps(payload).encode("utf-8")
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} from {sanitize_url(url)}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Connection error to {sanitize_url(url)}: {e.reason}") from e
    except OSError as e:
        raise RuntimeError(f"Request to {sanitize_url(url)} failed: {e}") from e
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON response from {sanitize_url(url)}") from e
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Unexpected JSON response type from {sanitize_url(url)}")
    return parsed


def _require_endpoint(endpoint: str, kind: str) -> str:
    if not endpoint.strip():
        raise ConfigurationError(f"{kind} endpoint must not be empty")
    return endpoint.rstrip("/")


def _require_model(model: str, kind: str) -> str:
    if not model.strip():
        raise ConfigurationError(f"{kind} model must not be empty")
    return model


def _require_timeout(timeout_seconds: float) -> float:
    if timeout_seconds <= 0:
        raise ConfigurationError(f"timeout_seconds must be positive, got {timeout_seconds}")
    return timeout_seconds


def _parse_embedding_vector(item: dict[str, object]) -> list[float]:
    """Parse and validate one ``embedding`` list from an embedding response item.

    Every component must be an int or float (bools are rejected explicitly),
    finite (no NaN/Infinity), and strings/None are rejected. Valid int
    components are converted to float after validation.
    """
    embedding = item.get("embedding")
    if not isinstance(embedding, list):
        raise RuntimeError("Embedding response item missing 'embedding' list")
    if not embedding:
        raise RuntimeError("Embedding response item has an empty 'embedding' list")
    vector: list[float] = []
    for position, value in enumerate(embedding):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RuntimeError(f"Embedding component {position} must be numeric")
        component = float(value)
        if not math.isfinite(component):
            raise RuntimeError(f"Embedding component {position} must be finite")
        vector.append(component)
    return vector


def _parse_usage_count(value: object, name: str) -> int | None:
    """Validate an LLM usage token count: None or a non-negative int.

    Booleans and negative values are malformed and rejected rather than
    silently dropped.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(f"LLM response '{name}' must be a non-negative integer")
    return value


class OpenAICompatibleEmbeddingProvider:
    """Embedding provider for any OpenAI-compatible ``/embeddings`` endpoint."""

    def __init__(
        self,
        endpoint: str,
        model: str,
        api_key: str = "",
        timeout_seconds: float = 60.0,
    ) -> None:
        self._endpoint = _require_endpoint(endpoint, "Embedding")
        self._model = _require_model(model, "Embedding")
        self._api_key = api_key
        self._timeout = _require_timeout(timeout_seconds)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        texts_list = list(texts)
        if not texts_list:
            return []
        expected = len(texts_list)
        payload: dict[str, object] = {"model": self._model, "input": texts_list}
        response = _post_json(f"{self._endpoint}/embeddings", payload, self._api_key, self._timeout)
        data = response.get("data")
        if not isinstance(data, list) or len(data) != expected:
            raise RuntimeError("Embedding response 'data' length does not match input texts")
        vectors_by_index: dict[int, list[float]] = {}
        for item in data:
            if not isinstance(item, dict):
                raise RuntimeError("Embedding response 'data' items must be JSON objects")
            if "index" not in item:
                raise RuntimeError("Embedding response item is missing explicit 'index'")
            index = item["index"]
            if isinstance(index, bool) or not isinstance(index, int):
                raise RuntimeError("Embedding response 'index' must be an integer")
            if not 0 <= index < expected:
                raise RuntimeError("Embedding response 'index' value is out of range")
            if index in vectors_by_index:
                raise RuntimeError("Embedding response contains duplicate 'index' values")
            vectors_by_index[index] = _parse_embedding_vector(item)
        if len(vectors_by_index) != expected:
            raise RuntimeError("Embedding response 'index' values do not cover 0..N-1")
        # Reorder strictly by the explicit index; response position is never
        # used as a fallback, so shuffled responses come back in input order.
        return [vectors_by_index[position] for position in range(expected)]

    def embed_query(self, text: str) -> list[float]:
        # A single-text response must carry explicit index 0 (enforced by the
        # shared validation in embed_documents).
        return self.embed_documents([text])[0]


class OpenAICompatibleLLMProvider:
    """LLM provider for any OpenAI-compatible ``/chat/completions`` endpoint."""

    def __init__(
        self,
        endpoint: str,
        model: str,
        api_key: str = "",
        timeout_seconds: float = 60.0,
    ) -> None:
        self._endpoint = _require_endpoint(endpoint, "LLM")
        self._model = _require_model(model, "LLM")
        self._api_key = api_key
        self._timeout = _require_timeout(timeout_seconds)

    def generate(
        self,
        question: str,
        context: Sequence[str],
        config: GenerationConfig,
    ) -> LLMGenerationResult:
        messages = build_llm_messages(question, context)
        payload: dict[str, object] = {
            "model": self._model,
            "messages": messages,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_tokens": config.max_tokens,
            "seed": config.seed,
        }
        response = _post_json(
            f"{self._endpoint}/chat/completions", payload, self._api_key, self._timeout
        )
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise RuntimeError("LLM response missing 'choices'")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise RuntimeError("LLM response choice missing 'message'")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("LLM returned empty completion content")
        input_tokens: int | None = None
        output_tokens: int | None = None
        usage = response.get("usage")
        if isinstance(usage, dict):
            input_tokens = _parse_usage_count(usage.get("prompt_tokens"), "prompt_tokens")
            output_tokens = _parse_usage_count(usage.get("completion_tokens"), "completion_tokens")
        return LLMGenerationResult(
            answer=content, input_tokens=input_tokens, output_tokens=output_tokens
        )
