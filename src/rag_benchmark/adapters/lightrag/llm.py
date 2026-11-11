"""OpenAI-compatible LLM wiring for the official LightRAG adapter.

LightRAG owns its own prompt construction (entity/relation extraction,
keyword extraction, final answer synthesis). The benchmark therefore must not
wrap those prompts in the traditional baseline template; instead the adapter
forwards LightRAG's prompts verbatim to the configured OpenAI-compatible
``/chat/completions`` endpoint using only the Python standard library.

API keys are never stored in configuration or source; the optional key is
resolved from the environment variable ``BENCHMARK_LLM_API_KEY`` at wiring
time (empty is allowed for local endpoints without authentication).
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from rag_benchmark.config.models import GenerationConfig, LLMConfig
from rag_benchmark.security.sanitizer import sanitize_url

#: Environment variable holding the optional LLM endpoint API key.
LLM_API_KEY_ENV = "BENCHMARK_LLM_API_KEY"

#: A LightRAG ``llm_model_func`` compatible coroutine.
LightRAGLLMFunc = Callable[..., Any]


@dataclass
class LLMCallRecord:
    """Wall-clock and token accounting for one LLM call made by LightRAG."""

    start: float
    end: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def elapsed_ms(self) -> float:
        if self.end is None:
            return 0.0
        return max((self.end - self.start) * 1000.0, 0.0)


@dataclass
class QueryUsageContext:
    """Query-local provider-call accounting.

    Bound to the running execution context for exactly one ``query()`` call.
    Sequential and interleaved queries each observe their own context, so
    provider calls (including failed attempts) can never leak between queries
    or from indexing into query accounting.
    """

    llm_records: list[LLMCallRecord] = field(default_factory=list)
    embedding_calls: int = 0

    @property
    def llm_calls(self) -> int:
        return len(self.llm_records)

    @property
    def input_tokens(self) -> int:
        return sum(record.input_tokens for record in self.llm_records)

    @property
    def output_tokens(self) -> int:
        return sum(record.output_tokens for record in self.llm_records)

    @property
    def last_llm_elapsed_ms(self) -> float:
        return self.llm_records[-1].elapsed_ms if self.llm_records else 0.0


#: Active per-query accounting context (None outside of query() execution).
_QUERY_USAGE_CONTEXT: contextvars.ContextVar[QueryUsageContext | None] = contextvars.ContextVar(
    "rag_benchmark_lightrag_query_usage", default=None
)


def bind_query_usage_context(
    context: QueryUsageContext,
) -> contextvars.Token[QueryUsageContext | None]:
    """Bind a fresh accounting context for the current execution context."""
    return _QUERY_USAGE_CONTEXT.set(context)


def reset_query_usage_context(token: contextvars.Token[QueryUsageContext | None]) -> None:
    _QUERY_USAGE_CONTEXT.reset(token)


def record_llm_call(record: LLMCallRecord) -> None:
    """Record an LLM call into the active query context, if one is bound.

    Indexing-time LLM calls (entity/relation extraction) run with no bound
    context and are intentionally excluded from query accounting.
    """
    context = _QUERY_USAGE_CONTEXT.get()
    if context is not None:
        context.llm_records.append(record)


def record_embedding_call() -> None:
    """Count one embedding provider invocation into the active query context.

    Must be called BEFORE the provider is invoked so a failed attempt is
    still counted; no-op outside a bound query context (indexing).
    """
    context = _QUERY_USAGE_CONTEXT.get()
    if context is not None:
        context.embedding_calls += 1


def build_messages(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: Sequence[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Assemble the chat payload from LightRAG's prompt contract.

    Only messages with string role/content are forwarded; anything else is
    dropped so a malformed history entry cannot corrupt the payload.
    """
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for entry in history_messages or []:
        if (
            isinstance(entry, dict)
            and isinstance(entry.get("role"), str)
            and isinstance(entry.get("content"), str)
        ):
            messages.append({"role": entry["role"], "content": entry["content"]})
    messages.append({"role": "user", "content": prompt})
    return messages


def _blocking_chat(
    endpoint: str,
    model: str,
    api_key: str,
    timeout_seconds: float,
    messages: list[dict[str, str]],
    generation: GenerationConfig,
) -> tuple[str, int, int]:
    """POST a chat completion and return (content, input_tokens, output_tokens).

    Blocking on purpose: callers run this in an executor so the asyncio event
    loop LightRAG drives stays responsive. Error messages are sanitized and
    never include credentials.
    """
    url = f"{endpoint}/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": generation.temperature,
        "top_p": generation.top_p,
        "max_tokens": generation.max_tokens,
        "seed": generation.seed,
    }
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
    choices = parsed.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise RuntimeError("LLM response missing 'choices'")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise RuntimeError("LLM response choice missing 'message'")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("LLM returned empty completion content")
    input_tokens = _parse_usage_count(parsed.get("usage"), "prompt_tokens")
    output_tokens = _parse_usage_count(parsed.get("usage"), "completion_tokens")
    return content, input_tokens, output_tokens


def _parse_usage_count(usage: Any, key: str) -> int:
    """Extract a non-negative integer token count from an OpenAI-style usage dict."""
    if not isinstance(usage, dict):
        return 0
    value = usage.get(key)
    if value is None or isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def build_default_llm_func(
    *,
    llm_config: LLMConfig,
    generation: GenerationConfig,
) -> LightRAGLLMFunc:
    """Build the default LightRAG ``llm_model_func`` from benchmark config.

    Every call is recorded (wall-clock bounds plus provider token usage) into
    the query-local ``QueryUsageContext`` bound by the adapter for the current
    ``query()`` execution; calls made outside a query (indexing) are dropped.
    """
    endpoint = llm_config.endpoint.rstrip("/")
    model = llm_config.model
    api_key = os.environ.get(LLM_API_KEY_ENV, "").strip()
    timeout_seconds = llm_config.timeout_seconds

    async def _llm_func(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: Sequence[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> str:
        if not prompt.strip():
            raise RuntimeError("LightRAG forwarded an empty prompt to the LLM endpoint")
        record = LLMCallRecord(start=time.perf_counter())
        record_llm_call(record)
        try:
            messages = build_messages(prompt, system_prompt, history_messages)
            loop = asyncio.get_running_loop()
            content, input_tokens, output_tokens = await loop.run_in_executor(
                None,
                _blocking_chat,
                endpoint,
                model,
                api_key,
                timeout_seconds,
                messages,
                generation,
            )
            record.input_tokens = input_tokens
            record.output_tokens = output_tokens
            return content
        finally:
            record.end = time.perf_counter()

    return _llm_func
