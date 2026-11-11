"""Unit tests for the LightRAG adapter's LLM wiring and offline tokenizer.

The OpenAI-compatible default LLM func is tested without network access by
monkeypatching the blocking transport layer.
"""

from __future__ import annotations

import asyncio
import io
import json
import urllib.error
import urllib.request
from email.message import Message
from typing import Any, Literal, cast

import pytest

from rag_benchmark.adapters.lightrag import llm as llm_module
from rag_benchmark.adapters.lightrag.llm import (
    QueryUsageContext,
    _blocking_chat,
    _parse_usage_count,
    bind_query_usage_context,
    build_default_llm_func,
    build_messages,
    reset_query_usage_context,
)
from rag_benchmark.adapters.lightrag.tokenizer import (
    CharacterTokenizer,
    build_offline_character_tokenizer,
)
from rag_benchmark.config.models import GenerationConfig, LLMConfig


def _configs() -> tuple[LLMConfig, GenerationConfig]:
    llm_config = LLMConfig.model_validate(
        {
            "model": "test-model",
            "endpoint": "http://localhost:9999/v1",
            "timeout_seconds": 5.0,
        }
    )
    return llm_config, GenerationConfig()


def run_llm_with_usage(coro: Any) -> tuple[Any, QueryUsageContext]:
    """Run an LLM func coroutine with a bound query usage context.

    Mirrors how the adapter binds a context for one query(); the context is
    reset afterwards so calls never leak between queries.
    """
    usage = QueryUsageContext()
    token = bind_query_usage_context(usage)
    try:
        result = asyncio.run(coro)
    finally:
        reset_query_usage_context(token)
    return result, usage


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


def test_character_tokenizer_roundtrip() -> None:
    tokenizer = CharacterTokenizer()
    sample = "متن آزمایشی with accents é and emoji 🚀 plus ascii 123"
    encoded = tokenizer.encode(sample)
    assert encoded == [ord(char) for char in sample]
    assert all(isinstance(token, int) for token in encoded)
    assert tokenizer.decode(encoded) == sample
    assert tokenizer.encode("") == []
    assert tokenizer.decode([]) == ""


def test_build_offline_character_tokenizer_returns_lightrag_tokenizer() -> None:
    pytest.importorskip("lightrag")
    from lightrag.utils import Tokenizer  # type: ignore[import-untyped]

    built = build_offline_character_tokenizer()
    assert isinstance(built, Tokenizer)
    assert built.tokenizer.encode("آب") == [ord("آ"), ord("ب")]
    assert built.tokenizer.decode([ord("آ"), ord("ب")]) == "آب"


# ---------------------------------------------------------------------------
# Message assembly
# ---------------------------------------------------------------------------


def test_build_messages_orders_system_history_and_user() -> None:
    history: list[object] = [
        {"role": "user", "content": "سابقه کاربر"},
        {"role": 42, "content": "entry with bad role"},
        "not a dict",
        {"role": "assistant", "content": "پاسخ قبلی"},
        {"role": "user", "content": None},
    ]
    messages = build_messages("پرسش جدید", system_prompt="سیستم", history_messages=history)  # type: ignore[arg-type]
    assert messages == [
        {"role": "system", "content": "سیستم"},
        {"role": "user", "content": "سابقه کاربر"},
        {"role": "assistant", "content": "پاسخ قبلی"},
        {"role": "user", "content": "پرسش جدید"},
    ]


def test_build_messages_without_system_or_history() -> None:
    assert build_messages("پرسش") == [{"role": "user", "content": "پرسش"}]


# ---------------------------------------------------------------------------
# Usage parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        (None, 0),
        ("not-a-dict", 0),
        ({}, 0),
        ({"prompt_tokens": 42}, 42),
        ({"prompt_tokens": True}, 0),
        ({"prompt_tokens": -3}, 0),
        ({"prompt_tokens": 1.5}, 0),
    ],
)
def test_parse_usage_count(usage: object, expected: int) -> None:
    assert _parse_usage_count(usage, "prompt_tokens") == expected


# ---------------------------------------------------------------------------
# Default LLM func (transport monkeypatched)
# ---------------------------------------------------------------------------


def test_default_llm_func_records_usage_and_returns_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm_config, generation = _configs()
    captured: dict[str, object] = {}

    def fake_blocking_chat(
        endpoint: str,
        model: str,
        api_key: str,
        timeout: float,
        messages: list[dict[str, str]],
        gen: GenerationConfig,
    ) -> tuple[str, int, int]:
        captured["endpoint"] = endpoint
        captured["model"] = model
        captured["api_key"] = api_key
        captured["messages"] = messages
        return "جواب تستی", 11, 4

    monkeypatch.setattr(llm_module, "_blocking_chat", fake_blocking_chat)
    monkeypatch.delenv(llm_module.LLM_API_KEY_ENV, raising=False)
    func = build_default_llm_func(llm_config=llm_config, generation=generation)

    result, usage = run_llm_with_usage(func("پرسش آزمایشی", system_prompt="سیستم"))

    assert result == "جواب تستی"
    assert captured["endpoint"] == "http://localhost:9999/v1"
    assert captured["model"] == "test-model"
    assert captured["api_key"] == ""
    assert captured["messages"] == [
        {"role": "system", "content": "سیستم"},
        {"role": "user", "content": "پرسش آزمایشی"},
    ]
    assert len(usage.llm_records) == 1
    assert usage.llm_records[0].input_tokens == 11
    assert usage.llm_records[0].output_tokens == 4
    assert usage.llm_records[0].elapsed_ms >= 0.0
    assert usage.llm_records[0].end is not None


def test_default_llm_func_reads_api_key_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    llm_config, generation = _configs()
    captured: dict[str, object] = {}

    def fake_blocking_chat(
        endpoint: str,
        model: str,
        api_key: str,
        timeout: float,
        messages: list[dict[str, str]],
        gen: GenerationConfig,
    ) -> tuple[str, int, int]:
        captured["api_key"] = api_key
        return "ok", 1, 1

    monkeypatch.setattr(llm_module, "_blocking_chat", fake_blocking_chat)
    monkeypatch.setenv(llm_module.LLM_API_KEY_ENV, "  test-key  ")
    func = build_default_llm_func(llm_config=llm_config, generation=generation)

    run_llm_with_usage(func("پرسش"))

    assert captured["api_key"] == "test-key"


def test_default_llm_func_empty_prompt_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    llm_config, generation = _configs()
    monkeypatch.setattr(llm_module, "_blocking_chat", lambda *a, **k: ("x", 0, 0))
    func = build_default_llm_func(llm_config=llm_config, generation=generation)

    with pytest.raises(RuntimeError, match="empty prompt"):
        run_llm_with_usage(func("   "))


def test_default_llm_func_records_when_chat_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    llm_config, generation = _configs()

    def boom(*args: object, **kwargs: object) -> tuple[str, int, int]:
        raise RuntimeError("boom")

    monkeypatch.setattr(llm_module, "_blocking_chat", boom)
    func = build_default_llm_func(llm_config=llm_config, generation=generation)

    usage = QueryUsageContext()
    token = bind_query_usage_context(usage)
    try:
        with pytest.raises(RuntimeError, match="boom"):
            asyncio.run(func("پرسش"))
    finally:
        reset_query_usage_context(token)

    assert len(usage.llm_records) == 1
    assert usage.llm_records[0].end is not None
    assert usage.llm_records[0].input_tokens == 0
    assert usage.llm_records[0].output_tokens == 0


# ---------------------------------------------------------------------------
# Blocking transport
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc_info: object) -> Literal[False]:
        return False


def test_blocking_chat_sends_payload_and_parses_response(monkeypatch: pytest.MonkeyPatch) -> None:
    llm_config, generation = _configs()
    seen: list[tuple[urllib.request.Request, float | None]] = []

    def fake_urlopen(
        request: urllib.request.Request, timeout: float | None = None
    ) -> _FakeResponse:
        seen.append((request, timeout))
        body = json.dumps(
            {
                "choices": [{"message": {"content": "جواب"}}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3},
            }
        ).encode("utf-8")
        return _FakeResponse(body)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    content, input_tokens, output_tokens = _blocking_chat(
        "http://localhost:9999/v1",
        "test-model",
        "",
        5.0,
        [{"role": "user", "content": "پرسش"}],
        generation,
    )

    assert content == "جواب"
    assert (input_tokens, output_tokens) == (7, 3)
    request, timeout = seen[0]
    assert request.get_full_url() == "http://localhost:9999/v1/chat/completions"
    assert timeout == 5.0
    assert request.get_header("Content-type") == "application/json"
    payload = json.loads(cast(bytes, request.data).decode("utf-8"))
    assert payload["model"] == "test-model"
    assert payload["messages"] == [{"role": "user", "content": "پرسش"}]
    assert payload["temperature"] == generation.temperature
    assert payload["top_p"] == generation.top_p
    assert payload["max_tokens"] == generation.max_tokens
    assert payload["seed"] == generation.seed


def test_blocking_chat_sends_bearer_auth_when_key_present(monkeypatch: pytest.MonkeyPatch) -> None:
    llm_config, generation = _configs()
    seen: list[urllib.request.Request] = []

    def fake_urlopen(
        request: urllib.request.Request, timeout: float | None = None
    ) -> _FakeResponse:
        seen.append(request)
        return _FakeResponse(json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    _blocking_chat("http://localhost:9999/v1", "m", "secret", 5.0, [], generation)
    assert seen[0].get_header("Authorization") == "Bearer secret"


def test_blocking_chat_http_error_is_sanitized(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(
        request: urllib.request.Request, timeout: float | None = None
    ) -> _FakeResponse:
        headers: Message[str, str] = Message()
        raise urllib.error.HTTPError(
            request.get_full_url(), 500, "Internal Server Error", headers, io.BytesIO(b"")
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="HTTP 500"):
        _blocking_chat(
            "http://localhost:9999/v1",
            "m",
            "",
            5.0,
            [{"role": "user", "content": "x"}],
            GenerationConfig(),
        )


def test_blocking_chat_rejects_empty_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(
        request: urllib.request.Request, timeout: float | None = None
    ) -> _FakeResponse:
        return _FakeResponse(json.dumps({"choices": [{"message": {"content": "  "}}]}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="empty completion"):
        _blocking_chat(
            "http://localhost:9999/v1",
            "m",
            "",
            5.0,
            [{"role": "user", "content": "x"}],
            GenerationConfig(),
        )
