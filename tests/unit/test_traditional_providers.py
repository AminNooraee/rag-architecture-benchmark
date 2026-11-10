"""Tests for provider abstractions and OpenAI-compatible implementations."""

from __future__ import annotations

import math

import pytest

from rag_benchmark.adapters.traditional import (
    LLMGenerationResult,
    OpenAICompatibleEmbeddingProvider,
    OpenAICompatibleLLMProvider,
    build_llm_messages,
    resolve_api_key_from_env,
)
from rag_benchmark.adapters.traditional import providers as providers_module
from rag_benchmark.config.models import GenerationConfig
from rag_benchmark.errors.models import ConfigurationError
from tests.unit.fakes import FakeEmbeddingProvider, FakeLLMProvider

ZWNJ = "\u200c"


def test_llm_generation_result_defaults() -> None:
    result = LLMGenerationResult(answer="پاسخ")
    assert result.input_tokens is None
    assert result.output_tokens is None


def test_build_llm_messages_structure() -> None:
    messages = build_llm_messages("سؤال چیست؟", ["پاسخ الف", "پاسخ ب"])
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "insufficient" in messages[0]["content"]
    user = messages[1]["content"]
    assert messages[1]["role"] == "user"
    assert "[1] پاسخ الف" in user
    assert "[2] پاسخ ب" in user
    assert user.index("[1]") < user.index("[2]")
    assert user.endswith("Question:\nسؤال چیست؟")


def test_build_llm_messages_empty_context() -> None:
    messages = build_llm_messages("سؤال", [])
    assert "(no context)" in messages[1]["content"]


def test_build_llm_messages_preserves_persian_verbatim() -> None:
    question = f"نیم{ZWNJ}فاصله چگونه نوشته می{ZWNJ}شود؟"
    messages = build_llm_messages(question, [])
    assert question in messages[1]["content"]
    assert ZWNJ in messages[1]["content"]


def test_fake_embedding_provider_is_deterministic() -> None:
    provider = FakeEmbeddingProvider(dimension=16)
    first = provider.embed_query("متن ناشناخته")
    second = provider.embed_query("متن ناشناخته")
    assert first == second
    assert len(first) == 16
    assert all(math.isfinite(value) for value in first)
    provider2 = FakeEmbeddingProvider(dimension=16)
    assert provider.embed_documents(["الف", "ب"]) == provider2.embed_documents(["الف", "ب"])


def test_fake_embedding_provider_honors_mapping() -> None:
    provider = FakeEmbeddingProvider(dimension=2, mapping={"الف": (3.0, 4.0)})
    assert provider.embed_query("الف") == [3.0, 4.0]


def test_fake_embedding_provider_failure_injection() -> None:
    provider = FakeEmbeddingProvider(
        dimension=2,
        fail_documents=RuntimeError("doc-down"),
        fail_query=RuntimeError("query-down"),
    )
    with pytest.raises(RuntimeError, match="doc-down"):
        provider.embed_documents(["الف"])
    with pytest.raises(RuntimeError, match="query-down"):
        provider.embed_query("الف")


def test_fake_llm_provider_captures_calls() -> None:
    provider = FakeLLMProvider()
    config = GenerationConfig()
    provider.generate("سؤال", ["الف"], config)
    assert provider.call_count == 1
    question, context, used_config = provider.calls[0]
    assert question == "سؤال"
    assert context == ("الف",)
    assert used_config is config
    assert "سؤال" in provider.generate("سؤال", ["الف"], config).answer


def test_resolve_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_RAG_API_KEY", "  sk-test-0123456789  ")
    assert resolve_api_key_from_env("TEST_RAG_API_KEY") == "sk-test-0123456789"
    monkeypatch.delenv("TEST_RAG_API_KEY", raising=False)
    with pytest.raises(ConfigurationError):
        resolve_api_key_from_env("TEST_RAG_API_KEY")
    monkeypatch.setenv("TEST_RAG_API_KEY", "   ")
    with pytest.raises(ConfigurationError):
        resolve_api_key_from_env("TEST_RAG_API_KEY")


def _capturing_post(
    monkeypatch: pytest.MonkeyPatch, response: dict[str, object]
) -> list[tuple[str, dict[str, object]]]:
    captured: list[tuple[str, dict[str, object]]] = []

    def fake_post(
        url: str, payload: dict[str, object], api_key: str, timeout_seconds: float
    ) -> dict[str, object]:
        captured.append((url, payload))
        return response

    monkeypatch.setattr(providers_module, "_post_json", fake_post)
    return captured


def test_openai_embedding_provider_orders_by_index(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capturing_post(
        monkeypatch,
        {
            "data": [
                {"index": 1, "embedding": [0.0, 1.0]},
                {"index": 0, "embedding": [1.0, 0.0]},
            ]
        },
    )
    provider = OpenAICompatibleEmbeddingProvider("https://api.example/v1", model="emb-1")
    vectors = provider.embed_documents(["متن-۱", "متن-۲"])
    assert vectors == [[1.0, 0.0], [0.0, 1.0]]
    url, payload = captured[0]
    assert url == "https://api.example/v1/embeddings"
    assert payload["model"] == "emb-1"
    assert payload["input"] == ["متن-۱", "متن-۲"]


def test_openai_embedding_provider_query(monkeypatch: pytest.MonkeyPatch) -> None:
    _capturing_post(monkeypatch, {"data": [{"index": 0, "embedding": [0.5, 0.5]}]})
    provider = OpenAICompatibleEmbeddingProvider("https://api.example/v1/", model="emb-1")
    assert provider.embed_query("سؤال") == [0.5, 0.5]
    assert provider.embed_documents([]) == []


def test_openai_embedding_provider_rejects_length_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    _capturing_post(monkeypatch, {"data": [{"index": 0, "embedding": [0.0, 1.0]}]})
    provider = OpenAICompatibleEmbeddingProvider("https://api.example/v1", model="emb-1")
    with pytest.raises(RuntimeError):
        provider.embed_documents(["الف", "ب"])


def test_openai_llm_provider_generate(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capturing_post(
        monkeypatch,
        {
            "choices": [{"message": {"content": "پاسخ فارسی"}}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3},
        },
    )
    provider = OpenAICompatibleLLMProvider("https://api.example/v1", model="llm-1")
    config = GenerationConfig(temperature=0.0, max_tokens=128, seed=7, top_p=1.0)
    result = provider.generate("سؤال", ["الف"], config)
    assert result.answer == "پاسخ فارسی"
    assert result.input_tokens == 7
    assert result.output_tokens == 3
    url, payload = captured[0]
    assert url == "https://api.example/v1/chat/completions"
    assert payload["model"] == "llm-1"
    assert payload["temperature"] == 0.0
    assert payload["max_tokens"] == 128
    assert payload["seed"] == 7
    assert payload["messages"] == build_llm_messages("سؤال", ["الف"])


def test_openai_llm_provider_empty_content_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _capturing_post(monkeypatch, {"choices": [{"message": {"content": "  "}}]})
    provider = OpenAICompatibleLLMProvider("https://api.example/v1", model="llm-1")
    with pytest.raises(RuntimeError):
        provider.generate("سؤال", [], GenerationConfig())


def test_openai_llm_provider_missing_usage_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    _capturing_post(monkeypatch, {"choices": [{"message": {"content": "پاسخ"}}]})
    provider = OpenAICompatibleLLMProvider("https://api.example/v1", model="llm-1")
    result = provider.generate("سؤال", [], GenerationConfig())
    assert result.input_tokens is None
    assert result.output_tokens is None


def test_provider_validation_rejects_bad_parameters() -> None:
    with pytest.raises(ConfigurationError):
        OpenAICompatibleEmbeddingProvider("", model="emb-1")
    with pytest.raises(ConfigurationError):
        OpenAICompatibleEmbeddingProvider("https://api.example/v1", model="")
    with pytest.raises(ConfigurationError):
        OpenAICompatibleEmbeddingProvider(
            "https://api.example/v1", model="emb-1", timeout_seconds=0.0
        )
    with pytest.raises(ConfigurationError):
        OpenAICompatibleLLMProvider("", model="llm-1")
    with pytest.raises(ConfigurationError):
        OpenAICompatibleLLMProvider("https://api.example/v1", model="")
    with pytest.raises(ConfigurationError):
        OpenAICompatibleLLMProvider("https://api.example/v1", model="llm-1", timeout_seconds=-1.0)


def test_fakes_expose_provider_method_surface() -> None:
    # The protocols are nominal; verify the fakes expose the required methods.
    assert callable(FakeEmbeddingProvider().embed_documents)
    assert callable(FakeEmbeddingProvider().embed_query)
    assert callable(FakeLLMProvider().generate)
