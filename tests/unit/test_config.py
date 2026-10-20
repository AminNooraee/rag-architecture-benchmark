"""Tests for shared configuration models, loading, and secret leakage prevention."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from rag_benchmark.config.loader import load_config
from rag_benchmark.config.models import (
    BenchmarkConfig,
    LLMConfig,
    sanitize_dict_for_manifest,
)
from rag_benchmark.errors.models import ConfigurationError


def test_default_config_creation() -> None:
    """Default benchmark configuration is valid."""
    config = BenchmarkConfig()
    assert config.benchmark.language == "fa"
    assert config.benchmark.seed == 42
    assert config.generation.temperature == 0.0
    assert config.generation.max_tokens == 512
    assert config.retrieval.top_k == 5


def test_load_config_from_yaml(tmp_path: Path) -> None:
    """Load BenchmarkConfig from YAML file."""
    yaml_content = """
system: traditional_rag
variant: dense_512
generation:
  temperature: 0.2
  max_tokens: 256
  seed: 123
retrieval:
  top_k: 10
embedding:
  model: "text-embedding-3-small"
  dimension: 1536
llm:
  model: "gpt-4o-mini"
  endpoint: "https://api.openai.com/v1"
benchmark:
  language: fa
  seed: 123
  dataset_version: "corpus-v1.0"
  question_set_version: "qset-v1.0"
"""
    cfg_file = tmp_path / "benchmark.yaml"
    cfg_file.write_text(yaml_content, encoding="utf-8")

    cfg = load_config(cfg_file)
    assert cfg.system == "traditional_rag"
    assert cfg.variant == "dense_512"
    assert cfg.generation.temperature == 0.2
    assert cfg.retrieval.top_k == 10
    assert cfg.embedding.dimension == 1536
    assert cfg.benchmark.dataset_version == "corpus-v1.0"


def test_load_config_missing_file_raises_error(tmp_path: Path) -> None:
    """Non-existent config file raises ConfigurationError."""
    missing = tmp_path / "does_not_exist.yaml"
    with pytest.raises(ConfigurationError, match="Configuration file not found"):
        load_config(missing)


def test_secret_leakage_prevention_in_extra() -> None:
    """Attempting to embed secrets in config extra fields must raise validation error."""
    with pytest.raises(ValidationError, match="sensitive information"):
        BenchmarkConfig(extra={"openai_api_key": "sk-secret12345"})

    with pytest.raises(ValidationError, match="sensitive information"):
        BenchmarkConfig(extra={"auth_token": "bearer xyz"})


def test_sanitize_dict_for_manifest_strips_secrets() -> None:
    """Sanitizer must recursively remove any keys matching credential patterns."""
    raw_dict = {
        "system": "traditional_rag",
        "nested": {
            "api_key": "SECRET_KEY_123",
            "password": "SUPER_SECRET_PW",
            "safe_param": 42,
            "deep": {
                "user_token": "token-xyz",
                "valid_field": "clean_value",
            },
        },
        "list_items": [
            {"secret_val": "forbidden", "allowed": "yes"},
            "simple_string",
        ],
    }

    sanitized = sanitize_dict_for_manifest(raw_dict)

    assert "api_key" not in sanitized["nested"]
    assert "password" not in sanitized["nested"]
    assert sanitized["nested"]["safe_param"] == 42
    assert "user_token" not in sanitized["nested"]["deep"]
    assert sanitized["nested"]["deep"]["valid_field"] == "clean_value"
    assert "secret_val" not in sanitized["list_items"][0]
    assert sanitized["list_items"][0]["allowed"] == "yes"


def test_to_safe_dict_produces_clean_manifest_dict() -> None:
    """to_safe_dict() ensures serializable output is safe for manifest."""
    config = BenchmarkConfig(
        system="neo4j_graphrag",
        variant="graph_only",
        llm=LLMConfig(model="local-persian-model", endpoint="http://localhost:8000"),
    )
    safe = config.to_safe_dict()
    assert safe["system"] == "neo4j_graphrag"
    assert safe["llm"]["model"] == "local-persian-model"
    # Ensure no secret keys exist in the safe dict
    for k in safe:
        assert "key" not in k.lower()
        assert "secret" not in k.lower()
