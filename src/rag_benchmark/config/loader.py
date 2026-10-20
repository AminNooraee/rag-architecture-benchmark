"""Loader for benchmark configuration files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from rag_benchmark.config.models import BenchmarkConfig
from rag_benchmark.errors.models import ConfigurationError


def load_config(source: str | Path | dict[str, Any]) -> BenchmarkConfig:
    """Load and validate a BenchmarkConfig from a file path or dictionary.

    Args:
        source: File path (YAML or JSON) or existing dictionary.

    Returns:
        Validated BenchmarkConfig instance.

    Raises:
        ConfigurationError: If the file cannot be read or validation fails.
    """
    data: dict[str, Any]

    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise ConfigurationError(f"Configuration file not found: {path}")

        try:
            with open(path, encoding="utf-8") as f:
                content = f.read()
                parsed = yaml.safe_load(content)
                if not isinstance(parsed, dict):
                    raise ConfigurationError(
                        f"Configuration at {path} must parse into a mapping/dict, "
                        f"got {type(parsed).__name__}"
                    )
                data = parsed
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Failed to parse configuration YAML at {path}: {e}") from e
        except OSError as e:
            raise ConfigurationError(f"Failed to read configuration file at {path}: {e}") from e
    elif isinstance(source, dict):
        data = source
    else:
        raise ConfigurationError(f"Unsupported configuration source type: {type(source).__name__}")

    try:
        return BenchmarkConfig.model_validate(data)
    except ValidationError as e:
        raise ConfigurationError(f"Configuration validation failed: {e}") from e
