"""Typed configuration models for Phase 3B dataset building.

These models enforce strict validation of the YAML configuration used by the
CLI. They reject unknown keys, require mandatory fields, and ensure that
`enabled` flags are real booleans. Duplicate keys in the YAML file are also
rejected by the custom loader.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator, validator
from typing import Any


def _unique_key_loader(stream: str) -> dict[str, Any]:
    """Load YAML with duplicate‑key detection.

    Raises ``yaml.YAMLError`` when a duplicate mapping key is encountered.
    """

    class UniqueKeyLoader(yaml.SafeLoader):
        pass

    def construct_mapping(loader: yaml.Loader, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        mapping: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in mapping:
                raise yaml.YAMLError(f"duplicate key: {key}")
            value = loader.construct_object(value_node, deep=deep)
            mapping[key] = value
        return mapping

    UniqueKeyLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping,
    )
    return yaml.load(stream, Loader=UniqueKeyLoader) or {}  # type: dict[str, Any]


class DatasetIdentityConfig(BaseModel):
    name: str = Field(..., description="Dataset name")
    version: str = Field(..., description="Dataset version")

    model_config = ConfigDict(extra="forbid", frozen=True)


class OutputConfig(BaseModel):
    directory: Path = Field(..., description="Output directory for the generated dataset")

    @validator("directory", pre=True)
    def _ensure_path(cls, v: str | Path) -> Path:
        return Path(v)

    model_config = ConfigDict(extra="forbid", frozen=True)


class BaseSourceConfig(BaseModel):
    enabled: bool = Field(False, description="Whether this source should be included")
    path: Path | None = None
    version: str = ""

    @validator("path", pre=True, always=True)
    def _require_path_if_enabled(cls, v: str | Path | None, values: dict[str, Any]) -> Path | None:
        if values.get("enabled") and not v:
            raise ValueError("'path' must be provided when source is enabled")
        return Path(v) if v is not None else None

    model_config = ConfigDict(extra="forbid", frozen=True)


class PQuADSourceConfig(BaseSourceConfig):
    pass


class PersianMHQASourceConfig(BaseSourceConfig):
    pass


class WikipediaSourceConfig(BaseSourceConfig):
    pass


class CustomWikiQuestionsSourceConfig(BaseSourceConfig):
    pass


class SourcesConfig(BaseModel):
    pquad: PQuADSourceConfig | None = None
    persian_mhqa: PersianMHQASourceConfig | None = None
    wikipedia: WikipediaSourceConfig | None = None
    custom_wikipedia_questions: CustomWikiQuestionsSourceConfig | None = None

    model_config = ConfigDict(extra="forbid", frozen=True)


class RootConfig(BaseModel):
    dataset: DatasetIdentityConfig
    output: OutputConfig
    sources: SourcesConfig = Field(default_factory=SourcesConfig)

    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="before")
    @classmethod
    def _ensure_mapping(cls, values: Any) -> dict[str, Any]:
        if not isinstance(values, dict):
            raise ValueError("Configuration root must be a mapping")
        return values


def load_config(path: Path) -> RootConfig:
    """Load and validate a YAML configuration file.

    The function performs duplicate‑key detection and then parses the content
    into :class:`RootConfig`. Validation errors are raised as ``ValidationError``
    instances.
    """
    raw_text = path.read_text(encoding="utf-8")
    data = _unique_key_loader(raw_text)
    return RootConfig.model_validate(data)
