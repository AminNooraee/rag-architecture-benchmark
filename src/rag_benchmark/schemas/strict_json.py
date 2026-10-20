"""Strict JSON-native validation contracts and deep copying utilities."""

from __future__ import annotations

import json
import math
from typing import Any


def validate_strict_json_value(val: Any, path: str = "value") -> Any:
    """Validate that a value conforms strictly to JSON-native types.

    Permitted types:
      - None
      - bool
      - int (excluding bool)
      - float (finite only: no NaN, Inf, -Inf)
      - str
      - list (tuples and sets are rejected)
      - dict (all keys MUST be strings)

    Rejected types:
      - tuple, set, frozenset
      - datetime, date, time
      - Enum (must be explicitly string-converted prior to validation)
      - float('nan'), float('inf'), float('-inf')
      - non-string dictionary keys
      - custom classes / objects

    Returns a defensive deep copy of the validated structure.
    """
    if val is None:
        return None

    # bool must be checked before int because bool subclasses int in Python
    if isinstance(val, bool):
        return val

    if isinstance(val, int):
        return val

    if isinstance(val, float):
        if not math.isfinite(val):
            raise ValueError(f"Strict JSON forbids non-finite float at {path}: {val}")
        return val

    if isinstance(val, str):
        return val

    if isinstance(val, tuple):
        raise ValueError(
            f"Strict JSON rejects tuples at {path} (tuples silently degrade to lists in JSON)"
        )

    if isinstance(val, list):
        return [
            validate_strict_json_value(item, path=f"{path}[{idx}]") for idx, item in enumerate(val)
        ]

    if isinstance(val, dict):
        cleaned: dict[str, Any] = {}
        for k, v in val.items():
            if not isinstance(k, str):
                raise ValueError(
                    f"Strict JSON requires string keys at {path}. Key: {k!r} ({type(k).__name__})"
                )
            cleaned[k] = validate_strict_json_value(v, path=f"{path}.{k}")
        return cleaned

    raise ValueError(
        f"Strict JSON rejects non-JSON-native type at {path}: {type(val).__name__} ({val!r})"
    )


def validate_strict_json_dict(d: Any, field_name: str = "metadata") -> dict[str, Any]:
    """Validate that input is a dictionary containing only strict JSON-native values.

    Returns a deep copy of the dictionary.
    """
    if not isinstance(d, dict):
        raise ValueError(f"Field '{field_name}' must be a dictionary, got {type(d).__name__}")
    validated = validate_strict_json_value(d, path=field_name)
    assert isinstance(validated, dict)
    return validated


def strict_json_dumps(obj: Any, indent: int | None = None) -> str:
    """Serialize object to JSON enforcing strict JSON standards (no NaN/Inf, UTF-8 unescaped)."""
    return json.dumps(
        obj,
        ensure_ascii=False,
        indent=indent,
        allow_nan=False,
    )
