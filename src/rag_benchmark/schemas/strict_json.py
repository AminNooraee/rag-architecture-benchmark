"""Strict JSON-native validation contracts and deep immutability utilities."""

from __future__ import annotations

import json
import math
from enum import Enum
from typing import Any, SupportsIndex


class FrozenDict(dict[str, Any]):
    """Recursively immutable dictionary representation for strict JSON metadata."""

    def __setitem__(self, key: str, value: Any) -> None:
        raise TypeError("FrozenDict is immutable; modification is not allowed.")

    def __delitem__(self, key: str) -> None:
        raise TypeError("FrozenDict is immutable; modification is not allowed.")

    def pop(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError("FrozenDict is immutable; modification is not allowed.")

    def popitem(self) -> Any:
        raise TypeError("FrozenDict is immutable; modification is not allowed.")

    def clear(self) -> None:
        raise TypeError("FrozenDict is immutable; modification is not allowed.")

    def update(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("FrozenDict is immutable; modification is not allowed.")

    def setdefault(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError("FrozenDict is immutable; modification is not allowed.")

    def __ior__(self, other: Any) -> Any:  # type: ignore[misc]
        raise TypeError("FrozenDict is immutable; modification is not allowed.")

    def __copy__(self) -> FrozenDict:
        return self

    def __deepcopy__(self, memo: Any = None) -> FrozenDict:
        return self


class FrozenList(list[Any]):
    """Recursively immutable list representation for strict JSON metadata."""

    def __setitem__(self, index: Any, value: Any) -> None:
        raise TypeError("FrozenList is immutable; modification is not allowed.")

    def __delitem__(self, index: Any) -> None:
        raise TypeError("FrozenList is immutable; modification is not allowed.")

    def append(self, item: Any) -> None:
        raise TypeError("FrozenList is immutable; modification is not allowed.")

    def extend(self, items: Any) -> None:
        raise TypeError("FrozenList is immutable; modification is not allowed.")

    def insert(self, index: SupportsIndex, item: Any) -> None:
        raise TypeError("FrozenList is immutable; modification is not allowed.")

    def pop(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError("FrozenList is immutable; modification is not allowed.")

    def remove(self, item: Any) -> None:
        raise TypeError("FrozenList is immutable; modification is not allowed.")

    def clear(self) -> None:
        raise TypeError("FrozenList is immutable; modification is not allowed.")

    def sort(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("FrozenList is immutable; modification is not allowed.")

    def reverse(self) -> None:
        raise TypeError("FrozenList is immutable; modification is not allowed.")

    def __iadd__(self, other: Any) -> Any:  # type: ignore[misc]
        raise TypeError("FrozenList is immutable; modification is not allowed.")

    def __imul__(self, other: Any) -> Any:  # type: ignore[misc]
        raise TypeError("FrozenList is immutable; modification is not allowed.")

    def __copy__(self) -> FrozenList:
        return self

    def __deepcopy__(self, memo: Any = None) -> FrozenList:
        return self


def validate_strict_json_value(val: Any, path: str = "value") -> Any:
    """Validate that a value conforms strictly to JSON-native types.

    Permitted types:
      - None
      - bool
      - int (excluding bool)
      - float (finite only: no NaN, Inf, -Inf)
      - str (including StrEnum which behaves as a string)
      - list (tuples and sets are rejected)
      - dict (all keys MUST be strings)

    Rejected types:
      - tuple, set, frozenset
      - datetime, date, time
      - non-string Enum (must be explicitly string-converted prior to validation)
      - float('nan'), float('inf'), float('-inf')
      - non-string dictionary keys
      - custom classes / objects

    Returns a deeply immutable (FrozenDict/FrozenList) representation of the validated value.
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
        return str(val)

    # Reject non-string enums
    if isinstance(val, Enum):
        raise ValueError(
            f"Strict JSON rejects non-string Enum at {path}: {val!r} of type {type(val).__name__}"
        )

    if isinstance(val, tuple):
        raise ValueError(
            f"Strict JSON rejects tuples at {path} (tuples silently degrade to lists in JSON)"
        )

    if isinstance(val, list):
        items = [
            validate_strict_json_value(item, path=f"{path}[{idx}]") for idx, item in enumerate(val)
        ]
        return FrozenList(items)

    if isinstance(val, dict):
        cleaned: dict[str, Any] = {}
        for k, v in val.items():
            if not isinstance(k, str):
                raise ValueError(
                    f"Strict JSON requires string keys at {path}. Key: {k!r} ({type(k).__name__})"
                )
            cleaned[k] = validate_strict_json_value(v, path=f"{path}.{k}")
        return FrozenDict(cleaned)

    raise ValueError(
        f"Strict JSON rejects non-JSON-native type at {path}: {type(val).__name__} ({val!r})"
    )


def validate_strict_json_dict(d: Any, field_name: str = "metadata") -> dict[str, Any]:
    """Validate that input is a dictionary containing only strict JSON-native values.

    Returns a recursively immutable FrozenDict.
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
