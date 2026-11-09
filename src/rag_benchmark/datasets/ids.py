"""Deterministic identifier helper.

Generates a stable SHA‑256 based identifier prefixed with a short human readable
token.  The function never touches randomness, process state or timestamps.
"""

from __future__ import annotations

import hashlib


def deterministic_id(prefix: str, *components: str) -> str:
    """Return a deterministic identifier.

    Parameters
    ----------
    prefix:
        Human readable token (e.g. ``"pquad-doc"``).
    *components:
        Strings that uniquely identify the source record.  They are joined by
        ``"|"`` and hashed with SHA‑256.  The full 64‑character hex digest is
        appended to the prefix with a hyphen.
    """

    if not prefix:
        raise ValueError("prefix must be a non‑empty string")
    # Normalise components to strings; ``None`` becomes the literal ``"None"``.
    joined = "|".join(str(c) for c in components)
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    return f"{prefix}-{digest}"
