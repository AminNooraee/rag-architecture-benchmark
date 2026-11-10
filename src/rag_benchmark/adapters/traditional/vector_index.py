"""Deterministic exact in-memory cosine-similarity vector index.

The baseline index is exact (no ANN approximation) and fully deterministic:
- stored vectors are immutable tuples (caller references can never mutate them);
- vector dimension is fixed by the first added vector and enforced thereafter;
- empty vectors, non-finite components, zero-norm vectors, and duplicate
  chunk IDs are all rejected explicitly;
- top-k ordering is by descending cosine similarity with an explicit
  deterministic tie-break on chunk_id;
- a zero-norm query vector yields no results (documented behavior);
- cosine similarity is computed with per-vector power-of-two scaling
  (components divided by an exact power of two before dot products and
  norms), so finite inputs such as [1e308, 1e308] can never overflow to
  NaN/Infinity; for ordinary-magnitude vectors the scaling is an exact
  exponent shift and produces bit-identical rounding to the direct formula;
  scores are clamped to [-1, 1] only after being finite.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from rag_benchmark.adapters.traditional.chunking import TextChunk


def validate_vector(vector: Sequence[float]) -> tuple[float, ...]:
    """Validate a numeric vector and return it as an immutable tuple of floats.

    Raises:
        ValueError: If the vector is empty, contains a non-numeric component,
            or contains a non-finite value (NaN/Infinity).
    """
    if isinstance(vector, (str, bytes)) or not isinstance(vector, Sequence):
        raise ValueError(f"Vector must be a sequence of numbers, got {type(vector).__name__}")
    if len(vector) == 0:
        raise ValueError("Vector must not be empty")
    components: list[float] = []
    for index, value in enumerate(vector):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"Vector component {index} must be a real number, got {type(value).__name__}"
            )
        component = float(value)
        if not math.isfinite(component):
            raise ValueError(f"Vector component {index} must be finite, got {component!r}")
        components.append(component)
    return tuple(components)


def _is_zero_vector(vector: Sequence[float]) -> bool:
    """Return True iff every component is exactly zero.

    A Euclidean norm is zero iff all components are zero; checking components
    directly avoids both overflow (huge vectors) and underflow (tiny vectors)
    in any norm computation.
    """
    return all(x == 0.0 for x in vector)


def _power_of_two_scale(max_abs: float) -> float:
    """Return the power of two that brings ``max_abs`` into [1.0, 2.0).

    Dividing a component by this factor is an exact binary operation (a pure
    exponent shift, never a mantissa rounding), so ordinary-magnitude vectors
    keep the exact rounding of the direct cosine formula while magnitudes near
    the floating-point limits are brought into a safe range.
    """
    if max_abs == 0.0:
        return 1.0
    _, exponent = math.frexp(max_abs)
    return 2.0 ** (exponent - 1)


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Numerically stable exact cosine similarity for same-length non-zero vectors.

    Each vector is first scaled by an exact power of two so its largest
    absolute component lies in [1.0, 2.0) (the scale factors cancel in the
    cosine ratio). Every intermediate product is then bounded, so finite
    inputs always produce a finite score even for magnitudes near the
    floating-point limit; for ordinary magnitudes the scale is an exact
    exponent shift and the score is bit-identical to the direct formula.

    Raises:
        ValueError: If the dimensions differ, either vector has zero norm, or
            the computation somehow yields a non-finite score.
    """
    if len(a) != len(b):
        raise ValueError(f"Dimension mismatch: {len(a)} vs {len(b)}")
    scale_a = max(abs(x) for x in a)
    scale_b = max(abs(y) for y in b)
    if scale_a == 0.0 or scale_b == 0.0:
        raise ValueError("Cosine similarity is undefined for zero-norm vectors")
    factor_a = _power_of_two_scale(scale_a)
    factor_b = _power_of_two_scale(scale_b)
    a_scaled = [x / factor_a for x in a]
    b_scaled = [y / factor_b for y in b]
    dot = math.fsum(x * y for x, y in zip(a_scaled, b_scaled, strict=True))
    norm_a = math.sqrt(math.fsum(x * x for x in a_scaled))
    norm_b = math.sqrt(math.fsum(y * y for y in b_scaled))
    if norm_a == 0.0 or norm_b == 0.0:
        raise ValueError("Cosine similarity is undefined for zero-norm vectors")
    score = dot / (norm_a * norm_b)
    if not math.isfinite(score):
        raise ValueError("Cosine similarity computation produced a non-finite score")
    if score > 1.0:
        return 1.0
    if score < -1.0:
        return -1.0
    return score


@dataclass(frozen=True)
class IndexEntry:
    """An indexed chunk paired with its stored embedding vector."""

    chunk: TextChunk
    vector: tuple[float, ...]


class ExactCosineIndex:
    """Exact in-memory cosine-similarity index over indexed chunks."""

    def __init__(self) -> None:
        self._entries: dict[str, IndexEntry] = {}
        self._dimension: int | None = None

    @property
    def dimension(self) -> int | None:
        """Vector dimension fixed by the first added vector, if any."""
        return self._dimension

    def __len__(self) -> int:
        return len(self._entries)

    def add(self, chunk: TextChunk, vector: Sequence[float]) -> None:
        """Add one indexed chunk with its embedding vector.

        Raises:
            ValueError: On invalid vectors, dimension mismatch, zero-norm
                vectors, or duplicate chunk IDs.
        """
        validated = validate_vector(vector)
        if self._dimension is None:
            self._dimension = len(validated)
        elif len(validated) != self._dimension:
            raise ValueError(
                f"Dimension mismatch: index has dimension {self._dimension}, "
                f"got vector with {len(validated)}"
            )
        if _is_zero_vector(validated):
            raise ValueError(f"Zero-norm vector rejected for chunk '{chunk.chunk_id}'")
        if chunk.chunk_id in self._entries:
            raise ValueError(f"Duplicate chunk_id '{chunk.chunk_id}' in index")
        self._entries[chunk.chunk_id] = IndexEntry(chunk=chunk, vector=validated)

    def search(
        self,
        query_vector: Sequence[float],
        top_k: int,
        score_threshold: float | None = None,
    ) -> tuple[tuple[float, IndexEntry], ...]:
        """Return up to top_k entries ranked by exact cosine similarity.

        Ordering is deterministic: descending score, then ascending chunk_id
        for equal scores. A zero-norm query vector yields no results.

        Raises:
            ValueError: On invalid query vectors or dimension mismatch.
        """
        validated = validate_vector(query_vector)
        if self._dimension is None:
            return ()
        if len(validated) != self._dimension:
            raise ValueError(
                f"Dimension mismatch: index has dimension {self._dimension}, "
                f"got query vector with {len(validated)}"
            )
        if _is_zero_vector(validated):
            return ()
        scored: list[tuple[float, IndexEntry]] = []
        for entry in self._entries.values():
            score = cosine_similarity(validated, entry.vector)
            if score_threshold is None or score >= score_threshold:
                scored.append((score, entry))
        scored.sort(key=lambda pair: (-pair[0], pair[1].chunk.chunk_id))
        return tuple(scored[:top_k])

    def clear(self) -> None:
        """Remove all entries and reset the dimension."""
        self._entries.clear()
        self._dimension = None
