"""Deterministic exact in-memory cosine-similarity vector index.

The baseline index is exact (no ANN approximation) and fully deterministic:
- stored vectors are immutable tuples (caller references can never mutate them);
- vector dimension is fixed by the first added vector and enforced thereafter;
- empty vectors, non-finite components, zero-norm vectors, and duplicate
  chunk IDs are all rejected explicitly;
- top-k ordering is by descending cosine similarity with an explicit
  deterministic tie-break on chunk_id;
- a zero-norm query vector yields no results (documented behavior).
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


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Exact cosine similarity between two same-length, non-zero vectors.

    Raises:
        ValueError: If the dimensions differ or either vector has zero norm.
    """
    if len(a) != len(b):
        raise ValueError(f"Dimension mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        raise ValueError("Cosine similarity is undefined for zero-norm vectors")
    return dot / (norm_a * norm_b)


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
        if math.fsum(x * x for x in validated) == 0.0:
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
        if math.fsum(x * x for x in validated) == 0.0:
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
