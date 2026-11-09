"""Abstract dataset loader contract.
+
The Phase 3A schemas (``BenchmarkDocument`` and ``BenchmarkQuestion``) define the
canonical representation used throughout the benchmark harness.  Loaders for
individual sources must implement a small, deterministic interface so that the
builder can combine them safely.
"""

from __future__ import annotations

import abc
from collections.abc import Iterable

from rag_benchmark.schemas.document import BenchmarkDocument
from rag_benchmark.schemas.question import BenchmarkQuestion


class DatasetLoader(abc.ABC):
    """Base class for all dataset source adapters.
    +
    +    Concrete subclasses expose a *source_name* and *source_version* attribute
    +    (both strings) and implement two generator methods returning the canonical
    +    models defined in :pymod:`rag_benchmark.schemas`.
    +"""

    @property
    @abc.abstractmethod
    def source_name(self) -> str:  # pragma: no cover – abstract
        """Human‑readable identifier of the dataset source (e.g. ``"pquad"``)."""

    @property
    @abc.abstractmethod
    def source_version(self) -> str:  # pragma: no cover – abstract
        """Version string of the upstream source (e.g. ``"2023-09"``)."""

    @abc.abstractmethod
    def load_documents(self) -> Iterable[BenchmarkDocument]:  # pragma: no cover
        """Yield :class:`BenchmarkDocument` objects for the source.
        +
        +        Implementations should be deterministic – the same underlying raw file
        +        must always produce the same sequence of documents with stable IDs.
        +"""

    @abc.abstractmethod
    def load_questions(self) -> Iterable[BenchmarkQuestion]:  # pragma: no cover
        """Yield :class:`BenchmarkQuestion` objects for the source.
        +
        +        The same deterministic guarantees as :meth:`load_documents` apply.
        +"""
