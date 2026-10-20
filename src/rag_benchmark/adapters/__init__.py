"""RAG Architecture Adapters package."""

from rag_benchmark.adapters.base import (
    AdapterCapabilities,
    BaseRAGAdapter,
    HealthCheckResult,
    IndexingResult,
)

__all__ = [
    "AdapterCapabilities",
    "BaseRAGAdapter",
    "HealthCheckResult",
    "IndexingResult",
]
