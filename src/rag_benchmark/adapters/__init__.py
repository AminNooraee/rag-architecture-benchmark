"""RAG Architecture Adapters package."""

from rag_benchmark.adapters.base import (
    AdapterCapabilities,
    BaseRAGAdapter,
    HealthCheckResult,
    IndexingResult,
)
from rag_benchmark.adapters.traditional import TraditionalRAGConfig, TraditionalVectorRAGAdapter

__all__ = [
    "AdapterCapabilities",
    "BaseRAGAdapter",
    "HealthCheckResult",
    "IndexingResult",
    "TraditionalRAGConfig",
    "TraditionalVectorRAGAdapter",
]
