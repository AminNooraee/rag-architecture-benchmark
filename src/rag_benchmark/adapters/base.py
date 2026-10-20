"""Abstract Base RAG Adapter interface and contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from rag_benchmark.config.models import BenchmarkConfig
from rag_benchmark.errors.models import (
    InvalidCapabilityDeclarationError,
    UnsupportedOperationError,
)
from rag_benchmark.schemas.document import BenchmarkDocument
from rag_benchmark.schemas.question import AdapterQueryInput
from rag_benchmark.schemas.result import QueryResult


class AdapterCapabilities(BaseModel):
    """Structured declaration of features supported by a RAG system adapter.

    Note: Asynchronous execution (supports_async) is intentionally deferred until
    Phase 4 when an explicit async query contract is scheduled for introduction.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", validate_assignment=True)

    supports_incremental_insert: bool = Field(
        default=False,
        description=(
            "Whether the architecture can incrementally insert new documents without reindexing."
        ),
    )
    supports_update: bool = Field(
        default=False,
        description="Whether the architecture can update existing documents in place.",
    )
    supports_delete: bool = Field(
        default=False,
        description="Whether the architecture can remove documents by ID.",
    )
    supports_persistence: bool = Field(
        default=False,
        description="Whether the index is persisted to disk or an external database.",
    )
    supports_streaming: bool = Field(
        default=False,
        description="Whether the adapter supports token-by-token streaming responses.",
    )
    supports_native_citations: bool = Field(
        default=False,
        description="Whether the architecture natively generates grounded citation references.",
    )
    extra_capabilities: dict[str, bool] = Field(
        default_factory=dict,
        description="Additional system-specific capability flags.",
    )


class IndexingResult(BaseModel):
    """Summary metrics from document indexing."""

    model_config = ConfigDict(frozen=True, extra="forbid", validate_assignment=True)

    indexed_documents: int = Field(default=0, ge=0)
    indexed_chunks: int = Field(default=0, ge=0)
    elapsed_ms: float = Field(default=0.0, ge=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class HealthCheckResult(BaseModel):
    """Outcome of adapter health and connectivity verification."""

    model_config = ConfigDict(frozen=True, extra="forbid", validate_assignment=True)

    healthy: bool
    system: str
    message: str = "OK"
    details: dict[str, Any] = Field(default_factory=dict)


class BaseRAGAdapter(ABC):
    """Abstract base adapter for all RAG architectures evaluated in the benchmark.

    FAIRNESS ARCHITECTURAL CONTRACT:
    The adapter is only responsible for:
      - input normalization
      - configuration
      - execution
      - output normalization
      - metrics collection

    It must NOT secretly add:
      - reranking
      - HyDE (Hypothetical Document Embeddings)
      - query rewriting
      - multi-query expansion
      - additional retrieval stages
    unless a benchmark variant explicitly enables and tests those features.
    """

    @property
    @abstractmethod
    def system_name(self) -> str:
        """Name of the RAG architecture (e.g. 'traditional_rag', 'lightrag', 'neo4j_graphrag')."""
        ...

    @property
    @abstractmethod
    def variant_name(self) -> str:
        """Variant name of this adapter instance (e.g. 'baseline', 'dense_512')."""
        ...

    @property
    @abstractmethod
    def capabilities(self) -> AdapterCapabilities:
        """Report features supported by this adapter implementation."""
        ...

    def validate_capabilities(self) -> None:
        """Verify that any declared capability actually has an overridden implementation.

        Fails early if an adapter advertises capabilities without providing the required
        methods.
        """
        caps = self.capabilities
        if caps.supports_incremental_insert and type(self).insert is BaseRAGAdapter.insert:
            raise InvalidCapabilityDeclarationError(
                f"Adapter '{self.system_name}' declares supports_incremental_insert=True "
                "but does not override the insert() method."
            )
        if caps.supports_update and type(self).update is BaseRAGAdapter.update:
            raise InvalidCapabilityDeclarationError(
                f"Adapter '{self.system_name}' declares supports_update=True "
                "but does not override the update() method."
            )
        if caps.supports_delete and type(self).delete is BaseRAGAdapter.delete:
            raise InvalidCapabilityDeclarationError(
                f"Adapter '{self.system_name}' declares supports_delete=True "
                "but does not override the delete() method."
            )

    @abstractmethod
    def prepare(self, config: BenchmarkConfig) -> None:
        """Initialize and configure adapter with benchmark parameters."""
        ...

    @abstractmethod
    def index(self, documents: Sequence[BenchmarkDocument]) -> IndexingResult:
        """Index a collection of Persian documents into the RAG architecture."""
        ...

    @abstractmethod
    def query(self, query_input: AdapterQueryInput) -> QueryResult:
        """Execute a single query through the RAG pipeline.

        CRITICAL: Receives only AdapterQueryInput (ground truth strictly excluded).
        Must return a QueryResult even if execution encounters errors (status=FAILED).
        """
        ...

    @abstractmethod
    def health(self) -> HealthCheckResult:
        """Perform a quick health verification (e.g. database connectivity, model access)."""
        ...

    @abstractmethod
    def cleanup(self) -> None:
        """Release resources, close connections, or clear temporary structures."""
        ...

    # Optional lifecycle operations (only valid if declared in capabilities)

    def insert(self, documents: Sequence[BenchmarkDocument]) -> IndexingResult:
        """Incrementally insert documents into an existing index."""
        if not self.capabilities.supports_incremental_insert:
            raise UnsupportedOperationError(
                f"RAG system '{self.system_name}' does not support incremental insert."
            )
        raise NotImplementedError(
            f"insert() declared supported by '{self.system_name}' but not implemented."
        )

    def update(self, documents: Sequence[BenchmarkDocument]) -> IndexingResult:
        """Update existing documents within the index."""
        if not self.capabilities.supports_update:
            raise UnsupportedOperationError(
                f"RAG system '{self.system_name}' does not support document update."
            )
        raise NotImplementedError(
            f"update() declared supported by '{self.system_name}' but not implemented."
        )

    def delete(self, document_ids: Sequence[str]) -> None:
        """Delete documents from the index by ID."""
        if not self.capabilities.supports_delete:
            raise UnsupportedOperationError(
                f"RAG system '{self.system_name}' does not support document deletion."
            )
        raise NotImplementedError(
            f"delete() declared supported by '{self.system_name}' but not implemented."
        )
