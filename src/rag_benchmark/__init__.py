"""Production-oriented Persian RAG architecture benchmark harness."""

from rag_benchmark.adapters.base import (
    AdapterCapabilities,
    BaseRAGAdapter,
    HealthCheckResult,
    IndexingResult,
)
from rag_benchmark.config.loader import load_config
from rag_benchmark.config.models import BenchmarkConfig
from rag_benchmark.errors.models import (
    ArtifactCollisionError,
    BenchmarkErrorType,
    BenchmarkException,
    ConfigurationError,
    StructuredError,
    UnsupportedOperationError,
)
from rag_benchmark.runs.manifest import create_manifest, load_manifest, save_manifest
from rag_benchmark.runs.paths import RunPaths
from rag_benchmark.runs.writer import RunArtifactWriter
from rag_benchmark.schemas.document import BenchmarkDocument, DocumentSection
from rag_benchmark.schemas.evidence import EvidenceType, RetrievedEvidence
from rag_benchmark.schemas.manifest import RunManifest, RunStatus, RuntimeMetadata
from rag_benchmark.schemas.question import (
    AdapterQueryInput,
    BenchmarkQuestion,
    DifficultyLevel,
    QuestionScope,
    QuestionType,
)
from rag_benchmark.schemas.result import (
    QueryResult,
    QueryStatus,
    QueryTiming,
    QueryUsage,
)

__version__ = "0.1.0"

__all__ = [
    "AdapterCapabilities",
    "AdapterQueryInput",
    "ArtifactCollisionError",
    "BaseRAGAdapter",
    "BenchmarkConfig",
    "BenchmarkDocument",
    "BenchmarkErrorType",
    "BenchmarkException",
    "BenchmarkQuestion",
    "ConfigurationError",
    "DifficultyLevel",
    "DocumentSection",
    "EvidenceType",
    "HealthCheckResult",
    "IndexingResult",
    "QueryResult",
    "QueryStatus",
    "QueryTiming",
    "QueryUsage",
    "QuestionScope",
    "QuestionType",
    "RetrievedEvidence",
    "RunArtifactWriter",
    "RunManifest",
    "RunPaths",
    "RunStatus",
    "RuntimeMetadata",
    "StructuredError",
    "UnsupportedOperationError",
    "create_manifest",
    "load_config",
    "load_manifest",
    "save_manifest",
]
