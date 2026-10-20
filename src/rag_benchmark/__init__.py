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
    DuplicateResultError,
    GroundTruthLeakageError,
    IdentityMismatchError,
    InvalidCapabilityDeclarationError,
    InvalidLifecycleError,
    InvalidRunIdError,
    StructuredError,
    UnsupportedOperationError,
    WriterConcurrencyError,
    normalize_datetime_to_utc,
)
from rag_benchmark.runs.manifest import (
    atomic_write_json,
    create_manifest,
    get_git_commit,
    load_manifest,
    save_manifest,
)
from rag_benchmark.runs.paths import RunPaths, validate_run_id
from rag_benchmark.runs.writer import RunArtifactWriter, WriterMode
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
from rag_benchmark.schemas.strict_json import (
    strict_json_dumps,
    validate_strict_json_dict,
    validate_strict_json_value,
)
from rag_benchmark.security.sanitizer import (
    REDACTED_STR,
    is_sensitive_key,
    sanitize_dict_for_manifest,
    sanitize_error_message,
    sanitize_url,
)

__version__ = "0.1.0"

__all__ = [
    "REDACTED_STR",
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
    "DuplicateResultError",
    "EvidenceType",
    "GroundTruthLeakageError",
    "HealthCheckResult",
    "IdentityMismatchError",
    "IndexingResult",
    "InvalidCapabilityDeclarationError",
    "InvalidLifecycleError",
    "InvalidRunIdError",
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
    "WriterConcurrencyError",
    "WriterMode",
    "atomic_write_json",
    "create_manifest",
    "get_git_commit",
    "is_sensitive_key",
    "load_config",
    "load_manifest",
    "normalize_datetime_to_utc",
    "save_manifest",
    "sanitize_dict_for_manifest",
    "sanitize_error_message",
    "sanitize_url",
    "strict_json_dumps",
    "validate_run_id",
    "validate_strict_json_dict",
    "validate_strict_json_value",
]
