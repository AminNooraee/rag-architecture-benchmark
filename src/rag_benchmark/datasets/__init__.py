"""Dataset ingestion package for Phase 3B.

Provides abstract loader definitions, concrete source adapters, a deterministic
ID utility, a dataset builder, manifest and statistics models, and a tiny CLI.
"""

from .base import DatasetLoader
from .builder import BenchmarkDatasetBuilder
from .ids import deterministic_id
from .manifest import BenchmarkDatasetManifest
from .statistics import DatasetStatistics

__all__ = [
    "DatasetLoader",
    "deterministic_id",
    "BenchmarkDatasetBuilder",
    "BenchmarkDatasetManifest",
    "DatasetStatistics",
]
