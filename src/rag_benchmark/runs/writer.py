"""Run artifact writer for JSON manifests and JSONL results/errors."""

from __future__ import annotations

import json
from pathlib import Path

from rag_benchmark.errors.models import ArtifactCollisionError, StructuredError
from rag_benchmark.runs.manifest import load_manifest, save_manifest
from rag_benchmark.runs.paths import RunPaths
from rag_benchmark.schemas.manifest import RunManifest, RunStatus
from rag_benchmark.schemas.result import QueryResult, QueryStatus


class RunArtifactWriter:
    """Manages writing of run artifacts (manifest.json, raw_results.jsonl, errors.jsonl).

    Ensures UTF-8 encoding, Persian Unicode preservation (ensure_ascii=False),
    safe directory creation, and accidental overwrite protection.
    """

    def __init__(
        self,
        base_dir: str | Path,
        run_id: str,
        allow_overwrite: bool = False,
    ) -> None:
        self.paths = RunPaths(Path(base_dir), run_id)
        self.run_id = run_id
        self.allow_overwrite = allow_overwrite

        self._query_count = 0
        self._success_count = 0
        self._failed_count = 0

    def initialize(self, manifest: RunManifest) -> None:
        """Initialize run directory and write initial RUNNING manifest.

        Raises:
            ArtifactCollisionError: If manifest already exists and overwrite is not allowed.
        """
        if self.paths.manifest_path.exists() and not self.allow_overwrite:
            existing = load_manifest(self.paths.manifest_path)
            if existing.status in {RunStatus.COMPLETED, RunStatus.FAILED}:
                raise ArtifactCollisionError(
                    f"Run '{self.run_id}' already exists at {self.paths.run_dir} with status "
                    f"'{existing.status.value}'. Overwrite is disabled to prevent data loss."
                )

        self.paths.ensure_directories()
        self.write_manifest(manifest)

    def write_manifest(self, manifest: RunManifest) -> None:
        """Write or update manifest.json with Persian Unicode preserved."""
        save_manifest(manifest, self.paths.manifest_path)

    def append_result(self, result: QueryResult) -> None:
        """Append a QueryResult record to raw_results.jsonl.

        If the result contains an error or represents a failure, the error is also
        appended to errors.jsonl.
        """
        self.paths.ensure_directories()

        line = json.dumps(result.model_dump(mode="json"), ensure_ascii=False)
        with open(self.paths.raw_results_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

        self._query_count += 1
        if result.status == QueryStatus.SUCCESS:
            self._success_count += 1
        else:
            self._failed_count += 1

        if result.error is not None:
            self.append_error(result.error)

    def append_error(self, error: StructuredError) -> None:
        """Append a StructuredError record to errors.jsonl."""
        self.paths.ensure_directories()

        line = json.dumps(error.model_dump(mode="json"), ensure_ascii=False)
        with open(self.paths.errors_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def finalize(
        self,
        status: RunStatus = RunStatus.COMPLETED,
    ) -> RunManifest:
        """Finalize the manifest with end time, status, and query totals."""
        if not self.paths.manifest_path.exists():
            raise FileNotFoundError(
                f"Cannot finalize: Manifest not found at {self.paths.manifest_path}"
            )

        current = load_manifest(self.paths.manifest_path)
        finalized = current.finalize(
            status=status,
            total_queries=self._query_count,
            successful_queries=self._success_count,
            failed_queries=self._failed_count,
        )
        self.write_manifest(finalized)
        return finalized

    def read_results(self) -> list[QueryResult]:
        """Read all query results from raw_results.jsonl if present."""
        if not self.paths.raw_results_path.exists():
            return []

        results: list[QueryResult] = []
        with open(self.paths.raw_results_path, encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    data = json.loads(stripped)
                    results.append(QueryResult.model_validate(data))
        return results

    def read_errors(self) -> list[StructuredError]:
        """Read all errors from errors.jsonl if present."""
        if not self.paths.errors_path.exists():
            return []

        errors: list[StructuredError] = []
        with open(self.paths.errors_path, encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    data = json.loads(stripped)
                    errors.append(StructuredError.model_validate(data))
        return errors
