"""Run artifact writer managing durable, single-writer benchmark outputs."""

from __future__ import annotations

import ctypes
import json
import os
import platform
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Self

from rag_benchmark.errors.models import (
    ArtifactCollisionError,
    DuplicateResultError,
    IdentityMismatchError,
    InvalidLifecycleError,
    StructuredError,
    WriterConcurrencyError,
)
from rag_benchmark.runs.manifest import load_manifest, save_manifest
from rag_benchmark.runs.paths import RunPaths
from rag_benchmark.schemas.manifest import RunManifest, RunStatus
from rag_benchmark.schemas.result import QueryResult, QueryStatus
from rag_benchmark.schemas.strict_json import strict_json_dumps


class WriterMode(StrEnum):
    """Lifecycle initialization mode for RunArtifactWriter."""

    CREATE = "create"
    RESUME = "resume"
    OVERWRITE = "overwrite"


def _is_pid_alive(pid: int) -> bool:
    """Check whether a given PID corresponds to a running process on this host."""
    if pid <= 0:
        return False
    # Windows check using OpenProcess
    if platform.system() == "Windows":
        try:
            kernel32 = ctypes.windll.kernel32
            SYNCHRONIZE = 0x00100000
            process = kernel32.OpenProcess(SYNCHRONIZE, False, pid)
            if process:
                kernel32.CloseHandle(process)
                return True
            return False
        except Exception:
            return False
    # POSIX check using os.kill(pid, 0)
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


class RunArtifactWriter:
    """Single-writer durable artifact sink for benchmark runs.

    Guarantees:
      - Explicit lifecycle modes: CREATE, RESUME, OVERWRITE
      - Exclusive single-writer process ownership via writer.lock
      - Run identity binding (run_id, system, variant)
      - Question uniqueness (run_id, question_id) with resume reconstruction
      - Atomic manifest replacement and strict JSON serialization
    """

    def __init__(
        self,
        base_dir: str | Path,
        run_id: str,
        mode: WriterMode = WriterMode.CREATE,
    ) -> None:
        self.paths = RunPaths(Path(base_dir), run_id)
        self.run_id = run_id
        self.mode = mode

        self._initialized = False
        self._finalized = False
        self._owns_lock = False

        self.bound_run_id: str = ""
        self.bound_system: str = ""
        self.bound_variant: str = ""

        self._query_count = 0
        self._success_count = 0
        self._failed_count = 0
        self._recorded_question_ids: set[str] = set()

    @classmethod
    def create(
        cls,
        base_dir: str | Path,
        manifest: RunManifest,
    ) -> RunArtifactWriter:
        """Create a new run. Fails if any artifact already exists."""
        writer = cls(base_dir=base_dir, run_id=manifest.run_id, mode=WriterMode.CREATE)
        writer.initialize(manifest)
        return writer

    @classmethod
    def resume(
        cls,
        base_dir: str | Path,
        run_id: str,
    ) -> RunArtifactWriter:
        """Resume an existing RUNNING run, reconstructing durable state."""
        writer = cls(base_dir=base_dir, run_id=run_id, mode=WriterMode.RESUME)
        writer.initialize()
        return writer

    @classmethod
    def overwrite(
        cls,
        base_dir: str | Path,
        manifest: RunManifest,
    ) -> RunArtifactWriter:
        """Overwrite an existing run, replacing all previous artifacts."""
        writer = cls(base_dir=base_dir, run_id=manifest.run_id, mode=WriterMode.OVERWRITE)
        writer.initialize(manifest)
        return writer

    def _acquire_lock(self) -> None:
        """Acquire exclusive writer ownership of the run directory."""
        self.paths.ensure_directories()
        lock_path = self.paths.lock_path

        lock_data = {
            "pid": os.getpid(),
            "host": platform.node(),
            "acquired_at": datetime.now(UTC).isoformat(),
            "run_id": self.run_id,
        }

        # Attempt atomic file creation
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(strict_json_dumps(lock_data))
            self._owns_lock = True
            return
        except FileExistsError:
            pass

        # If file exists, check if holder process is still alive on this machine
        is_stale = False
        existing_pid: int | None = None
        try:
            with open(lock_path, encoding="utf-8") as f:
                info = json.load(f)
                existing_pid = info.get("pid")
                existing_host = info.get("host")
                if existing_host == platform.node() and existing_pid:
                    if not _is_pid_alive(existing_pid):
                        is_stale = True
        except Exception:
            pass

        if is_stale:
            try:
                lock_path.unlink()
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(strict_json_dumps(lock_data))
                self._owns_lock = True
                return
            except Exception as e:
                raise WriterConcurrencyError(
                    f"Failed to acquire stale lock for run '{self.run_id}': {e}"
                ) from e

        raise WriterConcurrencyError(
            f"Run '{self.run_id}' is owned by active writer (PID {existing_pid}). "
            "Parallel query workers must route results through the single owning writer."
        )

    def _release_lock(self) -> None:
        """Release the exclusive writer lock if owned by this instance."""
        if self._owns_lock and self.paths.lock_path.exists():
            try:
                self.paths.lock_path.unlink()
            except OSError:
                pass
            self._owns_lock = False

    def initialize(self, manifest: RunManifest | None = None) -> None:
        """Initialize writer under the configured mode (CREATE, RESUME, or OVERWRITE)."""
        if self._initialized:
            raise InvalidLifecycleError(f"Writer for run '{self.run_id}' is already initialized.")

        self.paths.ensure_directories()
        self._acquire_lock()

        try:
            if self.mode == WriterMode.CREATE:
                if manifest is None:
                    raise ValueError("CREATE mode requires an initial RunManifest.")
                if manifest.run_id != self.run_id:
                    raise IdentityMismatchError(
                        f"Manifest run_id '{manifest.run_id}' != writer run_id '{self.run_id}'"
                    )

                # Fail if ANY artifact exists
                if (
                    self.paths.manifest_path.exists()
                    or self.paths.raw_results_path.exists()
                    or self.paths.errors_path.exists()
                ):
                    raise ArtifactCollisionError(
                        f"Cannot CREATE run '{self.run_id}': artifacts exist."
                    )

                self._bind_identity(manifest)
                save_manifest(manifest, self.paths.manifest_path)
                self._initialized = True

            elif self.mode == WriterMode.RESUME:
                if not self.paths.manifest_path.exists():
                    raise FileNotFoundError(f"Cannot RESUME '{self.run_id}': manifest missing.")

                existing_manifest = load_manifest(self.paths.manifest_path)
                if existing_manifest.run_id != self.run_id:
                    raise IdentityMismatchError(
                        f"Manifest run_id '{existing_manifest.run_id}' != writer '{self.run_id}'"
                    )

                if existing_manifest.status != RunStatus.RUNNING:
                    raise InvalidLifecycleError(
                        f"Cannot RESUME '{self.run_id}': status '{existing_manifest.status.value}'"
                    )

                self._bind_identity(existing_manifest)
                self._reconstruct_durable_state()
                self._initialized = True

            elif self.mode == WriterMode.OVERWRITE:
                if manifest is None:
                    raise ValueError("OVERWRITE mode requires an initial RunManifest.")
                if manifest.run_id != self.run_id:
                    raise IdentityMismatchError(
                        f"Manifest run_id '{manifest.run_id}' != writer run_id '{self.run_id}'"
                    )

                # Cleanly remove old results and errors to guarantee clean state
                if self.paths.raw_results_path.exists():
                    self.paths.raw_results_path.unlink()
                if self.paths.errors_path.exists():
                    self.paths.errors_path.unlink()
                if self.paths.manifest_path.exists():
                    self.paths.manifest_path.unlink()

                self._bind_identity(manifest)
                save_manifest(manifest, self.paths.manifest_path)
                self._initialized = True
        except Exception:
            self._release_lock()
            raise

    def _bind_identity(self, manifest: RunManifest) -> None:
        """Bind writer to manifest identity."""
        self.bound_run_id = manifest.run_id
        self.bound_system = manifest.system
        self.bound_variant = manifest.variant

    def _reconstruct_durable_state(self) -> None:
        """Reconstruct counters and seen question IDs from durable raw_results.jsonl."""
        self._query_count = 0
        self._success_count = 0
        self._failed_count = 0
        self._recorded_question_ids.clear()

        if self.paths.raw_results_path.exists():
            results = self.read_results()
            for r in results:
                self._query_count += 1
                if r.status == QueryStatus.SUCCESS:
                    self._success_count += 1
                else:
                    self._failed_count += 1
                self._recorded_question_ids.add(r.question_id)

    def append_result(self, result: QueryResult) -> None:
        """Append a QueryResult record to raw_results.jsonl.

        Validates:
          - Writer is initialized and actively RUNNING
          - System and variant match bound run identity
          - question_id has not already been recorded for this run
        """
        if not self._initialized:
            raise InvalidLifecycleError("Cannot append result: writer is not initialized.")
        if self._finalized:
            raise InvalidLifecycleError("Cannot append result: writer has already been finalized.")

        # Identity binding verification
        if result.system != self.bound_system:
            raise IdentityMismatchError(
                f"Result system '{result.system}' != run system '{self.bound_system}'"
            )
        if result.variant != self.bound_variant:
            raise IdentityMismatchError(
                f"Result variant '{result.variant}' != run variant '{self.bound_variant}'"
            )

        # Result uniqueness verification: (run_id, question_id)
        if result.question_id in self._recorded_question_ids:
            raise DuplicateResultError(
                f"Duplicate question_id '{result.question_id}' in run '{self.run_id}'. "
                "Each question may have only one canonical final result per run."
            )

        # If an error is nested, verify its identity matches as well
        if result.error is not None:
            if result.error.question_id and result.error.question_id != result.question_id:
                raise IdentityMismatchError(
                    f"Error question_id '{result.error.question_id}' != '{result.question_id}'"
                )
            if result.error.system and result.error.system != self.bound_system:
                raise IdentityMismatchError(
                    f"Error system '{result.error.system}' != run system '{self.bound_system}'"
                )
            if result.error.variant and result.error.variant != self.bound_variant:
                raise IdentityMismatchError(
                    f"Error variant '{result.error.variant}' != run variant '{self.bound_variant}'"
                )

        line = strict_json_dumps(result.model_dump(mode="json"))
        with open(self.paths.raw_results_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()

        self._recorded_question_ids.add(result.question_id)
        self._query_count += 1
        if result.status == QueryStatus.SUCCESS:
            self._success_count += 1
        else:
            self._failed_count += 1

        if result.error is not None:
            self._append_error_internal(result.error)

    def append_error(self, error: StructuredError) -> None:
        """Explicitly append a standalone StructuredError record to errors.jsonl."""
        if not self._initialized:
            raise InvalidLifecycleError("Cannot append error: writer is not initialized.")
        if self._finalized:
            raise InvalidLifecycleError("Cannot append error: writer has already been finalized.")

        if error.system and error.system != self.bound_system:
            raise IdentityMismatchError(
                f"Error system '{error.system}' != run system '{self.bound_system}'"
            )
        if error.variant and error.variant != self.bound_variant:
            raise IdentityMismatchError(
                f"Error variant '{error.variant}' != run variant '{self.bound_variant}'"
            )

        self._append_error_internal(error)

    def _append_error_internal(self, error: StructuredError) -> None:
        """Internal append helper for errors.jsonl."""
        line = strict_json_dumps(error.model_dump(mode="json"))
        with open(self.paths.errors_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()

    def finalize(
        self,
        status: RunStatus = RunStatus.COMPLETED,
    ) -> RunManifest:
        """Finalize the manifest with status, timestamps, and query totals, releasing the lock."""
        if not self._initialized:
            raise InvalidLifecycleError("Cannot finalize: writer is not initialized.")
        if self._finalized:
            raise InvalidLifecycleError("Cannot finalize: writer has already been finalized.")

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
        save_manifest(finalized, self.paths.manifest_path)
        self._finalized = True
        self._release_lock()
        return finalized

    def close(self) -> None:
        """Release any held locks."""
        self._release_lock()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

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
