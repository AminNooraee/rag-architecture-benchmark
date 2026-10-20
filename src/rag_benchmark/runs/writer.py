"""Run artifact writer managing durable, single-writer benchmark outputs."""

from __future__ import annotations

import ctypes
import json
import os
import platform
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Self

from rag_benchmark.errors.models import (
    ArtifactCollisionError,
    ArtifactConsistencyError,
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


class WriterLifecycle(StrEnum):
    """Explicit lifecycle states for RunArtifactWriter."""

    NEW = "new"
    ACTIVE = "active"
    FINALIZED = "finalized"
    CLOSED = "closed"


@contextmanager
def _guard_lock(guard_path: Path, timeout: float = 3.0) -> Iterator[None]:
    """Acquire an exclusive OS file lock on guard_path to serialize acquisition/recovery."""
    guard_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(guard_path), os.O_CREAT | os.O_RDWR)
    f = os.fdopen(fd, "r+", encoding="utf-8")
    f.seek(0, os.SEEK_END)
    if f.tell() == 0:
        f.write("0")
        f.flush()
    f.seek(0)

    start = time.monotonic()
    locked = False
    is_windows = platform.system() == "Windows"

    while True:
        try:
            if is_windows:
                import msvcrt

                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(  # type: ignore[attr-defined]
                    f.fileno(),
                    fcntl.LOCK_EX | fcntl.LOCK_NB,  # type: ignore[attr-defined]
                )
            locked = True
            break
        except (OSError, BlockingIOError) as e:
            if time.monotonic() - start > timeout:
                f.close()
                raise WriterConcurrencyError(
                    f"Acquisition guard '{guard_path.name}' timed out after {timeout}s."
                ) from e
            time.sleep(0.01)

    try:
        yield
    finally:
        if locked:
            try:
                if is_windows:
                    import msvcrt

                    f.seek(0)
                    msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)  # type: ignore[attr-defined]
            except OSError:
                pass
        f.close()


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
      - Explicit writer lifecycle: NEW -> ACTIVE -> FINALIZED / CLOSED
      - Exclusive single-writer process ownership with race-safe atomic claim
      - Run identity binding (run_id, system, variant)
      - Question uniqueness (run_id, question_id) with strict resume reconstruction
      - Strict cross-file validation on RESUME (detecting orphan, missing, or corrupt records)
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

        self.lifecycle = WriterLifecycle.NEW
        self._owns_lock = False
        self._owner_token = uuid.uuid4().hex

        self.bound_run_id: str = ""
        self.bound_system: str = ""
        self.bound_variant: str = ""

        self._query_count = 0
        self._success_count = 0
        self._failed_count = 0
        self._recorded_question_ids: set[str] = set()

    @property
    def is_active(self) -> bool:
        """Return True if the writer is actively initialized and holds its lock."""
        return self.lifecycle == WriterLifecycle.ACTIVE and self._owns_lock

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
        """Overwrite an existing run, replacing previous Phase 3A artifacts."""
        writer = cls(base_dir=base_dir, run_id=manifest.run_id, mode=WriterMode.OVERWRITE)
        writer.initialize(manifest)
        return writer

    def _acquire_lock(self) -> None:
        """Acquire exclusive writer ownership of the run directory.

        Serializes lock acquisition and stale-lock reclamation via an exclusive OS
        file lock on writer.lock.guard before inspecting or mutating writer.lock.
        """
        self.paths.ensure_directories()
        lock_path = self.paths.lock_path
        guard_path = self.paths.guard_path

        lock_data = {
            "owner_token": self._owner_token,
            "pid": os.getpid(),
            "host": platform.node(),
            "acquired_at": datetime.now(UTC).isoformat(),
            "run_id": self.run_id,
        }

        with _guard_lock(guard_path):
            # 1. If writer.lock does not exist, create it atomically
            if not lock_path.exists():
                try:
                    fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
                    with os.fdopen(fd, "w", encoding="utf-8") as f:
                        f.write(strict_json_dumps(lock_data))
                        f.flush()
                    self._owns_lock = True
                    return
                except FileExistsError as e:
                    raise WriterConcurrencyError(
                        f"Lock collision during creation for run '{self.run_id}': {e}"
                    ) from e

            # 2. writer.lock exists: inspect owner info
            is_stale = False
            existing_pid: int | None = None
            existing_token: str | None = None
            existing_host: str | None = None
            try:
                with open(lock_path, encoding="utf-8") as f:
                    info = json.load(f)
                    existing_pid = info.get("pid")
                    existing_host = info.get("host")
                    existing_token = info.get("owner_token")
                    if existing_host == platform.node() and existing_pid:
                        if not _is_pid_alive(existing_pid):
                            is_stale = True
            except Exception:
                # File may be mid-creation or unreadable; treat as active
                pass

            # 3. If stale, re-verify and reclaim while holding the guard
            if is_stale and existing_pid is not None and existing_token is not None:
                # Re-verify lock file still matches inspected token
                try:
                    with open(lock_path, encoding="utf-8") as f:
                        verify_info = json.load(f)
                    if verify_info.get("owner_token") != existing_token:
                        raise WriterConcurrencyError(
                            f"Stale lock for run '{self.run_id}' changed before removal."
                        )
                    lock_path.unlink()
                except OSError as e:
                    raise WriterConcurrencyError(
                        f"Failed to remove stale lock for run '{self.run_id}': {e}"
                    ) from e

                # Create new writer.lock using O_CREAT | O_EXCL
                try:
                    fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
                    with os.fdopen(fd, "w", encoding="utf-8") as f:
                        f.write(strict_json_dumps(lock_data))
                        f.flush()
                    self._owns_lock = True
                    return
                except Exception as e:
                    raise WriterConcurrencyError(
                        f"Failed to establish new lock for run '{self.run_id}': {e}"
                    ) from e

            # 4. Otherwise, holder is active (or on remote host)
            raise WriterConcurrencyError(
                f"Run '{self.run_id}' is owned by active writer (PID {existing_pid}). "
                "Parallel query workers must route results through the single owning writer."
            )

    def _verify_lock_held(self) -> bool:
        """Verify that this writer instance still owns the lock file on disk."""
        if not self._owns_lock:
            return False
        if not self.paths.lock_path.exists():
            return False
        try:
            with open(self.paths.lock_path, encoding="utf-8") as f:
                info = json.load(f)
                return bool(info.get("owner_token") == self._owner_token)
        except Exception:
            return False

    def _release_lock(self) -> None:
        """Release the exclusive writer lock if owned by this instance."""
        if self._owns_lock:
            if self._verify_lock_held():
                try:
                    self.paths.lock_path.unlink()
                except OSError:
                    pass
            self._owns_lock = False

    def initialize(self, manifest: RunManifest | None = None) -> None:
        """Initialize writer under the configured mode (CREATE, RESUME, or OVERWRITE)."""
        if self.lifecycle == WriterLifecycle.CLOSED:
            raise InvalidLifecycleError(
                f"Cannot initialize writer for run '{self.run_id}': writer is permanently CLOSED."
            )
        if self.lifecycle == WriterLifecycle.FINALIZED:
            raise InvalidLifecycleError(
                f"Cannot initialize writer for run '{self.run_id}': writer is FINALIZED."
            )
        if self.lifecycle != WriterLifecycle.NEW:
            raise InvalidLifecycleError(
                f"Writer for run '{self.run_id}' is already initialized "
                f"(state: {self.lifecycle.value})."
            )

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
                self.lifecycle = WriterLifecycle.ACTIVE

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
                self._reconstruct_and_validate_durable_state()
                self.lifecycle = WriterLifecycle.ACTIVE

            elif self.mode == WriterMode.OVERWRITE:
                if manifest is None:
                    raise ValueError("OVERWRITE mode requires an initial RunManifest.")
                if manifest.run_id != self.run_id:
                    raise IdentityMismatchError(
                        f"Manifest run_id '{manifest.run_id}' != writer run_id '{self.run_id}'"
                    )

                # Sequential cleanup of previous Phase 3A artifacts
                if self.paths.raw_results_path.exists():
                    self.paths.raw_results_path.unlink()
                if self.paths.errors_path.exists():
                    self.paths.errors_path.unlink()
                if self.paths.manifest_path.exists():
                    self.paths.manifest_path.unlink()

                self._bind_identity(manifest)
                save_manifest(manifest, self.paths.manifest_path)
                self.lifecycle = WriterLifecycle.ACTIVE

        except Exception:
            self._release_lock()
            self.lifecycle = WriterLifecycle.CLOSED
            raise

    def _bind_identity(self, manifest: RunManifest) -> None:
        """Bind writer to manifest identity."""
        self.bound_run_id = manifest.run_id
        self.bound_system = manifest.system
        self.bound_variant = manifest.variant

    def _reconstruct_and_validate_durable_state(self) -> None:
        """Reconstruct counters and perform consistency checks on durable artifacts."""
        self._query_count = 0
        self._success_count = 0
        self._failed_count = 0
        self._recorded_question_ids.clear()

        expected_errors: dict[str, StructuredError] = {}

        # 1. Parse and validate raw_results.jsonl
        if self.paths.raw_results_path.exists():
            with open(self.paths.raw_results_path, encoding="utf-8") as f:
                for line_no, line in enumerate(f, 1):
                    stripped = line.strip()
                    if not stripped:
                        raise ArtifactConsistencyError(
                            f"Corrupt or empty row at line {line_no} in "
                            f"'{self.paths.raw_results_path}'"
                        )
                    try:
                        data = json.loads(stripped)
                    except json.JSONDecodeError as e:
                        raise ArtifactConsistencyError(
                            f"Malformed JSON at line {line_no} in "
                            f"'{self.paths.raw_results_path}': {e}"
                        ) from e

                    try:
                        res = QueryResult.model_validate(data)
                    except Exception as e:
                        raise ArtifactConsistencyError(
                            f"Invalid QueryResult schema at line {line_no} in "
                            f"'{self.paths.raw_results_path}': {e}"
                        ) from e

                    if res.system != self.bound_system:
                        raise IdentityMismatchError(
                            f"QueryResult at line {line_no} has foreign system '{res.system}' "
                            f"!= run system '{self.bound_system}'"
                        )
                    if res.variant != self.bound_variant:
                        raise IdentityMismatchError(
                            f"QueryResult at line {line_no} has foreign variant '{res.variant}' "
                            f"!= run variant '{self.bound_variant}'"
                        )

                    if res.question_id in self._recorded_question_ids:
                        raise DuplicateResultError(
                            f"Duplicate question_id '{res.question_id}' at line {line_no} "
                            "in durable raw_results.jsonl"
                        )

                    self._recorded_question_ids.add(res.question_id)
                    self._query_count += 1
                    if res.status == QueryStatus.SUCCESS:
                        self._success_count += 1
                    else:
                        self._failed_count += 1
                        if res.error is not None:
                            if res.error.question_id and res.error.question_id != res.question_id:
                                raise IdentityMismatchError(
                                    f"Nested error question_id '{res.error.question_id}' != "
                                    f"'{res.question_id}' at line {line_no}"
                                )
                            if res.error.system and res.error.system != self.bound_system:
                                raise IdentityMismatchError(
                                    f"Nested error system '{res.error.system}' != "
                                    f"'{self.bound_system}' at line {line_no}"
                                )
                            if res.error.variant and res.error.variant != self.bound_variant:
                                raise IdentityMismatchError(
                                    f"Nested error variant '{res.error.variant}' != "
                                    f"'{self.bound_variant}' at line {line_no}"
                                )
                            expected_errors[res.question_id] = res.error
                        else:
                            raise ArtifactConsistencyError(
                                f"Failed QueryResult at line {line_no} in raw_results.jsonl "
                                "has error=None"
                            )

        # 2. Parse and validate errors.jsonl
        durable_errors: list[StructuredError] = []
        durable_error_questions: set[str] = set()

        if self.paths.errors_path.exists():
            with open(self.paths.errors_path, encoding="utf-8") as f:
                for line_no, line in enumerate(f, 1):
                    stripped = line.strip()
                    if not stripped:
                        raise ArtifactConsistencyError(
                            f"Corrupt or empty row at line {line_no} in '{self.paths.errors_path}'"
                        )
                    try:
                        data = json.loads(stripped)
                    except json.JSONDecodeError as e:
                        raise ArtifactConsistencyError(
                            f"Malformed JSON at line {line_no} in '{self.paths.errors_path}': {e}"
                        ) from e

                    try:
                        err = StructuredError.model_validate(data)
                    except Exception as e:
                        raise ArtifactConsistencyError(
                            f"Invalid StructuredError schema at line {line_no} in "
                            f"'{self.paths.errors_path}': {e}"
                        ) from e

                    if err.system and err.system != self.bound_system:
                        raise IdentityMismatchError(
                            f"Error at line {line_no} has foreign system '{err.system}' "
                            f"!= run system '{self.bound_system}'"
                        )
                    if err.variant and err.variant != self.bound_variant:
                        raise IdentityMismatchError(
                            f"Error at line {line_no} has foreign variant '{err.variant}' "
                            f"!= run variant '{self.bound_variant}'"
                        )

                    if err.question_id:
                        if err.question_id in durable_error_questions:
                            raise ArtifactConsistencyError(
                                f"Duplicate error record for question_id '{err.question_id}' "
                                f"at line {line_no} in errors.jsonl"
                            )
                        durable_error_questions.add(err.question_id)

                    durable_errors.append(err)

        # 3. Cross-file consistency checks
        if (
            not self.paths.raw_results_path.exists()
            and self.paths.errors_path.exists()
            and durable_errors
        ):
            raise ArtifactConsistencyError(
                f"Orphan errors.jsonl exists with {len(durable_errors)} errors "
                "but raw_results.jsonl is missing"
            )

        # Missing expected error records for failed results
        for qid in expected_errors:
            if qid not in durable_error_questions:
                raise ArtifactConsistencyError(
                    f"Missing expected error record in errors.jsonl for failed question '{qid}'"
                )

        # Orphan error records with question_id not corresponding to failed result
        failed_qids = set(expected_errors.keys())
        for err in durable_errors:
            if err.question_id:
                if err.question_id not in self._recorded_question_ids:
                    raise ArtifactConsistencyError(
                        f"Orphan error record in errors.jsonl: question '{err.question_id}' "
                        "does not exist in raw_results.jsonl"
                    )
                if err.question_id not in failed_qids:
                    raise ArtifactConsistencyError(
                        f"Orphan error record in errors.jsonl: question '{err.question_id}' "
                        "succeeded in raw_results.jsonl"
                    )

    def append_result(self, result: QueryResult) -> None:
        """Append a QueryResult record to raw_results.jsonl.

        Validates:
          - Writer is in ACTIVE lifecycle and retains valid lock ownership
          - System and variant match bound run identity
          - question_id has not already been recorded for this run
        """
        if self.lifecycle == WriterLifecycle.CLOSED:
            raise InvalidLifecycleError(
                f"Cannot append result: writer for run '{self.run_id}' is permanently CLOSED."
            )
        if self.lifecycle == WriterLifecycle.FINALIZED:
            raise InvalidLifecycleError(
                f"Cannot append result: writer for run '{self.run_id}' has already been finalized."
            )
        if self.lifecycle != WriterLifecycle.ACTIVE:
            raise InvalidLifecycleError(
                f"Cannot append result: writer is not initialized or active "
                f"(state: {self.lifecycle.value})."
            )

        if not self._verify_lock_held():
            raise WriterConcurrencyError(
                f"Writer lost lock ownership for run '{self.run_id}'. Mutation rejected."
            )

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
        if self.lifecycle == WriterLifecycle.CLOSED:
            raise InvalidLifecycleError(
                f"Cannot append error: writer for run '{self.run_id}' is permanently CLOSED."
            )
        if self.lifecycle == WriterLifecycle.FINALIZED:
            raise InvalidLifecycleError(
                f"Cannot append error: writer for run '{self.run_id}' has already been finalized."
            )
        if self.lifecycle != WriterLifecycle.ACTIVE:
            raise InvalidLifecycleError(
                f"Cannot append error: writer is not initialized or active "
                f"(state: {self.lifecycle.value})."
            )

        if not self._verify_lock_held():
            raise WriterConcurrencyError(
                f"Writer lost lock ownership for run '{self.run_id}'. Mutation rejected."
            )

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
        if self.lifecycle == WriterLifecycle.CLOSED:
            raise InvalidLifecycleError(
                f"Cannot finalize: writer for run '{self.run_id}' is permanently CLOSED."
            )
        if self.lifecycle == WriterLifecycle.FINALIZED:
            raise InvalidLifecycleError(
                f"Cannot finalize: writer for run '{self.run_id}' has already been finalized."
            )
        if self.lifecycle != WriterLifecycle.ACTIVE:
            raise InvalidLifecycleError(
                f"Cannot finalize: writer is not initialized or active "
                f"(state: {self.lifecycle.value})."
            )

        if not self._verify_lock_held():
            raise WriterConcurrencyError(
                f"Writer lost lock ownership for run '{self.run_id}'. Finalize rejected."
            )

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
        self.lifecycle = WriterLifecycle.FINALIZED
        self._release_lock()
        return finalized

    def close(self) -> None:
        """Permanently close writer and release any held locks."""
        self._release_lock()
        if self.lifecycle != WriterLifecycle.FINALIZED:
            self.lifecycle = WriterLifecycle.CLOSED

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
            for line_no, line in enumerate(f, 1):
                stripped = line.strip()
                if not stripped:
                    raise ArtifactConsistencyError(
                        f"Empty line at line {line_no} in '{self.paths.raw_results_path}'"
                    )
                data = json.loads(stripped)
                results.append(QueryResult.model_validate(data))
        return results

    def read_errors(self) -> list[StructuredError]:
        """Read all errors from errors.jsonl if present."""
        if not self.paths.errors_path.exists():
            return []

        errors: list[StructuredError] = []
        with open(self.paths.errors_path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                stripped = line.strip()
                if not stripped:
                    raise ArtifactConsistencyError(
                        f"Empty line at line {line_no} in '{self.paths.errors_path}'"
                    )
                data = json.loads(stripped)
                errors.append(StructuredError.model_validate(data))
        return errors
