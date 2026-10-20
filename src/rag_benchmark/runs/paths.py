"""Path utilities, safe run_id validation, and layout for benchmark run artifacts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from rag_benchmark.errors.models import InvalidRunIdError

# Conservative regex for run_id: must start with an alphanumeric char,
# followed by alphanumeric, underscore, hyphen, or dot.
_RUN_ID_REGEX = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_\-\.]*$")

_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    "COM1",
    "COM2",
    "COM3",
    "COM4",
    "COM5",
    "COM6",
    "COM7",
    "COM8",
    "COM9",
    "LPT1",
    "LPT2",
    "LPT3",
    "LPT4",
    "LPT5",
    "LPT6",
    "LPT7",
    "LPT8",
    "LPT9",
}


def validate_run_id(run_id: str) -> str:
    """Validate that run_id is a safe, single-component filesystem identifier.

    Rejects:
      - Path traversal (., .., ../outside, ../../tmp/test)
      - Absolute paths (/foo, \\foo)
      - Drive-qualified paths (C:\\temp, D:foo)
      - Directory separators (/ and \\)
      - Empty or whitespace-only values
      - Windows reserved device names (CON, NUL, AUX, etc.)

    Raises:
        InvalidRunIdError: If run_id fails validation.
    """
    if not isinstance(run_id, str):
        raise InvalidRunIdError(f"run_id must be a string, got {type(run_id).__name__}")

    stripped = run_id.strip()
    if not stripped:
        raise InvalidRunIdError("run_id cannot be empty or whitespace only")

    if run_id != stripped:
        raise InvalidRunIdError(f"run_id cannot have leading or trailing whitespace: {run_id!r}")

    if "/" in run_id or "\\" in run_id or ":" in run_id:
        raise InvalidRunIdError(f"run_id cannot contain path separators or colons: {run_id!r}")

    if ".." in run_id:
        raise InvalidRunIdError(f"run_id cannot contain traversal component '..': {run_id!r}")

    if not _RUN_ID_REGEX.match(run_id):
        raise InvalidRunIdError(
            f"run_id '{run_id}' does not match allowed alphanumeric/hyphen/dot format."
        )

    name_without_ext = run_id.split(".")[0].upper()
    if name_without_ext in _WINDOWS_RESERVED_NAMES:
        raise InvalidRunIdError(f"run_id '{run_id}' uses a reserved device name.")

    return run_id


@dataclass(frozen=True)
class RunPaths:
    """Manages filesystem layout for a benchmark execution run with traversal protection."""

    base_dir: Path
    run_id: str

    def __post_init__(self) -> None:
        validated_id = validate_run_id(self.run_id)
        # Verify resolved path remains strictly within base directory
        base_resolved = Path(self.base_dir).resolve()
        target_resolved = (base_resolved / validated_id).resolve()

        if base_resolved not in target_resolved.parents or target_resolved == base_resolved:
            raise InvalidRunIdError(
                f"Resolved run path '{target_resolved}' escapes base dir '{base_resolved}'"
            )

    @property
    def run_dir(self) -> Path:
        """Directory path for this run: <base_dir>/<run_id>."""
        return self.base_dir / self.run_id

    @property
    def manifest_path(self) -> Path:
        """Path to manifest.json."""
        return self.run_dir / "manifest.json"

    @property
    def raw_results_path(self) -> Path:
        """Path to raw_results.jsonl."""
        return self.run_dir / "raw_results.jsonl"

    @property
    def errors_path(self) -> Path:
        """Path to errors.jsonl."""
        return self.run_dir / "errors.jsonl"

    @property
    def lock_path(self) -> Path:
        """Path to exclusive writer lock file: writer.lock."""
        return self.run_dir / "writer.lock"

    def ensure_directories(self) -> None:
        """Ensure that the run directory exists."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
