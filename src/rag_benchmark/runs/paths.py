"""Path utilities and layout for benchmark run artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunPaths:
    """Manages filesystem layout for a benchmark execution run."""

    base_dir: Path
    run_id: str

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

    def ensure_directories(self) -> None:
        """Ensure that the run directory exists."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
