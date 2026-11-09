"""Wikipedia long‑document loader.

Parses a JSONL file where each line is a single article with optional sections.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from rag_benchmark.datasets.base import DatasetLoader
from rag_benchmark.datasets.ids import deterministic_id
from rag_benchmark.schemas.document import BenchmarkDocument, DocumentSection
from rag_benchmark.schemas.question import BenchmarkQuestion


class WikipediaLoader(DatasetLoader):
    def __init__(self, source_path: str | Path, source_version: str = "") -> None:
        self._path = Path(source_path)
        if not self._path.is_file():
            raise FileNotFoundError(f"Wikipedia source file not found: {self._path}")
        self._source_version = source_version

    @property
    def source_name(self) -> str:
        return "wikipedia"

    @property
    def source_version(self) -> str:
        return self._source_version

    def _iter_lines(self) -> Iterable[dict[str, Any]]:
        with self._path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

    def load_documents(self) -> Iterable[BenchmarkDocument]:
        for obj in self._iter_lines():
            page_id = obj.get("page_id", "")
            title = obj.get("title", "")
            doc_id = deterministic_id("wiki-doc", page_id)
            sections: list[DocumentSection] = []
            for idx, sec in enumerate(obj.get("sections", [])):
                heading = sec.get("heading", "")
                text = sec.get("text", "")
                sec_id = deterministic_id("wiki-sec", doc_id, str(idx))
                sections.append(
                    DocumentSection(
                        section_id=sec_id,
                        heading=heading,
                        text=text,
                        order=idx,
                    )
                )
            if not sections:
                sec_id = deterministic_id("wiki-sec", doc_id, "0")
                sections.append(
                    DocumentSection(
                        section_id=sec_id,
                        heading="",
                        text=obj.get("text", ""),
                        order=0,
                    )
                )
            yield BenchmarkDocument(
                document_id=doc_id,
                title=title or "(no title)",
                text=obj.get("text", ""),
                language="fa",
                source=self.source_name,
                url=obj.get("url"),
                sections=sections,
                metadata={"native_page_id": page_id},
            )

    def load_questions(self) -> Iterable[BenchmarkQuestion]:
        return []
