"""Benchmark dataset builder.

Collects documents and questions from multiple :class:`DatasetLoader`
instances, validates referential integrity, writes deterministic artifacts
and returns a manifest.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path

from rag_benchmark.datasets.base import DatasetLoader
from rag_benchmark.datasets.io import write_json, write_jsonl
from rag_benchmark.datasets.manifest import BenchmarkDatasetManifest, SourceInfo
from rag_benchmark.datasets.statistics import DatasetStatistics
from rag_benchmark.schemas.document import BenchmarkDocument
from rag_benchmark.schemas.question import BenchmarkQuestion


class BenchmarkDatasetBuilder:
    def __init__(self) -> None:
        self._loaders: list[DatasetLoader] = []

    def add_source(self, loader: DatasetLoader) -> None:
        self._loaders.append(loader)

    def _collect(self) -> tuple[dict[str, BenchmarkDocument], dict[str, BenchmarkQuestion]]:
        docs: dict[str, BenchmarkDocument] = {}
        qs: dict[str, BenchmarkQuestion] = {}
        for loader in self._loaders:
            for doc in loader.load_documents():
                if doc.document_id in docs:
                    raise ValueError(f"Duplicate document ID detected: {doc.document_id}")
                docs[doc.document_id] = doc
            for q in loader.load_questions():
                if q.question_id in qs:
                    raise ValueError(f"Duplicate question ID detected: {q.question_id}")
                qs[q.question_id] = q
        return docs, qs

    def _validate(
        self,
        docs: dict[str, BenchmarkDocument],
        qs: dict[str, BenchmarkQuestion],
    ) -> None:
        missing = []
        for q in qs.values():
            for doc_id in q.supporting_document_ids:
                if doc_id not in docs:
                    missing.append((q.question_id, doc_id))
        if missing:
            msgs = ", ".join([f"{qid}->{did}" for qid, did in missing])
            raise ValueError(f"Dangling supporting document references: {msgs}")

    def build(self, output_dir: str | Path) -> BenchmarkDatasetManifest:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        docs, qs = self._collect()
        self._validate(docs, qs)
        # Write deterministic artifacts
        write_jsonl(out_path / "documents.jsonl", docs.values())
        write_jsonl(out_path / "questions.jsonl", qs.values())

        # Determine deterministic splits (train/dev/test/benchmark) and store in metadata
        split_labels = ["train", "dev", "test", "benchmark"]
        split_counts: dict[str, int] = {label: 0 for label in split_labels}
        new_qs: dict[str, BenchmarkQuestion] = {}
        for q in qs.values():
            h = int(hashlib.sha256(q.question_id.encode()).hexdigest(), 16)
            split = split_labels[h % len(split_labels)]
            meta = dict(q.metadata)
            meta["split"] = split
            new_q = q.model_copy(update={"metadata": meta})
            new_qs[q.question_id] = new_q
            split_counts[split] += 1
        # Replace qs with enriched questions
        qs = new_qs

        # Compute distribution counts
        question_type_counts: dict[str, int] = {}
        scope_counts: dict[str, int] = {}
        answerable_counts: dict[str, int] = {"answerable": 0, "unanswerable": 0}
        for q in qs.values():
            qtype = q.question_type.name
            question_type_counts[qtype] = question_type_counts.get(qtype, 0) + 1
            scope = q.scope.name
            scope_counts[scope] = scope_counts.get(scope, 0) + 1
            if q.answerable:
                answerable_counts["answerable"] += 1
            else:
                answerable_counts["unanswerable"] += 1

        # Statistics and manifest preparation
        stats = DatasetStatistics.from_items(docs.values(), qs.values())

        # Build source information including file checksums
        sources: list[SourceInfo] = []
        for loader in self._loaders:
            doc_cnt = sum(1 for d in docs.values() if d.source == loader.source_name)
            ques_cnt = sum(1 for q in qs.values() if q.dataset_source == loader.source_name)
            # Compute source checksum if the loader exposes a path attribute
            checksum = ""
            path = getattr(loader, "_path", None)
            if isinstance(path, Path) and path.is_file():
                hasher = hashlib.sha256()
                with path.open("rb") as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        hasher.update(chunk)
                checksum = hasher.hexdigest()
            sources.append(
                SourceInfo(
                    source_name=loader.source_name,
                    source_version=loader.source_version,
                    source_checksum=checksum,
                    document_count=doc_cnt,
                    question_count=ques_cnt,
                )
            )

        # Compute content checksum over canonical ordered JSONL records
        def _ordered_jsonl(path: Path) -> Iterable[bytes]:
            lines = path.read_text(encoding="utf-8").splitlines()
            for line in sorted(lines):
                yield (line + "\n").encode()

        doc_path = out_path / "documents.jsonl"
        ques_path = out_path / "questions.jsonl"
        content_hasher = hashlib.sha256()
        for b in _ordered_jsonl(doc_path):
            content_hasher.update(b)
        for b in _ordered_jsonl(ques_path):
            content_hasher.update(b)
        content_checksum = content_hasher.hexdigest()

        manifest = BenchmarkDatasetManifest(
            dataset_name=out_path.name,
            dataset_version="v1",
            created_at=stats.created_at,
            builder_version="phase3b",
            sources=sources,
            document_count=len(docs),
            question_count=len(qs),
            content_checksum=content_checksum,
            question_type_counts=question_type_counts,
            scope_counts=scope_counts,
            answerable_counts=answerable_counts,
            split_counts=split_counts,
            statistics=stats,
        )
        write_json(out_path / "manifest.json", manifest.model_dump(mode="json"))
        write_json(out_path / "statistics.json", stats.model_dump(mode="json"))
        return manifest
