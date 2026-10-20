"""End-to-end integration test for benchmark harness foundation."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from rag_benchmark.adapters.base import (
    AdapterCapabilities,
    BaseRAGAdapter,
    HealthCheckResult,
    IndexingResult,
)
from rag_benchmark.config.loader import load_config
from rag_benchmark.config.models import BenchmarkConfig
from rag_benchmark.errors.models import BenchmarkErrorType
from rag_benchmark.runs.manifest import create_manifest, load_manifest
from rag_benchmark.runs.writer import RunArtifactWriter
from rag_benchmark.schemas.document import BenchmarkDocument, DocumentSection
from rag_benchmark.schemas.evidence import EvidenceType, RetrievedEvidence
from rag_benchmark.schemas.manifest import RunStatus
from rag_benchmark.schemas.question import (
    AdapterQueryInput,
    BenchmarkQuestion,
    QuestionScope,
    QuestionType,
)
from rag_benchmark.schemas.result import (
    QueryResult,
    QueryStatus,
    QueryTiming,
    QueryUsage,
)


class BenchmarkTestRAGAdapter(BaseRAGAdapter):
    """Deterministic in-memory mock RAG adapter for integration testing."""

    def __init__(self) -> None:
        self.config: BenchmarkConfig | None = None
        self.indexed_corpus: dict[str, BenchmarkDocument] = {}

    @property
    def system_name(self) -> str:
        return "test_inmemory_rag"

    @property
    def variant_name(self) -> str:
        return "integration_test"

    @property
    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            supports_incremental_insert=True,
            supports_persistence=False,
        )

    def prepare(self, config: BenchmarkConfig) -> None:
        self.config = config

    def index(self, documents: Sequence[BenchmarkDocument]) -> IndexingResult:
        for doc in documents:
            self.indexed_corpus[doc.document_id] = doc
        return IndexingResult(
            indexed_documents=len(documents),
            indexed_chunks=len(documents) * 2,
            elapsed_ms=12.5,
        )

    def health(self) -> HealthCheckResult:
        return HealthCheckResult(
            healthy=True,
            system=self.system_name,
            message="Integration mock is healthy",
        )

    def query(self, query_input: AdapterQueryInput) -> QueryResult:
        # Simulate query failure for special question ID
        if query_input.question_id == "q-fail":
            return QueryResult.create_failure(
                question_id=query_input.question_id,
                system=self.system_name,
                variant=self.variant_name,
                error_message="شبیه‌سازی خطای پردازش مدل در تست یکپارچگی",
                error_type=BenchmarkErrorType.GENERATION_ERROR,
                status=QueryStatus.FAILED,
                timing=QueryTiming(retrieval_ms=5.0, generation_ms=15.0, total_ms=20.0),
            )

        # Retrieve mock evidence from indexed documents
        doc = next(iter(self.indexed_corpus.values())) if self.indexed_corpus else None
        evidence = (
            [
                RetrievedEvidence(
                    document_id=doc.document_id,
                    chunk_id=f"{doc.document_id}-c0",
                    text=doc.text[:100],
                    score=0.92,
                    rank=1,
                    evidence_type=EvidenceType.CHUNK,
                )
            ]
            if doc
            else []
        )

        return QueryResult.create_success(
            question_id=query_input.question_id,
            system=self.system_name,
            variant=self.variant_name,
            answer=f"پاسخ استخراج‌شده برای سؤال «{query_input.question}» با نیم‌فاصله.",
            retrieved_evidence=evidence,
            usage=QueryUsage(llm_calls=1, embedding_calls=1, input_tokens=120, output_tokens=30),
            timing=QueryTiming(retrieval_ms=10.0, generation_ms=150.0, total_ms=160.0),
        )

    def cleanup(self) -> None:
        self.indexed_corpus.clear()


def test_full_benchmark_harness_integration_flow(tmp_path: Path) -> None:
    """Execute complete benchmark flow: configure -> index -> query -> record -> finalize."""
    run_id = "run-integration-test-01"
    runs_dir = tmp_path / "runs"

    # 1. Config
    config_dict = {
        "system": "test_inmemory_rag",
        "variant": "integration_test",
        "benchmark": {
            "language": "fa",
            "dataset_version": "persian_test_v1",
            "question_set_version": "qset_test_v1",
        },
    }
    config = load_config(config_dict)

    # 2. Documents
    docs = [
        BenchmarkDocument(
            document_id="doc-fa-1",
            title="تاریخچه هوش مصنوعی در ایران",
            text="پژوهش‌های هوش مصنوعی و پردازش زبان فارسی از اوایل دهه هفتاد شمسی آغاز شد.",
            sections=[
                DocumentSection(
                    section_id="s1",
                    heading="مقدمه",
                    text="متن مقدمه با نیم‌فاصله: پایگاه‌های داده و بازیابی اطلاعات.",
                    order=0,
                )
            ],
            metadata={"domain": "AI", "verified": True},
        ),
        BenchmarkDocument(
            document_id="doc-fa-2",
            title="روش‌های ارزیابی سیستم‌های بازیابی متن",
            text="معیارهای ارزیابی شامل دقت، بازخوانی و رتبه‌بندی نتایج می‌باشند.",
        ),
    ]

    # 3. Questions (including ground truth)
    questions = [
        BenchmarkQuestion(
            question_id="q-01",
            question="پژوهش‌های هوش مصنوعی در ایران از چه زمانی آغاز شد؟",
            question_type=QuestionType.DIRECT_FACTUAL,
            scope=QuestionScope.LOCAL,
            reasoning_hops=1,
            gold_answer="اوایل دهه هفتاد شمسی",
            supporting_document_ids=["doc-fa-1"],
        ),
        BenchmarkQuestion(
            question_id="q-02",
            question="معیارهای کلیدی ارزیابی سیستم‌های بازیابی متن کدامند؟",
            question_type=QuestionType.DETAILED_FACTUAL,
            scope=QuestionScope.LOCAL,
            reasoning_hops=1,
            gold_answer="دقت، بازخوانی و رتبه‌بندی نتایج",
            supporting_document_ids=["doc-fa-2"],
        ),
        BenchmarkQuestion(
            question_id="q-fail",
            question="این سؤال برای شبیه‌سازی خطا طراحی شده است.",
            question_type=QuestionType.UNANSWERABLE,
            scope=QuestionScope.LOCAL,
            reasoning_hops=0,
            gold_answer="",
            answerable=False,
        ),
    ]

    # 4. Initialize Adapter
    adapter = BenchmarkTestRAGAdapter()
    assert adapter.health().healthy is True

    adapter.prepare(config)
    idx_res = adapter.index(docs)
    assert idx_res.indexed_documents == 2

    # 5. Initialize Artifact Writer
    writer = RunArtifactWriter(base_dir=runs_dir, run_id=run_id)
    manifest = create_manifest(
        run_id=run_id,
        system=adapter.system_name,
        variant=adapter.variant_name,
        config=config,
    )
    writer.initialize(manifest)

    # 6. Query Adapter with AdapterQueryInput ONLY (Fairness Isolation)
    for q in questions:
        query_input = q.to_query_input()
        result = adapter.query(query_input)
        writer.append_result(result)

    # 7. Finalize Run
    finalized = writer.finalize(status=RunStatus.COMPLETED)
    assert finalized.status == RunStatus.COMPLETED
    assert finalized.total_queries == 3
    assert finalized.successful_queries == 2
    assert finalized.failed_queries == 1

    # 8. Adapter Cleanup
    adapter.cleanup()
    assert len(adapter.indexed_corpus) == 0

    # 9. Verify Disk Artifacts
    paths = writer.paths
    assert paths.manifest_path.exists()
    assert paths.raw_results_path.exists()
    assert paths.errors_path.exists()

    # Verify manifest on disk
    loaded_manifest = load_manifest(paths.manifest_path)
    assert loaded_manifest.run_id == run_id
    assert loaded_manifest.status == RunStatus.COMPLETED
    assert loaded_manifest.total_queries == 3
    assert loaded_manifest.successful_queries == 2
    assert loaded_manifest.failed_queries == 1

    # Verify results from disk
    stored_results = writer.read_results()
    assert len(stored_results) == 3
    assert stored_results[0].status == QueryStatus.SUCCESS
    assert "پاسخ استخراج‌شده" in (stored_results[0].answer or "")
    assert stored_results[2].status == QueryStatus.FAILED
    assert stored_results[2].error is not None

    # Verify errors from disk
    stored_errors = writer.read_errors()
    assert len(stored_errors) == 1
    assert stored_errors[0].question_id == "q-fail"
    assert "شبیه‌سازی خطای پردازش" in stored_errors[0].message
