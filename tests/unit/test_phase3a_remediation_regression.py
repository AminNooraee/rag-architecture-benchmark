"""Comprehensive regression test suite for Phase 3A Codex findings F-01 through F-17."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from rag_benchmark.adapters.base import (
    AdapterCapabilities,
    BaseRAGAdapter,
    HealthCheckResult,
    IndexingResult,
)
from rag_benchmark.config.models import BenchmarkConfig
from rag_benchmark.errors.models import (
    ArtifactCollisionError,
    BenchmarkErrorType,
    ConfigurationError,
    DuplicateResultError,
    GroundTruthLeakageError,
    IdentityMismatchError,
    InvalidCapabilityDeclarationError,
    InvalidLifecycleError,
    InvalidRunIdError,
    StructuredError,
    WriterConcurrencyError,
)
from rag_benchmark.runs.manifest import (
    atomic_write_json,
    create_manifest,
)
from rag_benchmark.runs.paths import RunPaths, validate_run_id
from rag_benchmark.runs.writer import RunArtifactWriter, WriterMode
from rag_benchmark.schemas.document import BenchmarkDocument
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
)
from rag_benchmark.schemas.strict_json import (
    validate_strict_json_dict,
    validate_strict_json_value,
)
from rag_benchmark.security.sanitizer import (
    REDACTED_STR,
    sanitize_dict_for_manifest,
)

# -------------------------------------------------------------------------
# F-01: Safe run_id / Path Traversal
# -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "malicious_id",
    [
        ".",
        "..",
        "../outside",
        "../../tmp/test",
        "/foo",
        "\\foo",
        "a/b",
        "a\\b",
        "C:\\temp\\x",
        "D:outside",
        "run..test",
        " run-spaces ",
        "",
        "   ",
        "NUL",
        "CON",
    ],
)
def test_regression_path_traversal_rejected(malicious_id: str, tmp_path: Path) -> None:
    """F-01: Reject path traversal, directory separators, and drive letters."""
    with pytest.raises(InvalidRunIdError):
        validate_run_id(malicious_id)

    with pytest.raises(InvalidRunIdError):
        RunPaths(base_dir=tmp_path, run_id=malicious_id)


@pytest.mark.parametrize(
    "safe_id",
    [
        "run-20261020-001",
        "20261020_120000_lightrag",
        "benchmark_v1_test-01",
        "run.123",
    ],
)
def test_regression_safe_run_ids_accepted(safe_id: str, tmp_path: Path) -> None:
    """F-01: Conservative run_id formats must be accepted."""
    assert validate_run_id(safe_id) == safe_id
    paths = RunPaths(base_dir=tmp_path, run_id=safe_id)
    assert paths.run_dir == tmp_path / safe_id


# -------------------------------------------------------------------------
# F-02: Explicit Writer Lifecycle (CREATE, RESUME, OVERWRITE, State Rules)
# -------------------------------------------------------------------------


def test_regression_create_rejects_existing_artifacts(tmp_path: Path) -> None:
    """F-02: CREATE must fail if ANY artifact already exists."""
    run_id = "run-create-guard"
    manifest = create_manifest(run_id=run_id, system="traditional_rag")

    writer = RunArtifactWriter.create(base_dir=tmp_path, manifest=manifest)
    writer.close()

    # Re-creating an existing RUNNING run fails
    with pytest.raises(ArtifactCollisionError, match="artifacts exist"):
        RunArtifactWriter.create(base_dir=tmp_path, manifest=manifest)


def test_regression_resume_rejects_terminal_runs(tmp_path: Path) -> None:
    """F-02: RESUME must reject COMPLETED and FAILED runs."""
    run_id = "run-terminal-resume"
    manifest = create_manifest(run_id=run_id, system="traditional_rag")
    writer = RunArtifactWriter.create(base_dir=tmp_path, manifest=manifest)
    writer.finalize(status=RunStatus.COMPLETED)

    # Resume of completed run must fail
    with pytest.raises(InvalidLifecycleError, match="status 'completed'"):
        RunArtifactWriter.resume(base_dir=tmp_path, run_id=run_id)


def test_regression_overwrite_clears_old_jsonl(tmp_path: Path) -> None:
    """F-02: OVERWRITE must wipe old raw_results and errors, not keep stale data."""
    run_id = "run-overwrite-clean"
    manifest = create_manifest(run_id=run_id, system="traditional_rag")
    writer = RunArtifactWriter.create(base_dir=tmp_path, manifest=manifest)
    res = QueryResult.create_success(
        question_id="q-stale", system="traditional_rag", answer="قدیمی"
    )
    writer.append_result(res)
    writer.close()

    assert writer.paths.raw_results_path.exists()
    assert len(writer.read_results()) == 1

    # Overwrite run
    new_writer = RunArtifactWriter.overwrite(base_dir=tmp_path, manifest=manifest)
    assert not new_writer.paths.raw_results_path.exists()
    assert len(new_writer.read_results()) == 0
    new_writer.close()


def test_regression_append_and_finalize_state_rules(tmp_path: Path) -> None:
    """F-02: Appends and finalizes must fail when not actively RUNNING."""
    run_id = "run-state-rules"
    manifest = create_manifest(run_id=run_id, system="traditional_rag")

    uninit_writer = RunArtifactWriter(base_dir=tmp_path, run_id=run_id, mode=WriterMode.CREATE)
    res = QueryResult.create_success(question_id="q-1", system="traditional_rag", answer="پاسخ")

    # Fail before initialization
    with pytest.raises(InvalidLifecycleError, match="not initialized"):
        uninit_writer.append_result(res)

    with pytest.raises(InvalidLifecycleError, match="not initialized"):
        uninit_writer.finalize()

    # Initialize and finalize
    uninit_writer.initialize(manifest)
    uninit_writer.append_result(res)
    uninit_writer.finalize(status=RunStatus.COMPLETED)

    # Fail after finalization
    res2 = QueryResult.create_success(question_id="q-2", system="traditional_rag", answer="پاسخ")
    with pytest.raises(InvalidLifecycleError, match="already been finalized"):
        uninit_writer.append_result(res2)

    with pytest.raises(InvalidLifecycleError, match="already been finalized"):
        uninit_writer.finalize()


# -------------------------------------------------------------------------
# F-03 & F-04: Bind Identity & Duplicate Question IDs
# -------------------------------------------------------------------------


def test_regression_identity_mismatches_rejected(tmp_path: Path) -> None:
    """F-03: Results and errors with conflicting system or variant are rejected."""
    run_id = "run-identity-bind"
    manifest = create_manifest(run_id=run_id, system="traditional_rag", variant="dense_512")
    writer = RunArtifactWriter.create(base_dir=tmp_path, manifest=manifest)

    # Wrong system
    res_wrong_sys = QueryResult.create_success(
        question_id="q-1", system="lightrag", variant="dense_512", answer="پاسخ"
    )
    with pytest.raises(IdentityMismatchError, match="Result system 'lightrag' != run system"):
        writer.append_result(res_wrong_sys)

    # Wrong variant
    res_wrong_var = QueryResult.create_success(
        question_id="q-1", system="traditional_rag", variant="graph_hybrid", answer="پاسخ"
    )
    with pytest.raises(IdentityMismatchError, match="Result variant 'graph_hybrid' != run variant"):
        writer.append_result(res_wrong_var)

    writer.close()


def test_regression_duplicate_question_ids_rejected(tmp_path: Path) -> None:
    """F-04: (run_id, question_id) uniqueness rule."""
    run_id = "run-dup-questions"
    manifest = create_manifest(run_id=run_id, system="traditional_rag")
    writer = RunArtifactWriter.create(base_dir=tmp_path, manifest=manifest)

    res1 = QueryResult.create_success(
        question_id="q-dup-1", system="traditional_rag", answer="پاسخ اول"
    )
    writer.append_result(res1)

    # Second result for same question_id in same process
    res2 = QueryResult.create_success(
        question_id="q-dup-1", system="traditional_rag", answer="پاسخ تکراری"
    )
    with pytest.raises(DuplicateResultError, match="Duplicate question_id 'q-dup-1'"):
        writer.append_result(res2)

    writer.close()

    # Reopen via RESUME and verify uniqueness is still enforced
    resumed = RunArtifactWriter.resume(base_dir=tmp_path, run_id=run_id)
    with pytest.raises(DuplicateResultError, match="Duplicate question_id 'q-dup-1'"):
        resumed.append_result(res2)

    resumed.close()


def test_regression_resume_reconstructs_counters(tmp_path: Path) -> None:
    """F-04 / F-02: Reconstruct exact query totals on resume."""
    run_id = "run-resume-counters"
    manifest = create_manifest(run_id=run_id, system="traditional_rag")
    writer = RunArtifactWriter.create(base_dir=tmp_path, manifest=manifest)

    res_ok = QueryResult.create_success(question_id="q-1", system="traditional_rag", answer="موفق")
    res_fail = QueryResult.create_failure(
        question_id="q-2",
        system="traditional_rag",
        error_message="خطا",
        error_type=BenchmarkErrorType.GENERATION_ERROR,
    )
    writer.append_result(res_ok)
    writer.append_result(res_fail)
    writer.close()

    resumed = RunArtifactWriter.resume(base_dir=tmp_path, run_id=run_id)
    finalized = resumed.finalize(status=RunStatus.COMPLETED)

    assert finalized.total_queries == 2
    assert finalized.successful_queries == 1
    assert finalized.failed_queries == 1


# -------------------------------------------------------------------------
# F-05: Ground-Truth Isolation
# -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "forbidden_key",
    [
        "gold_answer",
        "reference_answer",
        "acceptable_answers",
        "ground_truth",
        "gold_evidence",
        "supporting_facts",
        "supporting_document_ids",
        "expected_entities",
        "answer_key",
        "reference_documents",
    ],
)
def test_regression_ground_truth_leakage_prevented(forbidden_key: str) -> None:
    """F-05: Ground-truth aliases in query_metadata raise GroundTruthLeakageError."""
    q = BenchmarkQuestion(
        question_id="q-test-leak",
        question="سؤال؟",
        question_type=QuestionType.DIRECT_FACTUAL,
        scope=QuestionScope.LOCAL,
        gold_answer="پاسخ طلایی",
        query_metadata={forbidden_key: "leaked_data"},
    )
    with pytest.raises(GroundTruthLeakageError, match="Forbidden ground-truth key"):
        q.to_query_input()


def test_regression_evaluation_metadata_not_in_adapter_input() -> None:
    """F-05: Evaluation metadata is never passed to AdapterQueryInput."""
    q = BenchmarkQuestion(
        question_id="q-eval-meta",
        question="سؤال؟",
        question_type=QuestionType.DIRECT_FACTUAL,
        scope=QuestionScope.LOCAL,
        gold_answer="پاسخ",
        query_metadata={"safe_param": 10},
        metadata={"annotator_notes": "محرمانه", "gold_metric": 0.95},
    )
    query_in = q.to_query_input()
    assert query_in.metadata == {"safe_param": 10}
    assert "annotator_notes" not in query_in.metadata
    assert "gold_metric" not in query_in.metadata


# -------------------------------------------------------------------------
# F-06: Centralized Secret Redaction
# -------------------------------------------------------------------------


def test_regression_secret_canaries_and_url_redaction() -> None:
    """F-06: Secrets and credentials in structured artifacts and URLs are redacted."""
    canary_pw = "SuperSecretCanary123"
    canary_token = "sk-CanarySecretToken123456789"

    raw = {
        "api_key": canary_token,
        "password": canary_pw,
        "llm_endpoint": f"https://admin:{canary_pw}@api.example.com/v1?token={canary_token}",
        "safe_info": "اطلاعات عمومی",
    }

    sanitized = sanitize_dict_for_manifest(raw)
    dumped = json.dumps(sanitized)

    assert canary_pw not in dumped
    assert canary_token not in dumped
    assert REDACTED_STR in dumped


def test_regression_sanitizer_avoids_false_positives() -> None:
    """F-06: Legitimate provenance keys (tokenizer, author) must NOT be redacted."""
    raw = {
        "author": "دکتر حسابی",
        "authors": ["پژوهشگر ۱", "پژوهشگر ۲"],
        "tokenizer": "cl100k_base",
        "tokenizer_name": "persian_bpe",
        "authorization_mode": "rbac",
        "token_count": 150,
        "input_tokens": 100,
        "output_tokens": 50,
    }
    sanitized = sanitize_dict_for_manifest(raw)
    assert sanitized["author"] == "دکتر حسابی"
    assert sanitized["tokenizer"] == "cl100k_base"
    assert sanitized["authorization_mode"] == "rbac"
    assert sanitized["input_tokens"] == 100


# -------------------------------------------------------------------------
# F-07 & F-08: Strict JSON & Mutable Nested Metadata
# -------------------------------------------------------------------------


def test_regression_strict_json_rejects_nan_and_tuples() -> None:
    """F-07: Strict JSON rejects NaN, Infinity, tuples, and non-string keys."""
    with pytest.raises(ValueError, match="non-finite float"):
        validate_strict_json_dict({"bad": float("nan")})

    with pytest.raises(ValueError, match="rejects tuples"):
        validate_strict_json_dict({"bad": (1, 2)})

    with pytest.raises(ValueError, match="requires string keys"):
        bad_dict: dict[Any, Any] = {123: "val"}
        validate_strict_json_value(bad_dict)


def test_regression_post_construction_metadata_mutation_isolation() -> None:
    """F-08: Mutating the caller's dictionary after construction cannot change model metadata."""
    original = {"param": "value", "list": [1, 2, 3]}
    doc = BenchmarkDocument(
        document_id="doc-mut",
        title="عنوان",
        text="متن",
        metadata=original,
    )

    # Mutate original caller dict
    original["param"] = "MUTATED"
    original["list"].append(999)  # type: ignore[attr-defined]

    # Model metadata must remain unchanged
    assert doc.metadata["param"] == "value"
    assert doc.metadata["list"] == [1, 2, 3]


# -------------------------------------------------------------------------
# F-09 & F-10: Manifest Lifecycle Validation & Timezone Handling
# -------------------------------------------------------------------------


def test_regression_manifest_counter_validation() -> None:
    """F-09: Terminal manifest must satisfy total == successful + failed."""
    manifest = create_manifest(run_id="run-counter-val", system="sys")

    # Mismatch total != success + failed
    with pytest.raises(ValidationError, match="Terminal count mismatch"):
        manifest.finalize(
            status=RunStatus.COMPLETED,
            total_queries=10,
            successful_queries=8,
            failed_queries=0,
        )


def test_regression_naive_timestamps_rejected() -> None:
    """F-10: Naive datetimes are strictly rejected."""
    naive_now = datetime.now()  # no tzinfo

    with pytest.raises(ValidationError, match="Naive datetimes are rejected"):
        StructuredError(
            system="sys",
            message="خطا",
            timestamp=naive_now,
        )


def test_regression_non_utc_timestamps_normalized() -> None:
    """F-10: Non-UTC aware datetimes are normalized to UTC."""
    tehran_tz = timezone(timedelta(hours=3, minutes=30))
    dt_tehran = datetime(2026, 10, 20, 12, 0, 0, tzinfo=tehran_tz)

    err = StructuredError(system="sys", message="خطا", timestamp=dt_tehran)
    assert err.timestamp.tzinfo == UTC
    assert err.timestamp.hour == 8
    assert err.timestamp.minute == 30


# -------------------------------------------------------------------------
# F-11: Atomic Manifest Writes
# -------------------------------------------------------------------------


def test_regression_atomic_manifest_write_failure_preserves_original(tmp_path: Path) -> None:
    """F-11: Interrupted or failing serialization leaves previously valid manifest intact."""
    manifest_file = tmp_path / "manifest.json"
    initial_data = {"run_id": "atomic-run", "valid": True}
    atomic_write_json(manifest_file, initial_data)
    assert manifest_file.exists()

    # Attempt an invalid write with non-finite float
    with pytest.raises(ValueError):
        atomic_write_json(manifest_file, {"run_id": "atomic-run", "score": float("nan")})

    # Original manifest must remain completely intact
    with open(manifest_file, encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["valid"] is True


# -------------------------------------------------------------------------
# F-12: Single-Writer Ownership
# -------------------------------------------------------------------------


def test_regression_single_writer_concurrency_rejected(tmp_path: Path) -> None:
    """F-12: Active writer locks run directory; second concurrent writer is rejected."""
    run_id = "run-concurrency-test"
    manifest = create_manifest(run_id=run_id, system="traditional_rag")

    writer1 = RunArtifactWriter.create(base_dir=tmp_path, manifest=manifest)

    # Second concurrent writer on the same run must be rejected
    with pytest.raises(WriterConcurrencyError, match="owned by active writer"):
        RunArtifactWriter(base_dir=tmp_path, run_id=run_id, mode=WriterMode.CREATE).initialize(
            manifest
        )

    writer1.close()


# -------------------------------------------------------------------------
# F-13 & F-14: QueryResult State Matrix & Retained Evidence
# -------------------------------------------------------------------------


def test_regression_query_result_state_matrix_validation() -> None:
    """F-13: Validate full success vs. failure state matrix."""
    # SUCCESS cannot have error
    err = StructuredError(question_id="q-1", system="sys", message="err")
    with pytest.raises(ValidationError, match="Successful QueryResult must not contain an error"):
        QueryResult(
            question_id="q-1",
            system="sys",
            answer="پاسخ",
            status=QueryStatus.SUCCESS,
            error=err,
        )

    # SUCCESS cannot have answer=None
    with pytest.raises(ValidationError, match="Successful QueryResult must have a non-None answer"):
        QueryResult(
            question_id="q-1",
            system="sys",
            answer=None,
            status=QueryStatus.SUCCESS,
        )

    # FAILED must have error
    with pytest.raises(ValidationError, match="must include a StructuredError"):
        QueryResult(
            question_id="q-1",
            system="sys",
            answer=None,
            status=QueryStatus.FAILED,
            error=None,
        )


def test_regression_generation_failure_retains_retrieved_evidence() -> None:
    """F-14: When retrieval succeeds and generation fails, evidence is retained."""
    ev = RetrievedEvidence(
        document_id="doc-1",
        text="شواهد بازیابی شده",
        evidence_type=EvidenceType.CHUNK,
    )
    result = QueryResult.create_failure(
        question_id="q-fail-gen",
        system="traditional_rag",
        error_message="خطای تولید در مدل زبانی",
        error_type=BenchmarkErrorType.GENERATION_ERROR,
        status=QueryStatus.FAILED,
        retrieved_evidence=[ev],
        timing=QueryTiming(retrieval_ms=15.0, generation_ms=500.0, total_ms=515.0),
    )

    assert result.status == QueryStatus.FAILED
    assert len(result.retrieved_evidence) == 1
    assert result.retrieved_evidence[0].document_id == "doc-1"
    assert result.timing.retrieval_ms == 15.0


# -------------------------------------------------------------------------
# F-15: Manifest Configuration Identity
# -------------------------------------------------------------------------


def test_regression_create_manifest_config_reconciliation() -> None:
    """F-15: Reconcile configuration values and reject conflicting arguments."""
    cfg = BenchmarkConfig(system="traditional_rag", variant="dense_1024")

    # Reconciles correctly from config
    m = create_manifest(run_id="run-cfg-ok", config=cfg)
    assert m.system == "traditional_rag"
    assert m.variant == "dense_1024"

    # Rejects conflicting system
    with pytest.raises(ConfigurationError, match="Conflicting system specification"):
        create_manifest(run_id="run-cfg-bad", system="lightrag", config=cfg)

    # Rejects conflicting variant
    with pytest.raises(ConfigurationError, match="Conflicting variant specification"):
        create_manifest(run_id="run-cfg-bad", variant="sparse_only", config=cfg)


# -------------------------------------------------------------------------
# F-16: Question Semantic Validation
# -------------------------------------------------------------------------


def test_regression_question_semantics_matrix() -> None:
    """F-16: Enforce question semantic consistency."""
    # UNANSWERABLE must have answerable=False
    with pytest.raises(ValidationError, match="must have answerable=False"):
        BenchmarkQuestion(
            question_id="q-unans-bad",
            question="سؤال؟",
            question_type=QuestionType.UNANSWERABLE,
            scope=QuestionScope.LOCAL,
            answerable=True,
        )

    # answerable=False must have empty gold_answer
    with pytest.raises(ValidationError, match="must have an empty gold_answer"):
        BenchmarkQuestion(
            question_id="q-unans-ans",
            question="سؤال؟",
            question_type=QuestionType.UNANSWERABLE,
            scope=QuestionScope.LOCAL,
            answerable=False,
            gold_answer="پاسخ غیرمجاز",
        )

    # MULTI_HOP requires reasoning_hops >= 2
    with pytest.raises(ValidationError, match="requires reasoning_hops >= 2"):
        BenchmarkQuestion(
            question_id="q-hop-bad",
            question="سؤال چندمرحله‌ای؟",
            question_type=QuestionType.MULTI_HOP,
            scope=QuestionScope.MULTI_DOCUMENT,
            reasoning_hops=1,
            gold_answer="پاسخ",
        )


# -------------------------------------------------------------------------
# F-17: Adapter Capability Trustworthiness
# -------------------------------------------------------------------------


def test_regression_adapter_capability_contract_validation() -> None:
    """F-17: Adapter declaring capability without override fails validate_capabilities()."""

    class BadAdapter(BaseRAGAdapter):
        @property
        def system_name(self) -> str:
            return "bad_rag"

        @property
        def variant_name(self) -> str:
            return "v1"

        @property
        def capabilities(self) -> AdapterCapabilities:
            return AdapterCapabilities(supports_update=True)

        def prepare(self, config: BenchmarkConfig) -> None:
            pass

        def index(self, documents: Sequence[BenchmarkDocument]) -> IndexingResult:
            return IndexingResult()

        def query(self, query_input: AdapterQueryInput) -> QueryResult:
            return QueryResult.create_success(
                question_id=query_input.question_id,
                system=self.system_name,
                answer="پاسخ",
            )

        def health(self) -> HealthCheckResult:
            return HealthCheckResult(healthy=True, system=self.system_name)

        def cleanup(self) -> None:
            pass

    bad = BadAdapter()
    with pytest.raises(InvalidCapabilityDeclarationError, match="does not override.*update"):
        bad.validate_capabilities()
