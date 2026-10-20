"""Comprehensive regression test suite for Phase 3A Codex findings F-01 through F-17."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import platform
import time
import uuid
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
    ArtifactConsistencyError,
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
    load_manifest,
    save_manifest,
)
from rag_benchmark.runs.paths import RunPaths, validate_run_id
from rag_benchmark.runs.writer import RunArtifactWriter, WriterLifecycle, WriterMode
from rag_benchmark.schemas.document import BenchmarkDocument, DocumentSection
from rag_benchmark.schemas.evidence import EvidenceType, RetrievedEvidence
from rag_benchmark.schemas.manifest import RunManifest, RunStatus
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
    sanitize_value,
)


def _contender_worker(base_dir_str: str, run_id: str, result_queue: Any) -> None:
    """Multiprocessing worker contending for a stale lock."""
    try:
        writer = RunArtifactWriter(base_dir=base_dir_str, run_id=run_id, mode=WriterMode.RESUME)
        writer.initialize()
        result_queue.put(("success", os.getpid(), writer._owner_token))
        time.sleep(0.3)
        writer.close()
    except WriterConcurrencyError as e:
        result_queue.put(("concurrency_error", os.getpid(), str(e)))
    except Exception as e:
        result_queue.put(("error", os.getpid(), f"{type(e).__name__}: {e}"))


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


# =========================================================================
# FINAL VERIFICATION PASS: V-01 THROUGH V-08
# =========================================================================

# -------------------------------------------------------------------------
# V-01: Writer Must Become Permanently CLOSED
# -------------------------------------------------------------------------


def test_v01_writer_explicit_close_transitions_to_closed(tmp_path: Path) -> None:
    """V-01: Explicit close transitions writer to CLOSED and blocks further mutations."""
    run_id = "v01-explicit-close"
    manifest = create_manifest(run_id=run_id, system="traditional_rag")
    writer = RunArtifactWriter.create(base_dir=tmp_path, manifest=manifest)
    assert str(writer.lifecycle) == str(WriterLifecycle.ACTIVE)

    res = QueryResult.create_success(question_id="q-1", system="traditional_rag", answer="پاسخ ۱")
    writer.append_result(res)

    writer.close()
    assert str(writer.lifecycle) == str(WriterLifecycle.CLOSED)
    assert not writer.is_active

    res2 = QueryResult.create_success(question_id="q-2", system="traditional_rag", answer="پاسخ ۲")
    with pytest.raises(InvalidLifecycleError, match="permanently CLOSED"):
        writer.append_result(res2)

    err = StructuredError(system="traditional_rag", message="خطای آزمایشی")
    with pytest.raises(InvalidLifecycleError, match="permanently CLOSED"):
        writer.append_error(err)

    with pytest.raises(InvalidLifecycleError, match="permanently CLOSED"):
        writer.finalize()

    with pytest.raises(InvalidLifecycleError, match="permanently CLOSED"):
        writer.initialize(manifest)


def test_v01_writer_context_manager_exit_permanently_closes(tmp_path: Path) -> None:
    """V-01: Context manager exit closes the writer permanently."""
    run_id = "v01-ctx-exit"
    manifest = create_manifest(run_id=run_id, system="traditional_rag")

    with RunArtifactWriter.create(base_dir=tmp_path, manifest=manifest) as writer:
        res = QueryResult.create_success(
            question_id="q-1", system="traditional_rag", answer="پاسخ ۱"
        )
        writer.append_result(res)
        assert str(writer.lifecycle) == str(WriterLifecycle.ACTIVE)

    assert str(writer.lifecycle) == str(WriterLifecycle.CLOSED)
    res2 = QueryResult.create_success(question_id="q-2", system="traditional_rag", answer="پاسخ ۲")
    with pytest.raises(InvalidLifecycleError, match="permanently CLOSED"):
        writer.append_result(res2)


def test_v01_writer_a_closes_writer_b_resumes_writer_a_mutation_blocked(tmp_path: Path) -> None:
    """V-01: Writer A closes, Writer B resumes; Writer A attempts mutation and must fail."""
    run_id = "v01-a-closes-b-resumes"
    manifest = create_manifest(run_id=run_id, system="traditional_rag")

    writer_a = RunArtifactWriter.create(base_dir=tmp_path, manifest=manifest)
    res_1 = QueryResult.create_success(question_id="q-1", system="traditional_rag", answer="پاسخ ۱")
    writer_a.append_result(res_1)

    # Writer A closes and releases lock
    writer_a.close()
    assert writer_a.lifecycle == WriterLifecycle.CLOSED

    # Writer B resumes the run and acquires lock
    writer_b = RunArtifactWriter.resume(base_dir=tmp_path, run_id=run_id)
    assert writer_b.lifecycle == WriterLifecycle.ACTIVE

    res_2 = QueryResult.create_success(question_id="q-2", system="traditional_rag", answer="پاسخ ۲")
    writer_b.append_result(res_2)

    # Writer A attempts mutation: MUST fail with InvalidLifecycleError
    res_a_late = QueryResult.create_success(
        question_id="q-3", system="traditional_rag", answer="پاسخ دیررس"
    )
    with pytest.raises(InvalidLifecycleError, match="permanently CLOSED"):
        writer_a.append_result(res_a_late)

    # Writer B finalizes the run cleanly
    finalized = writer_b.finalize(status=RunStatus.COMPLETED)
    assert finalized.total_queries == 2
    assert finalized.successful_queries == 2


# -------------------------------------------------------------------------
# V-02: RESUME Must Validate All Durable Artifacts
# -------------------------------------------------------------------------


def test_v02_resume_truncated_raw_results_aborts(tmp_path: Path) -> None:
    """V-02: Truncated JSONL record in raw_results.jsonl must abort RESUME loudly."""
    run_id = "v02-truncated-raw"
    manifest = create_manifest(run_id=run_id, system="traditional_rag")
    writer = RunArtifactWriter.create(base_dir=tmp_path, manifest=manifest)
    res = QueryResult.create_success(question_id="q-1", system="traditional_rag", answer="پاسخ")
    writer.append_result(res)
    writer.close()

    # Append half a JSON line (simulating crash during write)
    paths = RunPaths(tmp_path, run_id)
    with open(paths.raw_results_path, "a", encoding="utf-8") as f:
        f.write('{"question_id": "q-2", "system": "traditional_rag"\n')

    with pytest.raises(ArtifactConsistencyError, match="Malformed JSON at line 2"):
        RunArtifactWriter.resume(base_dir=tmp_path, run_id=run_id)


def test_v02_resume_empty_blank_line_aborts(tmp_path: Path) -> None:
    """V-02: Empty/corrupt blank row in raw_results.jsonl must abort RESUME loudly."""
    run_id = "v02-blank-row"
    manifest = create_manifest(run_id=run_id, system="traditional_rag")
    writer = RunArtifactWriter.create(base_dir=tmp_path, manifest=manifest)
    res = QueryResult.create_success(question_id="q-1", system="traditional_rag", answer="پاسخ")
    writer.append_result(res)
    writer.close()

    # Append empty row
    paths = RunPaths(tmp_path, run_id)
    with open(paths.raw_results_path, "a", encoding="utf-8") as f:
        f.write("\n")

    with pytest.raises(ArtifactConsistencyError, match="Corrupt or empty row at line 2"):
        RunArtifactWriter.resume(base_dir=tmp_path, run_id=run_id)


def test_v02_resume_duplicate_question_id_aborts(tmp_path: Path) -> None:
    """V-02: Duplicate question IDs in durable raw_results.jsonl must abort RESUME loudly."""
    run_id = "v02-dup-qid"
    manifest = create_manifest(run_id=run_id, system="traditional_rag")
    writer = RunArtifactWriter.create(base_dir=tmp_path, manifest=manifest)
    res1 = QueryResult.create_success(
        question_id="q-dup", system="traditional_rag", answer="پاسخ ۱"
    )
    writer.append_result(res1)
    writer.close()

    # Artificially inject duplicate question_id
    paths = RunPaths(tmp_path, run_id)
    res2 = QueryResult.create_success(
        question_id="q-dup", system="traditional_rag", answer="پاسخ ۲"
    )
    with open(paths.raw_results_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(res2.model_dump(mode="json")) + "\n")

    with pytest.raises(DuplicateResultError, match="Duplicate question_id 'q-dup'"):
        RunArtifactWriter.resume(base_dir=tmp_path, run_id=run_id)


def test_v02_resume_foreign_system_or_variant_aborts(tmp_path: Path) -> None:
    """V-02: Foreign system or variant result must abort RESUME loudly."""
    run_id = "v02-foreign-id"
    manifest = create_manifest(run_id=run_id, system="traditional_rag", variant="v1")
    writer = RunArtifactWriter.create(base_dir=tmp_path, manifest=manifest)
    writer.close()

    paths = RunPaths(tmp_path, run_id)
    foreign_res = QueryResult.create_success(
        question_id="q-1", system="lightrag", variant="v1", answer="پاسخ"
    )
    with open(paths.raw_results_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(foreign_res.model_dump(mode="json")) + "\n")

    with pytest.raises(IdentityMismatchError, match="foreign system 'lightrag'"):
        RunArtifactWriter.resume(base_dir=tmp_path, run_id=run_id)


def test_v02_resume_malformed_errors_aborts(tmp_path: Path) -> None:
    """V-02: Malformed errors.jsonl must abort RESUME loudly."""
    run_id = "v02-malformed-errors"
    manifest = create_manifest(run_id=run_id, system="traditional_rag")
    writer = RunArtifactWriter.create(base_dir=tmp_path, manifest=manifest)
    writer.close()

    paths = RunPaths(tmp_path, run_id)
    with open(paths.errors_path, "w", encoding="utf-8") as f:
        f.write('{"invalid_json": true\n')

    with pytest.raises(ArtifactConsistencyError, match="Malformed JSON"):
        RunArtifactWriter.resume(base_dir=tmp_path, run_id=run_id)


def test_v02_resume_missing_expected_error_aborts(tmp_path: Path) -> None:
    """V-02: Failed query result with missing error in errors.jsonl must abort RESUME."""
    run_id = "v02-missing-error"
    manifest = create_manifest(run_id=run_id, system="traditional_rag")
    writer = RunArtifactWriter.create(base_dir=tmp_path, manifest=manifest)

    res = QueryResult.create_failure(
        question_id="q-fail",
        system="traditional_rag",
        error_message="شکست خط لوله",
        status=QueryStatus.FAILED,
    )
    writer.append_result(res)
    writer.close()

    # Clear errors.jsonl to simulate missing error record
    paths = RunPaths(tmp_path, run_id)
    paths.errors_path.unlink()

    with pytest.raises(ArtifactConsistencyError, match="Missing expected error record"):
        RunArtifactWriter.resume(base_dir=tmp_path, run_id=run_id)


def test_v02_resume_orphan_error_aborts(tmp_path: Path) -> None:
    """V-02: Orphan error in errors.jsonl for question not failed must abort RESUME."""
    run_id = "v02-orphan-error"
    manifest = create_manifest(run_id=run_id, system="traditional_rag")
    writer = RunArtifactWriter.create(base_dir=tmp_path, manifest=manifest)
    res = QueryResult.create_success(
        question_id="q-ok", system="traditional_rag", answer="پاسخ موفق"
    )
    writer.append_result(res)
    writer.close()

    # Artificially inject an error for question q-ok (which succeeded)
    paths = RunPaths(tmp_path, run_id)
    orphan_err = StructuredError(
        question_id="q-ok", system="traditional_rag", message="خطای نامعتبر"
    )
    with open(paths.errors_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(orphan_err.model_dump(mode="json")) + "\n")

    with pytest.raises(ArtifactConsistencyError, match="Orphan error record.*succeeded"):
        RunArtifactWriter.resume(base_dir=tmp_path, run_id=run_id)


def test_v02_resume_duplicate_error_record_aborts(tmp_path: Path) -> None:
    """V-02: Duplicate error records for same question_id in errors.jsonl must abort RESUME."""
    run_id = "v02-dup-error"
    manifest = create_manifest(run_id=run_id, system="traditional_rag")
    writer = RunArtifactWriter.create(base_dir=tmp_path, manifest=manifest)

    res = QueryResult.create_failure(
        question_id="q-fail",
        system="traditional_rag",
        error_message="شکست",
        status=QueryStatus.FAILED,
    )
    writer.append_result(res)
    writer.close()

    # Inject duplicate error record in errors.jsonl
    paths = RunPaths(tmp_path, run_id)
    assert res.error is not None
    with open(paths.errors_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(res.error.model_dump(mode="json")) + "\n")

    with pytest.raises(ArtifactConsistencyError, match="Duplicate error record"):
        RunArtifactWriter.resume(base_dir=tmp_path, run_id=run_id)


# -------------------------------------------------------------------------
# V-03: Make RunManifest Truly Safe Against Update Bypass
# -------------------------------------------------------------------------


def test_v03_manifest_immutable_and_model_copy_update_validation() -> None:
    """V-03: RunManifest is frozen and model_copy(update=...) re-validates invariants."""
    manifest = create_manifest(run_id="v03-copy-val", system="traditional_rag")

    # Direct mutation blocked
    with pytest.raises(ValidationError):
        manifest.total_queries = 10

    # Negative counters via model_copy(update=...) blocked
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        manifest.model_copy(update={"total_queries": -1})

    # COMPLETED status without finished_at via model_copy(update=...) blocked
    with pytest.raises(ValidationError, match="finished_at must be set"):
        manifest.model_copy(update={"status": RunStatus.COMPLETED, "finished_at": None})

    # Conflicting benchmark_config identity via model_copy(update=...) blocked
    with pytest.raises(ValidationError, match="contradicts benchmark_config.system"):
        manifest.model_copy(update={"benchmark_config": {"system": "other_system"}})


def test_v03_save_manifest_revalidates_completely(tmp_path: Path) -> None:
    """V-03: save_manifest() re-validates manifest before atomic serialization."""
    manifest = create_manifest(run_id="v03-save-val", system="traditional_rag")
    manifest_path = tmp_path / "manifest.json"

    # Valid save succeeds
    save_manifest(manifest, manifest_path)
    assert manifest_path.exists()

    # Loaded manifest is strictly validated
    loaded = load_manifest(manifest_path)
    assert loaded.run_id == "v03-save-val"


# -------------------------------------------------------------------------
# V-04: Fully Recursive Secret Sanitization
# -------------------------------------------------------------------------


def test_v04_deeply_nested_url_and_dict_redaction() -> None:
    """V-04: Recursively sanitize canary credentials in arbitrary nested dicts, lists, and URLs."""
    canary = "CANARY_SECRET_XYZ"
    data = {
        "urls": [
            f"https://user:{canary}@host/path?token={canary}",
            "https://safe.example.com/api",
        ],
        "nested_levels": [
            {
                "branches": [
                    {
                        "api_key": canary,
                        "password": canary,
                        "normal_info": "اطلاعات مجاز",
                    }
                ]
            }
        ],
        "legitimate_fields": {
            "author": "امین نورایی",
            "tokenizer": "cl100k_base",
            "tokenizer_name": "persian_bpe",
            "authorization_mode": "bearer_standard",
            "token_count": 512,
            "input_tokens": 100,
            "output_tokens": 50,
        },
    }

    sanitized = sanitize_value(data)

    # Convert to serialized string to verify no trace of canary exists
    serialized = json.dumps(sanitized)
    assert canary not in serialized
    assert "https://user:[REDACTED]@host/path?token=" in serialized
    assert "[REDACTED]" in serialized

    # Verify legitimate fields are strictly preserved
    legit = sanitized["legitimate_fields"]
    assert legit["author"] == "امین نورایی"
    assert legit["tokenizer"] == "cl100k_base"
    assert legit["tokenizer_name"] == "persian_bpe"
    assert legit["authorization_mode"] == "bearer_standard"
    assert legit["token_count"] == 512
    assert legit["input_tokens"] == 100
    assert legit["output_tokens"] == 50


# -------------------------------------------------------------------------
# V-05: Race-Safe Stale Lock Takeover
# -------------------------------------------------------------------------


def test_v05_multiprocessing_stale_lock_race(tmp_path: Path) -> None:
    """V-05: Race on stale lock produces exactly one winner and one concurrency failure."""
    run_id = "v05-lock-race"
    paths = RunPaths(tmp_path, run_id)
    paths.ensure_directories()

    manifest = create_manifest(run_id=run_id, system="traditional_rag")
    save_manifest(manifest, paths.manifest_path)

    # Plant a stale lock with dead PID 999999
    dead_pid = 999999
    stale_token = str(uuid.uuid4())
    lock_data = {
        "owner_token": stale_token,
        "pid": dead_pid,
        "host": platform.node(),
        "acquired_at": datetime.now(UTC).isoformat(),
        "run_id": run_id,
    }
    with open(paths.lock_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(lock_data))

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()

    p1 = ctx.Process(target=_contender_worker, args=(str(tmp_path), run_id, result_queue))
    p2 = ctx.Process(target=_contender_worker, args=(str(tmp_path), run_id, result_queue))

    p1.start()
    p2.start()

    p1.join(timeout=10)
    p2.join(timeout=10)

    results = []
    while not result_queue.empty():
        results.append(result_queue.get_nowait())

    statuses = [r[0] for r in results]
    assert "success" in statuses, f"Expected one successful takeover, got results: {results}"
    assert "concurrency_error" in statuses, (
        f"Expected one concurrency rejection, got results: {results}"
    )
    assert len(statuses) == 2


# -------------------------------------------------------------------------
# V-06: Enforce Manifest/Config Identity in RunManifest Itself
# -------------------------------------------------------------------------


def test_v06_manifest_config_identity_validation() -> None:
    """V-06: Direct RunManifest construction rejects contradictory benchmark_config identity."""
    # Contradictory system
    with pytest.raises(ValidationError, match="contradicts benchmark_config.system"):
        RunManifest(
            run_id="v06-id-1",
            system="traditional_rag",
            variant="baseline",
            benchmark_config={"system": "lightrag", "variant": "baseline"},
            status=RunStatus.RUNNING,
        )

    # Contradictory variant
    with pytest.raises(ValidationError, match="contradicts benchmark_config.variant"):
        RunManifest(
            run_id="v06-id-2",
            system="traditional_rag",
            variant="baseline",
            benchmark_config={"system": "traditional_rag", "variant": "graph_v2"},
            status=RunStatus.RUNNING,
        )

    # Consistent identity succeeds
    valid_manifest = RunManifest(
        run_id="v06-id-valid",
        system="traditional_rag",
        variant="baseline",
        benchmark_config={"system": "traditional_rag", "variant": "baseline"},
        status=RunStatus.RUNNING,
    )
    assert valid_manifest.system == "traditional_rag"


# -------------------------------------------------------------------------
# V-07: Fix or Clarify Nested Immutability
# -------------------------------------------------------------------------


def test_v07_nested_immutability_frozen_dict_and_list() -> None:
    """V-07: Nested metadata structures are deeply immutable and block in-place mutation."""
    doc = BenchmarkDocument(
        document_id="doc-v07",
        title="عنوان سند",
        text="متن سند",
        sections=[DocumentSection(section_id="s1", heading="بخش ۱", text="متن بخش")],
        metadata={
            "tag": "history",
            "tag_list": ["iran", "tehran"],
            "nested": {"level": 1},
        },
    )

    # In-place dictionary assignment blocked
    with pytest.raises(TypeError, match="FrozenDict is immutable"):
        doc.metadata["tag"] = "geography"

    # In-place list mutation blocked
    with pytest.raises(TypeError, match="FrozenList is immutable"):
        doc.metadata["tag_list"].append("isfahan")

    # In-place nested dictionary mutation blocked
    with pytest.raises(TypeError, match="FrozenDict is immutable"):
        doc.metadata["nested"]["level"] = 2

    # In-place deletion blocked
    with pytest.raises(TypeError, match="FrozenDict is immutable"):
        del doc.metadata["tag"]

    # Question metadata and query_metadata immutability
    q = BenchmarkQuestion(
        question_id="q-v07",
        question="پرسش آزمایشی؟",
        question_type=QuestionType.DIRECT_FACTUAL,
        scope=QuestionScope.LOCAL,
        gold_answer="پاسخ",
        query_metadata={"param_list": [1, 2, 3]},
        metadata={"eval_dict": {"score": 10}},
    )

    with pytest.raises(TypeError, match="FrozenList is immutable"):
        q.query_metadata["param_list"].append(4)

    with pytest.raises(TypeError, match="FrozenDict is immutable"):
        q.metadata["eval_dict"]["score"] = 20


# -------------------------------------------------------------------------
# V-08: Guarantee Capability Validation Before Adapter Use
# -------------------------------------------------------------------------


def test_v08_adapter_activate_enforces_capabilities() -> None:
    """V-08: activate() enforces capability validation before prepare() or indexing/querying."""

    class BrokenAdapter(BaseRAGAdapter):
        def __init__(self) -> None:
            self.prepare_called = False

        @property
        def system_name(self) -> str:
            return "broken_rag"

        @property
        def variant_name(self) -> str:
            return "v1"

        @property
        def capabilities(self) -> AdapterCapabilities:
            return AdapterCapabilities(
                supports_update=True
            )  # Declares update but doesn't override it

        def prepare(self, config: BenchmarkConfig) -> None:
            self.prepare_called = True

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

    adapter = BrokenAdapter()
    config = BenchmarkConfig(system="broken_rag", variant="v1")

    # activate() must fail BEFORE prepare() is executed
    with pytest.raises(InvalidCapabilityDeclarationError, match="does not override the update"):
        adapter.activate(config)

    assert not adapter.prepare_called


def test_v08_indexing_and_health_metadata_strict_json() -> None:
    """V-08: IndexingResult.metadata and HealthCheckResult.details validate strict JSON."""
    # Tuples rejected in IndexingResult
    with pytest.raises(ValidationError, match="Strict JSON rejects tuple"):
        IndexingResult(metadata={"invalid_tuple": (1, 2)})

    # Non-finite floats rejected in HealthCheckResult
    with pytest.raises(ValidationError, match="Strict JSON forbids non-finite float"):
        HealthCheckResult(healthy=True, system="rag", details={"bad_val": float("inf")})
