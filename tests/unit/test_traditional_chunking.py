"""Tests for deterministic Persian-safe chunking."""

import pytest

from rag_benchmark.adapters.traditional.chunking import (
    chunk_document,
    make_chunk_id,
    validate_chunking_params,
)
from rag_benchmark.schemas.document import BenchmarkDocument

ZWNJ = "\u200c"


def _doc(text: str, doc_id: str = "doc-1") -> BenchmarkDocument:
    return BenchmarkDocument(
        document_id=doc_id, title="عنوان آزمایشی", text=text, source="test-source"
    )


def test_chunk_ids_are_deterministic() -> None:
    doc = _doc(" ".join(f"کلمه{i}" for i in range(30)))
    first = chunk_document(doc, 64, 8)
    second = chunk_document(doc, 64, 8)
    assert len(first) > 1
    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert [chunk.text for chunk in first] == [chunk.text for chunk in second]
    assert [chunk.ordinal for chunk in first] == [chunk.ordinal for chunk in second]
    for position, chunk in enumerate(first):
        assert chunk.chunk_id == make_chunk_id(
            "doc-1", position, 64, 8, chunk.char_start, chunk.char_end, chunk.text
        )
        assert chunk.chunk_id.startswith(f"doc-1::chunk-{position:06d}::")
        assert chunk.ordinal == position


def test_single_chunk_equals_stripped_text() -> None:
    raw = "  سلام دنیا  "
    (chunk,) = chunk_document(_doc(raw), 100, 0)
    assert chunk.text == raw.strip()
    assert chunk.char_start == 0
    assert chunk.char_end == len(raw)


def test_zwnj_and_persian_preserved_verbatim() -> None:
    text = f"زبان{ZWNJ}فارسی دارای نیم{ZWNJ}فاصله است"
    (chunk,) = chunk_document(_doc(text), 100, 0)
    assert chunk.text == text
    assert ZWNJ in chunk.text


def test_interior_whitespace_preserved() -> None:
    text = "واژه   اول و واژه دوم"
    (chunk,) = chunk_document(_doc(text), 100, 0)
    assert "واژه   اول" in chunk.text


def test_whitespace_only_document_yields_no_chunks() -> None:
    assert chunk_document(_doc("   \n\t  "), 64, 8) == ()
    assert chunk_document(_doc(""), 64, 8) == ()


def test_empty_text_skipped_between_words() -> None:
    text = "alpha" + " " * 50 + "beta"
    chunks = chunk_document(_doc(text, "doc-gap"), 20, 0)
    assert [chunk.text for chunk in chunks] == ["alpha", "beta"]
    assert chunks[1].chunk_id == make_chunk_id(
        "doc-gap", 1, 20, 0, chunks[1].char_start, chunks[1].char_end, chunks[1].text
    )


def test_long_word_run_is_hard_cut() -> None:
    text = "خ" * 100
    chunks = chunk_document(_doc(text), 30, 0)
    assert len(chunks) == 4
    assert chunks[0].text == "خ" * 30
    assert chunks[-1].text == "خ" * 10
    for chunk in chunks:
        assert ZWNJ not in chunk.text


def test_word_boundary_backoff() -> None:
    text = "aaaa bbbb cccc dddd"
    chunks = chunk_document(_doc(text), 12, 0)
    assert [chunk.text for chunk in chunks] == ["aaaa bbbb", "cccc dddd"]


def test_overlap_advances_window_and_preserves_raw_span() -> None:
    text = "aaaa bbbb cccc dddd"
    overlap = 4
    chunks = chunk_document(_doc(text), 12, overlap)
    assert [chunk.text for chunk in chunks] == ["aaaa bbbb", "bbbb cccc", "cccc dddd"]
    for prev, nxt in zip(chunks[:-1], chunks[1:], strict=True):
        assert nxt.char_start <= prev.char_end
        span = text[nxt.char_start : prev.char_end]
        assert span
        assert text[prev.char_start : prev.char_end].endswith(span)
        assert text[nxt.char_start : nxt.char_end].startswith(span)


def test_no_duplicate_tail_chunk_after_final_full_window() -> None:
    text = "x " * 30  # 59 chars, windows of 20 with overlap 4
    chunks = chunk_document(_doc(text), 20, 4)
    tail = text[-4:]
    for chunk in chunks[-2:]:
        assert chunk.text != tail.strip()


def test_validate_chunking_params_rejects_invalid() -> None:
    with pytest.raises(ValueError):
        validate_chunking_params(0, 0)
    with pytest.raises(ValueError):
        validate_chunking_params(10, -1)
    with pytest.raises(ValueError):
        validate_chunking_params(10, 10)
    with pytest.raises(ValueError):
        validate_chunking_params(10, 11)


def test_chunk_document_propagates_invalid_params() -> None:
    with pytest.raises(ValueError):
        chunk_document(_doc("متن"), 10, 10)


def test_chunk_ids_deterministic_across_runs_with_full_identity() -> None:
    doc = _doc(" ".join(f"کلمه{i}" for i in range(30)), "doc-ident")
    first = chunk_document(doc, 64, 8)
    second = chunk_document(doc, 64, 8)
    assert len(first) > 1
    for a, b in zip(first, second, strict=True):
        assert a.chunk_id == b.chunk_id
        assert a.chunk_id == make_chunk_id(
            a.document_id, a.ordinal, 64, 8, a.char_start, a.char_end, a.text
        )


def test_chunk_ids_differ_across_chunk_size_config() -> None:
    text = "aaaa bbbb cccc dddd eeee ffff"
    small = chunk_document(_doc(text, "doc-cfg"), 12, 0)
    large = chunk_document(_doc(text, "doc-cfg"), 24, 0)
    assert small[0].chunk_id != large[0].chunk_id


def test_chunk_ids_differ_across_overlap_config() -> None:
    text = "aaaa bbbb cccc dddd"
    no_overlap = chunk_document(_doc(text, "doc-ov"), 12, 0)
    with_overlap = chunk_document(_doc(text, "doc-ov"), 12, 4)
    assert no_overlap[1].chunk_id != with_overlap[1].chunk_id


def test_chunk_ids_differ_when_text_changes() -> None:
    a = make_chunk_id("doc-t", 0, 64, 8, 0, 10, "متن اول")
    b = make_chunk_id("doc-t", 0, 64, 8, 0, 10, "متن دوم")
    assert a != b
    assert make_chunk_id("doc-t", 0, 64, 8, 0, 10, "متن اول") == a


def test_chunk_ids_differ_when_span_changes() -> None:
    a = make_chunk_id("doc-s", 0, 64, 8, 0, 10, "متن")
    b = make_chunk_id("doc-s", 0, 64, 8, 5, 15, "متن")
    assert a != b


def test_chunk_ids_preserve_persian_and_zwnj_verbatim() -> None:
    with_zwnj = make_chunk_id("doc-p", 0, 64, 8, 0, 10, f"نیم{ZWNJ}فاصله")
    without_zwnj = make_chunk_id("doc-p", 0, 64, 8, 0, 10, "نیمفاصله")
    assert with_zwnj != without_zwnj
    assert make_chunk_id("doc-p", 0, 64, 8, 0, 10, f"نیم{ZWNJ}فاصله") == with_zwnj


def test_chunk_id_format_has_ordinal_prefix_and_sha256_digest() -> None:
    chunk_id = make_chunk_id("doc-fmt", 7, 64, 8, 0, 12, "متن")
    prefix, digest = chunk_id.rsplit("::", 1)
    assert prefix == "doc-fmt::chunk-000007"
    assert len(digest) == 64
    int(digest, 16)
