"""Tests for canonical Persian document schema."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from rag_benchmark.schemas.document import BenchmarkDocument, DocumentSection


def test_valid_persian_document_creation() -> None:
    """Test valid Persian document with sections and Unicode characters."""
    section1 = DocumentSection(
        section_id="sec-01",
        heading="مقدمه و پیش‌زمینه",
        text="این یک بخش آزمایشی است که شامل نیم‌فاصله‌ها و حروف فارسی مانند گ، چ، پ، ژ می‌باشد.",
        order=0,
        metadata={"source_page": 1},
    )
    doc = BenchmarkDocument(
        document_id="doc-fa-101",
        title="معماری بازیابی اطلاعات در زبان فارسی",
        text="متن کامل سند شامل اطلاعات کلیدی پیرامون سیستم‌های RAG.",
        language="fa",
        source="persian_corpus_v1",
        url="https://example.ir/docs/rag-persian",
        sections=[section1],
        metadata={"author": "تیم ارزیابی", "year": 1403},
    )

    assert doc.document_id == "doc-fa-101"
    assert doc.title == "معماری بازیابی اطلاعات در زبان فارسی"
    assert doc.language == "fa"
    assert len(doc.sections) == 1
    assert doc.sections[0].heading == "مقدمه و پیش‌زمینه"
    assert doc.metadata["year"] == 1403


def test_persian_unicode_roundtrip() -> None:
    """Persian Unicode text and ZWNJ must survive serialization round trip unchanged."""
    title_with_zwnj = "بررسی سیستم‌های پیشنهاددهنده و بازیابی‌محور"
    persian_text = (
        "حروف خاص فارسی: گ، چ، پ، ژ. نیم‌فاصله: می‌توانیم، خانه‌ها، کتاب‌ها. اعداد فارسی: ۱۲۳۴۵"
    )

    doc = BenchmarkDocument(
        document_id="doc-unicode-01",
        title=title_with_zwnj,
        text=persian_text,
        sections=[
            DocumentSection(
                section_id="s1",
                heading="فصل اول: پیش‌نیازها",
                text="متن بخش ۱ با نیم‌فاصله.",
                order=0,
            )
        ],
    )

    # Serialize to JSON with ensure_ascii=False
    json_str = json.dumps(doc.model_dump(mode="json"), ensure_ascii=False)

    # Confirm raw Persian characters exist in json_str (not escaped)
    assert "پیشنهاددهنده" in json_str
    assert "نیم‌فاصله" in json_str
    assert "گ، چ، پ، ژ" in json_str

    # Deserialize back
    data = json.loads(json_str)
    deserialized = BenchmarkDocument.model_validate(data)

    assert deserialized.document_id == doc.document_id
    assert deserialized.title == title_with_zwnj
    assert deserialized.text == persian_text
    assert deserialized.sections[0].heading == "فصل اول: پیش‌نیازها"


def test_reject_empty_document_id() -> None:
    """document_id cannot be empty or whitespace only."""
    with pytest.raises(ValidationError):
        BenchmarkDocument(
            document_id="",
            title="عنوان معتبر",
            text="متن معتبر",
        )

    with pytest.raises(ValidationError):
        BenchmarkDocument(
            document_id="   ",
            title="عنوان معتبر",
            text="متن معتبر",
        )


def test_reject_empty_title() -> None:
    """title cannot be empty or whitespace only."""
    with pytest.raises(ValidationError):
        BenchmarkDocument(
            document_id="doc-1",
            title="",
            text="متن معتبر",
        )

    with pytest.raises(ValidationError):
        BenchmarkDocument(
            document_id="doc-1",
            title="   \t   ",
            text="متن معتبر",
        )


def test_reject_empty_section_id() -> None:
    """section_id cannot be empty."""
    with pytest.raises(ValidationError):
        DocumentSection(
            section_id="",
            heading="عنوان",
            text="متن",
            order=0,
        )


def test_reject_negative_section_order() -> None:
    """section order must be non-negative."""
    with pytest.raises(ValidationError):
        DocumentSection(
            section_id="sec-1",
            heading="عنوان",
            text="متن",
            order=-1,
        )


def test_reject_non_json_serializable_metadata() -> None:
    """Document metadata must be JSON-serializable."""

    class CustomObj:
        pass

    with pytest.raises(ValidationError):
        BenchmarkDocument(
            document_id="doc-invalid-meta",
            title="عنوان",
            text="متن",
            metadata={"bad_key": CustomObj()},
        )
