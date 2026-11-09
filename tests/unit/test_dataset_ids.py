"""Tests for deterministic ID generation."""

from rag_benchmark.datasets.ids import deterministic_id


def test_deterministic_id_stability() -> None:
    id1 = deterministic_id("pquad-doc", "article1")
    id2 = deterministic_id("pquad-doc", "article1")
    assert id1 == id2


def test_different_components_produce_different_ids() -> None:
    id_a = deterministic_id("pquad-doc", "a")
    id_b = deterministic_id("pquad-doc", "b")
    assert id_a != id_b


def test_prefix_required() -> None:
    try:
        deterministic_id("", "x")
    except ValueError:
        assert True
    else:
        raise AssertionError("ValueError not raised for empty prefix")
