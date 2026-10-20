"""Initial sanity test to verify pytest environment."""

import rag_benchmark


def test_package_import() -> None:
    assert rag_benchmark.__version__ == "0.1.0"
