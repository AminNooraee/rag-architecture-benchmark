import pytest
import pathlib

@pytest.fixture(scope="session", autouse=True)
def _override_tmp_path(tmp_path_factory: pytest.TempPathFactory):
    # Use a writable directory inside the project for pytest temporary files
    new_base = pathlib.Path(__file__).parent / "tmp_pytest"
    new_base.mkdir(exist_ok=True)
    tmp_path_factory._basetemp = new_base
