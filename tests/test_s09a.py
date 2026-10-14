"""Session 9a notebook checks (offline, no key needed). Run with:  uv run pytest tests/test_s09a.py"""
import json
import re
from pathlib import Path

import pytest

NB_PATH = Path(__file__).resolve().parents[1] / "module01" / "s09a_first_call.ipynb"
KEY_LIKE = re.compile(r"gsk_[A-Za-z0-9]{20,}")          # looks like a real Groq key
LOAD = re.compile(r"load_dotenv\([^)]*override=True")
READS_SETTINGS = re.compile(r"os\.getenv\(|os\.environ")


def _source(cell):
    s = cell["source"]
    return "".join(s) if isinstance(s, list) else s


@pytest.fixture(scope="module")
def nb():
    nbformat = pytest.importorskip("nbformat")             # comes with jupyter
    node = nbformat.read(str(NB_PATH), as_version=4)
    nbformat.validate(node)                                 # raises if the notebook is not valid nbformat
    return node


@pytest.fixture(scope="module")
def code_cells(nb):
    return [_source(c) for c in nb.cells if c.cell_type == "code"]


def test_notebook_exists_and_is_valid(nb):
    assert nb.nbformat == 4
    assert any(c.cell_type == "markdown" for c in nb.cells)


def test_enough_code_cells(code_cells):
    assert len(code_cells) >= 8, f"only {len(code_cells)} code cells"


def test_every_code_cell_compiles(code_cells):
    for i, src in enumerate(code_cells):
        compile(src, f"code cell {i}", "exec")


def test_no_real_looking_key_anywhere():
    raw = NB_PATH.read_text(encoding="utf-8")               # sources AND any saved outputs
    found = KEY_LIKE.findall(raw)
    assert not found, "the notebook contains a string that looks like a real API key: remove it and revoke the key"


def test_settings_are_loaded_before_they_are_read(code_cells):
    first_read = next((i for i, s in enumerate(code_cells) if READS_SETTINGS.search(s)), None)
    assert first_read is not None, "no cell reads the settings"
    first_load = next((i for i, s in enumerate(code_cells) if LOAD.search(s)), None)
    assert first_load is not None, "no cell calls load_dotenv(..., override=True)"
    assert first_load <= 2, "load_dotenv(override=True) must be in one of the first code cells"
    assert first_load <= first_read, "settings are read before load_dotenv(override=True) runs"
    if first_load == first_read:                            # same cell: the load must come first in the text
        src = code_cells[first_load]
        assert LOAD.search(src).start() < READS_SETTINGS.search(src).start() # type: ignore


def test_model_cells_check_ready_before_calling(code_cells):
    for src in code_cells:
        if "chat.completions.create(" in src or "models.list()" in src:
            assert "if not ready" in src and "skipped" in src, "a model cell does not skip gracefully without a key"
