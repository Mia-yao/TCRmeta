"""Test configuration.

torch/transformers are heavy dependencies needed for the actual
embedding model (real end users will have them installed). The pure
algorithmic pieces this test suite targets — column validation,
downsampling, the CSS density-scoring formula, the reference-fitting
math, the weighted energy-distance formula — don't need torch at all,
and every torch-requiring import in those modules (embed_repertoire)
was made lazy (imported inside function bodies, not at module top) so
these modules can be loaded and unit-tested without torch/transformers
installed.

To do that, we register a stub 'tcrmeta' package in sys.modules (with
__path__ pointing at the real src/tcrmeta directory) instead of letting
Python execute the real tcrmeta/__init__.py — which itself eagerly
imports the torch-dependent embedding module. Submodules (tcrmeta.css,
tcrmeta.reference, tcrmeta.energy, ...) are then imported normally
through this stub package, so their relative imports resolve correctly.
"""
import importlib
import sys
import types
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"


def _stub_tcrmeta_package():
    if "tcrmeta" not in sys.modules:
        pkg = types.ModuleType("tcrmeta")
        pkg.__path__ = [str(SRC_DIR / "tcrmeta")]
        sys.modules["tcrmeta"] = pkg
    return sys.modules["tcrmeta"]


def load_tcrmeta_module(dotted_name: str):
    """Import e.g. 'css', 'reference', 'energy', '_utils' from the real
    source tree without triggering tcrmeta/__init__.py's torch import.
    """
    _stub_tcrmeta_package()
    return importlib.import_module(f"tcrmeta.{dotted_name}")


@pytest.fixture(scope="session")
def tcrmeta_utils():
    return load_tcrmeta_module("_utils")


@pytest.fixture(scope="session")
def tcrmeta_css():
    return load_tcrmeta_module("css")


@pytest.fixture(scope="session")
def tcrmeta_reference():
    return load_tcrmeta_module("reference")


@pytest.fixture(scope="session")
def tcrmeta_energy():
    return load_tcrmeta_module("energy")


@pytest.fixture(scope="session")
def tcrmeta_umap_plot():
    return load_tcrmeta_module("umap_plot")


def torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False
