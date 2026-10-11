"""Shared fixtures. Tests run entirely offline: no network, no GPU, no NIMs."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Point runtime state at a throwaway directory so tests never touch real data.
os.environ.setdefault("BIONEMO_CONSOLE_HOME", str(ROOT / ".runtime" / "pytest"))

SEQUENCE = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKR"
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
CAFFEINE = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"


@pytest.fixture(scope="session")
def sequence() -> str:
    return SEQUENCE


@pytest.fixture(scope="session")
def aspirin() -> str:
    return ASPIRIN


@pytest.fixture(scope="session")
def folded(sequence):
    """A folded model, computed once for the whole session (it is slow)."""
    from bionemo_web.bio import structure as st_mod

    return st_mod.fold_sequence(sequence, seed=5)
