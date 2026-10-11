"""The app itself: every page must render, and the chat must drive a model.

Runs the real Streamlit script through Streamlit's own test harness. No browser,
no network, no GPU -- if these pass, the UI is wired correctly end to end.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

AppTest = pytest.importorskip(
    "streamlit.testing.v1", reason="streamlit test harness required"
).AppTest


@pytest.fixture(scope="module")
def app():
    os.environ["BIONEMO_BACKEND"] = "simulate"   # no GPU in CI, by design
    app_test = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300)
    app_test.run()
    return app_test


def _exceptions(app_test) -> list:
    return [str(e.value) for e in app_test.exception]


def test_app_boots_without_exceptions(app):
    assert _exceptions(app) == []


def test_the_3d_viewer_asset_is_vendored():
    asset = ROOT / "bionemo_web" / "static" / "3Dmol-min.js"
    assert asset.exists() and asset.stat().st_size > 400_000


@pytest.mark.parametrize(
    "page",
    ["Structure Lab", "Molecule Workbench", "Screening Pipeline", "Administration", "History"],
)
def test_every_page_renders(app, page):
    app.radio(key="nav_radio").set_value(page).run()
    assert _exceptions(app) == [], f"{page} raised: {_exceptions(app)}"


def test_administration_reports_the_environment(app):
    app.radio(key="nav_radio").set_value("Administration").run()
    text = " ".join(m.value for m in app.markdown)
    assert "GPU" in text or "gpu" in text
    assert _exceptions(app) == []


def test_chat_runs_a_fold_and_shows_a_result(app):
    app.radio(key="nav_radio").set_value("Chat").run()
    assert len(app.chat_input) == 1
    sequence = ("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPD"
                "AQFEVVHSLAKWKR")
    app.chat_input[0].set_value(f"fold {sequence}").run()
    assert _exceptions(app) == []
    text = " ".join(m.value for m in app.markdown)
    assert "residues" in text        # the fold summary
    assert "SIMULATED" in text       # and the honest badge


def test_chat_generates_molecules(app):
    app.radio(key="nav_radio").set_value("Chat").run()
    app.chat_input[0].set_value(
        "generate 5 analogues of CC(=O)Oc1ccccc1C(=O)O optimising QED"
    ).run()
    assert _exceptions(app) == []
    text = " ".join(m.value for m in app.markdown)
    assert "Generated" in text or "generated" in text
    assert len(app.dataframe) >= 1, "the molecule table should be rendered"


def test_chat_answers_a_question(app):
    app.radio(key="nav_radio").set_value("Chat").run()
    app.chat_input[0].set_value("what can you do?").run()
    assert _exceptions(app) == []
    text = " ".join(m.value for m in app.markdown)
    assert "fold" in text.lower()
