"""The console's look: one palette, one stylesheet, a few small HTML builders.

Everything visual lives here so the pages stay about science. The rules are:

* NVIDIA green on a near-black "ink" background, in a dark room, at 2am;
* results are cards -- a titled panel with a coloured badge saying where the
  numbers came from;
* monospace for anything a scientist may copy (SMILES, sequences, PDB text,
  file paths), because transcription errors in a structure file are expensive.

The HTML builders take already-formatted strings for badges and metrics, so a
caller can compose a card without this module knowing what any tool returns.
They deliberately do not escape their input: they are fed trusted markup built
by :mod:`bionemo_web.ui.cards` from data the app itself produced.
"""

from __future__ import annotations

from typing import Dict, List

#: Semantic colour map. ``accent`` is the BioNeMo/NVIDIA green; the rest are
#: chosen to stay distinguishable on the dark surface and to be readable by
#: red/green colour-blind users (good/warn/bad differ in lightness as well).
PALETTE: Dict[str, str] = {
    "bg": "#0b0f14",
    "bg_soft": "#111823",
    "surface": "#151d29",
    "surface_2": "#1b2534",
    "border": "#243044",
    "text": "#e8eef7",
    "text_dim": "#93a4bd",
    "accent": "#76b900",
    "accent_soft": "#8fd41f",
    "accent_dim": "#4d7a00",
    "good": "#34d399",
    "warn": "#fbbf24",
    "bad": "#f87171",
    "info": "#60a5fa",
    "violet": "#a78bfa",
}

#: Secondary-structure colours, in the usual convention (helix green, sheet
#: blue, coil grey) with the remaining DSSP codes given readable neighbours.
SS_COLOURS: Dict[str, str] = {
    "H": "#76b900",
    "G": "#a3e635",
    "I": "#d9f99d",
    "E": "#60a5fa",
    "B": "#93c5fd",
    "T": "#fbbf24",
    "S": "#f59e0b",
    "C": "#64748b",
}

#: Categorical colourway for plots with many series. Ordered so that the first
#: two series (usually "ours" and "the other one") are maximally distinct.
CHART_SEQUENCE: List[str] = [
    "#76b900", "#60a5fa", "#a78bfa", "#fbbf24",
    "#34d399", "#f472b6", "#22d3ee", "#fb923c",
]

#: How a run's provenance is labelled on a badge, and which tone it gets.
#: Simulations are amber on purpose: they must never look like measured output.
BACKEND_LABELS: Dict[str, str] = {
    "local": "LOCAL NIM",
    "hosted": "NVIDIA HOSTED",
    "framework": "LOCAL RUNTIME",
    "simulate": "SIMULATED",
    "local analysis": "LOCAL ANALYSIS",
}


def css() -> str:
    """The application stylesheet, injected once per run by ``app.py``."""
    return """
<style>
:root {
  --bm-bg: #0b0f14; --bm-surface: #151d29; --bm-surface-2: #1b2534;
  --bm-border: #243044; --bm-text: #e8eef7; --bm-dim: #93a4bd;
  --bm-accent: #76b900; --bm-good: #34d399; --bm-warn: #fbbf24; --bm-bad: #f87171;
  --bm-info: #60a5fa; --bm-violet: #a78bfa;
}
.stApp { background: radial-gradient(1200px 600px at 15% -10%, #16202e 0%, #0b0f14 55%) no-repeat; color: #e8eef7; }
section[data-testid="stSidebar"] { background: #111823; border-right: 1px solid #243044; }
h1, h2, h3, h4 { color: #e8eef7; letter-spacing: -0.01em; }
a { color: #8fd41f; }

/* ---- brand header ---- */
.bm-brand { display:flex; align-items:center; gap:12px; padding: 4px 0 12px 0; }
.bm-brand-mark {
  width: 38px; height: 38px; border-radius: 10px; flex: 0 0 38px;
  background: linear-gradient(140deg, #76b900 0%, #4d7a00 100%);
  display:flex; align-items:center; justify-content:center; font-weight:800; color:#08130a;
  font-size: 15px; letter-spacing:-0.5px;
}
.bm-brand-title { font-size: 1.06rem; font-weight: 700; line-height: 1.15; }
.bm-brand-sub { font-size: 0.74rem; color: #93a4bd; letter-spacing: 0.04em; text-transform: uppercase; }

/* ---- badges ---- */
.bm-badge {
  display:inline-flex; align-items:center; gap:6px; padding: 2px 9px; border-radius: 999px;
  font-size: 0.68rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase;
  border: 1px solid var(--bm-border); background: #1b2534; color: #93a4bd;
}
.bm-badge .dot { width:7px; height:7px; border-radius:50%; background: currentColor; }
.bm-badge[data-tone="good"] { color: #34d399; border-color: rgba(52,211,153,0.35); }
.bm-badge[data-tone="warn"] { color: #fbbf24; border-color: rgba(251,191,36,0.4); }
.bm-badge[data-tone="bad"] { color: #f87171; border-color: rgba(248,113,113,0.4); }
.bm-badge[data-tone="info"] { color: #60a5fa; border-color: rgba(96,165,250,0.35); }
.bm-badge[data-tone="accent"] { color: #8fd41f; border-color: rgba(118,185,0,0.4); }
.bm-badge[data-tone="violet"] { color: #a78bfa; border-color: rgba(167,139,250,0.35); }

/* ---- metric cards ---- */
.bm-metric {
  background: linear-gradient(180deg, #151d29 0%, #111823 100%);
  border: 1px solid #243044; border-radius: 12px; padding: 12px 14px; height: 100%;
}
.bm-metric .label { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.07em; color: #93a4bd; }
.bm-metric .value { font-size: 1.34rem; font-weight: 700; margin-top: 3px; color: #e8eef7; }
.bm-metric .note { font-size: 0.72rem; color: #93a4bd; margin-top: 2px; }

/* ---- result cards ---- */
.bm-card {
  border: 1px solid #243044; border-radius: 14px; background: #151d29;
  padding: 14px 16px; margin: 4px 0 10px 0;
}
.bm-card-head { display:flex; align-items:center; justify-content: space-between; gap: 10px; margin-bottom: 8px; }
.bm-card-title { font-weight: 700; font-size: 0.98rem; }
.bm-card-meta { font-size: 0.74rem; color: #93a4bd; }
.bm-summary { font-size: 0.9rem; line-height: 1.55; color: #e8eef7; }
.bm-mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.8rem; word-break: break-all; }

/* ---- chat ---- */
.bm-user-msg {
  background: #1b2534; border: 1px solid #243044; border-radius: 14px 14px 4px 14px;
  padding: 10px 14px; margin: 6px 0 6px 12%;
}
.bm-assistant-msg {
  background: transparent; border-left: 3px solid #76b900; padding: 2px 0 2px 14px; margin: 10px 0 4px 0;
}
.bm-msg-role { font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.08em; color: #93a4bd; margin-bottom: 4px; }
.bm-warning {
  border-left: 3px solid #fbbf24; background: rgba(251,191,36,0.07); padding: 8px 12px;
  border-radius: 0 8px 8px 0; font-size: 0.82rem; margin: 6px 0; color: #e8eef7;
}
.bm-empty { text-align:center; color: #93a4bd; padding: 26px 10px; font-size: 0.9rem; }

/* ---- streamlit tightening ---- */
div[data-testid="stChatInput"] textarea { font-size: 0.95rem; }
.stButton>button { border-radius: 9px; border: 1px solid #243044; background: #1b2534; color: #e8eef7; }
.stButton>button:hover { border-color: #76b900; color: #8fd41f; }
.stButton>button[kind="primary"] { background: #76b900; border-color: #76b900; color: #08130a; font-weight: 700; }
.stTabs [data-baseweb="tab"] { font-size: 0.86rem; }
div[data-testid="stDataFrame"] { border: 1px solid #243044; border-radius: 10px; }
code { background: #111823; color: #8fd41f; }
</style>
"""


def plotly_layout(height: int = 340, title: str = "") -> Dict:
    """Shared Plotly layout: transparent surfaces, dark grid, our colourway."""
    return {
        "template": "plotly_dark",
        "height": height,
        "title": {"text": title, "font": {"size": 14, "color": PALETTE["text"]}},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"color": PALETTE["text_dim"], "size": 12},
        "margin": {"l": 50, "r": 20, "t": 40, "b": 40},
        "legend": {"bgcolor": "rgba(0,0,0,0)", "font": {"size": 11}},
        "colorway": CHART_SEQUENCE,
        "xaxis": {"gridcolor": PALETTE["border"], "zerolinecolor": PALETTE["border"]},
        "yaxis": {"gridcolor": PALETTE["border"], "zerolinecolor": PALETTE["border"]},
    }


def badge_html(label: str, tone: str = "accent", dot: bool = True) -> str:
    """A pill. Tones: accent, info, violet, good, warn, bad."""
    dot_html = '<span class="dot"></span>' if dot else ""
    return f'<span class="bm-badge" data-tone="{tone}">{dot_html}{label}</span>'


def metric_html(label: str, value: str, note: str = "") -> str:
    """One number with its unit. Use for the handful of values worth reading."""
    note_html = f'<div class="note">{note}</div>' if note else ""
    return (
        f'<div class="bm-metric"><div class="label">{label}</div>'
        f'<div class="value">{value}</div>{note_html}</div>'
    )


def card_html(title: str, body: str, *, meta: str = "", badges: str = "") -> str:
    """A result panel: title, provenance badges, a small meta line, then body."""
    return (
        f'<div class="bm-card"><div class="bm-card-head">'
        f'<span class="bm-card-title">{title}</span>'
        f'<span>{badges}<span class="bm-card-meta">{meta}</span></span>'
        f"</div>{body}</div>"
    )


def empty_state(message: str, hint: str = "") -> str:
    """Placeholder for a panel with nothing to show yet."""
    hint_html = f'<div class="bm-card-meta" style="margin-top:6px">{hint}</div>' if hint else ""
    return f'<div class="bm-empty">{message}{hint_html}</div>'
