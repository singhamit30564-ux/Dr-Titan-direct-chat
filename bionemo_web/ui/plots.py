"""Charts. Every figure is built with Plotly so the app needs no extra service.

Each function returns a ``plotly.graph_objects.Figure`` (or ``None`` when there
is nothing to draw), so callers can simply do ``if fig: st.plotly_chart(fig)``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import plotly.graph_objects as go

from .theme import CHART_SEQUENCE, PALETTE, SS_COLOURS, plotly_layout


def embedding_projection(
    projection: Sequence[Sequence[float]],
    *,
    labels: Optional[Sequence[str]] = None,
    colour_values: Optional[Sequence[float]] = None,
    colour_label: str = "",
    title: str = "Per-residue embedding projection (PCA)",
    highlight: Optional[Sequence[int]] = None,
    height: int = 380,
) -> Optional[go.Figure]:
    """Scatter of the 2D embedding projection, optionally coloured by a value."""
    coords = np.asarray(projection, dtype=float)
    if coords.ndim != 2 or coords.shape[1] < 2 or len(coords) < 2:
        return None
    text = list(labels) if labels is not None else [str(i + 1) for i in range(len(coords))]
    fig = go.Figure()
    if colour_values is not None and len(colour_values) == len(coords):
        cbar = {"title": colour_label or "value", "thickness": 12, "len": 0.75}
        fig.add_trace(
            go.Scatter(
                x=coords[:, 0], y=coords[:, 1], mode="markers", text=text,
                hovertemplate="residue %{text}<br>PC1 %{x:.2f}<br>PC2 %{y:.2f}<extra></extra>",
                marker={
                    "size": 8, "color": list(colour_values), "colorscale": "Viridis",
                    "showscale": True, "colorbar": cbar,
                    "line": {"width": 0.5, "color": "rgba(0,0,0,0.35)"},
                },
                name="residues",
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=coords[:, 0], y=coords[:, 1], mode="markers+lines", text=text,
                line={"color": "rgba(118,185,0,0.25)", "width": 1},
                marker={"size": 7, "color": CHART_SEQUENCE[0]},
                hovertemplate="residue %{text}<extra></extra>", name="trace",
            )
        )
    if highlight:
        idx = [i for i in highlight if 0 <= i < len(coords)]
        if idx:
            fig.add_trace(
                go.Scatter(x=coords[idx, 0], y=coords[idx, 1], mode="markers",
                           marker={"size": 12, "symbol": "circle-open", "line": {"width": 2},
                                   "color": PALETTE["warn"]},
                           name="selected", hoverinfo="skip")
            )
    fig.update_layout(**plotly_layout(height, title))
    fig.update_xaxes(title_text="PC1")
    fig.update_yaxes(title_text="PC2")
    return fig


def contact_map(
    matrix: Sequence[Sequence[float]],
    *,
    ss: str = "",
    title: str = "Residue contact map",
    threshold: float = 8.0,
    height: int = 420,
) -> Optional[go.Figure]:
    """Contact map as a heatmap with the secondary-structure track on the edges."""
    m = np.asarray(matrix, dtype=float)
    if m.ndim != 2 or m.size == 0:
        return None
    fig = go.Figure(
        go.Heatmap(
            z=m, colorscale=[[0, PALETTE["bg"]], [0.35, "#1d4ed8"], [0.7, PALETTE["accent"]],
                             [1.0, "#fde68a"]],
            zmin=0, zmax=max(3.0, float(np.percentile(m, 99))),
            colorbar={"title": "CA-CA (A)", "thickness": 12, "len": 0.75},
            hovertemplate="res %{y} - res %{x}<br>%{z:.1f} A<extra></extra>",
        )
    )
    # Long-range contacts only (>4 apart in sequence) are the informative ones.
    fig.update_layout(**plotly_layout(height, title))
    fig.update_xaxes(title_text="residue", constrain="domain")
    fig.update_yaxes(title_text="residue", autorange="reversed", scaleanchor="x")
    if ss:
        colours = [SS_COLOURS.get(c, SS_COLOURS["C"]) for c in ss[: m.shape[0]]]
        bars = [[c, c] for c in colours]
        fig.add_trace(
            go.Scatter(
                x=list(range(len(colours))), y=[-6] * len(colours), mode="markers",
                marker={"size": 7, "symbol": "square", "color": colours},
                name="SS", hovertemplate="res %{x}<extra></extra>",
            )
        )
    return fig


def profile_track(
    values: Sequence[float],
    *,
    labels: Optional[Sequence[str]] = None,
    title: str = "",
    y_title: str = "",
    window: int = 1,
    ss: str = "",
    height: int = 260,
    colour: str = "",
) -> Optional[go.Figure]:
    """A per-residue line track (hydropathy, flexibility, confidence, ...)."""
    vals = np.asarray(values, dtype=float)
    if vals.size == 0:
        return None
    if window > 1 and vals.size > window:
        kernel = np.ones(window) / window
        vals = np.convolve(vals, kernel, mode="same")
    x = list(range(1, len(vals) + 1))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x, y=vals, mode="lines", line={"color": colour or CHART_SEQUENCE[0], "width": 1.8},
            text=list(labels) if labels else x, name=title or "value",
            hovertemplate="res %{x}: %{y:.3f}<extra></extra>",
        )
    )
    fig.update_layout(**plotly_layout(height, title))
    fig.update_xaxes(title_text="residue")
    fig.update_yaxes(title_text=y_title)
    return fig


def ss_track(ss: str, *, title: str = "Secondary structure", height: int = 150,
             predicted: str = "") -> Optional[go.Figure]:
    """Two-row secondary-structure ribbon (assigned vs predicted) for comparison."""
    if not ss and not predicted:
        return None
    rows = [(name, track) for name, track in (("assigned", ss), ("predicted", predicted)) if track]
    fig = go.Figure()
    for row_index, (name, track) in enumerate(rows):
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(track) + 1)),
                y=[row_index] * len(track),
                mode="markers",
                marker={"size": 9, "symbol": "square",
                        "color": [SS_COLOURS.get(c, SS_COLOURS["C"]) for c in track]},
                name=name, hovertemplate="res %{x}: " + name + "<extra></extra>",
            )
        )
    layout = plotly_layout(height, title)
    layout["showlegend"] = True
    fig.update_layout(**layout)
    fig.update_yaxes(tickvals=list(range(len(rows))), ticktext=[r[0] for r in rows],
                     showgrid=False, zeroline=False)
    fig.update_xaxes(title_text="residue")
    return fig


def molecule_set_summary(rows: Sequence[Dict[str, Any]], *, title: str = "Molecule set",
                         height: int = 340) -> Optional[go.Figure]:
    """Two-panel comparison of the generated set: QED vs cLogP and MW distribution."""
    if not rows:
        return None
    qed = [r.get("qed") for r in rows]
    clogp = [r.get("clogp") for r in rows]
    mw = [r.get("mw") for r in rows]
    if all(v is None for v in qed):
        return None
    labels = [str(r.get("smiles", ""))[:34] for r in rows]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[v if v is not None else 0 for v in clogp],
            y=[v if v is not None else 0 for v in qed],
            mode="markers", text=labels,
            marker={
                "size": [max(7, min(26, (v or 200) / 18.0)) for v in mw],
                "color": [v if v is not None else 0 for v in qed],
                "colorscale": "Bluered", "showscale": False,
                "line": {"width": 0.6, "color": "rgba(0,0,0,0.4)"},
            },
            hovertemplate="%{text}<br>cLogP %{x:.2f}<br>QED %{y:.3f}<extra></extra>",
            name="molecules",
        )
    )
    fig.add_vrect(x0=-0.4, x1=5.6, fillcolor=PALETTE["accent"], opacity=0.05, line_width=0)
    layout = plotly_layout(height, title)
    layout["xaxis"] = {"title": "cLogP (Crippen)", "gridcolor": PALETTE["border"]}
    layout["yaxis"] = {"title": "QED", "gridcolor": PALETTE["border"]}
    fig.update_layout(**layout)
    fig.add_annotation(x=5.6, y=0.02, text="Lipinski-friendly cLogP band", showarrow=False,
                       font={"size": 10, "color": PALETTE["text_dim"]}, xanchor="right")
    return fig


def screening_scatter(rows: Sequence[Dict[str, Any]], *, height: int = 380) -> Optional[go.Figure]:
    """Pose confidence vs drug-likeness for the ranked screening table."""
    usable = [r for r in rows if r.get("best_confidence") is not None and r.get("qed") is not None]
    if not usable:
        return None
    fig = go.Figure(
        go.Scatter(
            x=[r["best_confidence"] for r in usable],
            y=[r["qed"] for r in usable],
            mode="markers+text",
            text=[f"#{r.get('rank')}" for r in usable],
            textposition="top center",
            marker={
                "size": [max(8, min(28, (r.get("mw") or 300) / 16.0)) for r in usable],
                "color": [r.get("clashes") or 0 for r in usable],
                "colorscale": [[0, PALETTE["good"]], [0.5, PALETTE["warn"]], [1, PALETTE["bad"]]],
                "showscale": True,
                "colorbar": {"title": "clashes", "thickness": 12, "len": 0.7},
                "line": {"width": 0.6, "color": "rgba(0,0,0,0.4)"},
            },
            customdata=[[r.get("smiles", "")[:48], r.get("composite")] for r in usable],
            hovertemplate="%{customdata[0]}<br>confidence %{x:.2f}<br>QED %{y:.2f}"
                          "<br>composite %{customdata[1]}<extra></extra>",
            name="candidates",
        )
    )
    fig.update_layout(**plotly_layout(height, "Pose confidence vs drug-likeness"))
    fig.update_xaxes(title_text="best pose confidence")
    fig.update_yaxes(title_text="QED")
    return fig


def pocket_bars(pockets: Sequence[Dict[str, Any]], *, height: int = 300) -> Optional[go.Figure]:
    """Volume and druggability of each detected cavity."""
    if not pockets:
        return None
    ids = [str(p.get("id", "")) for p in pockets]
    volumes = [float(p.get("volume_a3") or 0) for p in pockets]
    drugg = [float(p.get("druggability_hint") or 0) for p in pockets]
    depth = [float(p.get("depth_a") or 0) for p in pockets]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=ids, y=volumes, name="volume (A^3)", marker_color=CHART_SEQUENCE[0]))
    fig.add_trace(go.Bar(x=ids, y=[d * 100 for d in drugg], name="druggability x100",
                         marker_color=CHART_SEQUENCE[1]))
    fig.add_trace(go.Scatter(x=ids, y=depth, name="max depth (A)", mode="lines+markers",
                             line={"color": CHART_SEQUENCE[3], "width": 2}))
    layout = plotly_layout(height, "Detected cavities")
    layout["barmode"] = "group"
    fig.update_layout(**layout)
    fig.update_yaxes(title_text="value")
    return fig


def descriptor_bars(report: Dict[str, Any], *, height: int = 300,
                    title: str = "Drug-likeness profile") -> Optional[go.Figure]:
    """Normalised descriptor bars for a single molecule, with the ideal windows marked."""
    if not report:
        return None
    # Each tuple: (label, value, ideal low, ideal high) -- the windows are the
    # conventional medicinal-chemistry ranges, shown so a bar can be judged.
    specs = [
        ("QED", report.get("qed"), 0.5, 1.0),
        ("1 - SA/10", 1 - (report.get("sa_score") or 0) / 10.0, 0.4, 1.0),
        ("cLogP", report.get("clogp"), 0.0, 5.0),
        ("TPSA/140", (report.get("tpsa") or 0) / 140.0, 0.0, 1.0),
        ("MW/500", (report.get("mw") or 0) / 500.0, 0.0, 1.0),
        ("Fsp3", report.get("fraction_csp3"), 0.25, 1.0),
    ]
    labels, values, colours = [], [], []
    for label, value, low, high in specs:
        if value is None:
            continue
        labels.append(label)
        values.append(round(float(value), 3))
        good = (value >= low) if high >= low else True
        if label == "1 - SA/10":
            good = value >= 0.5
        elif label == "MW/500":
            # MW reported raw: compare inside the window
            mw = report.get("mw") or 0
            good = 150 <= mw <= 500
        elif label == "TPSA/140":
            tpsa = report.get("tpsa") or 0
            good = tpsa <= 140
        elif label == "cLogP":
            good = -0.4 <= (report.get("clogp") or 0) <= 5.6
        elif label == "QED":
            good = value >= 0.5
        colours.append(PALETTE["good"] if good else PALETTE["warn"])
    if not labels:
        return None
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker_color=colours,
                           hovertemplate="%{y}: %{x}<extra></extra>"))
    fig.update_layout(**plotly_layout(height, title))
    fig.update_xaxes(range=[0, max(1.0, max(values) * 1.15)])
    return fig


def radar(rows: Sequence[Dict[str, Any]], *, height: int = 380) -> Optional[go.Figure]:
    """Radar of normalised descriptors for up to 5 molecules (seed vs analogues)."""
    usable = [r for r in rows if r.get("qed") is not None][:5]
    if len(usable) < 2:
        return None
    axes = ["QED", "cLogP suitability", "TPSA suitability", "MW suitability", "SA (inverted)", "Fsp3"]
    fig = go.Figure()
    for index, row in enumerate(usable):
        clogp = row.get("clogp") or 0
        tpsa = row.get("tpsa") or 0
        mw = row.get("mw") or 0
        values = [
            row.get("qed") or 0,
            max(0.0, 1 - abs(clogp - 2.5) / 3.5),
            max(0.0, 1 - abs(tpsa - 75) / 90),
            max(0.0, 1 - abs(mw - 350) / 300),
            1 - (row.get("sa_score") or 5) / 10,
            row.get("fraction_csp3") or 0,
        ]
        fig.add_trace(
            go.Scatterpolar(
                r=values + [values[0]], theta=axes + [axes[0]], fill="toself", opacity=0.55,
                name=str(row.get("smiles", f"molecule {index + 1}"))[:28],
                hovertemplate="%{theta}: %{r:.2f}<extra></extra>",
            )
        )
    layout = plotly_layout(height, "Descriptor radar")
    layout["polar"] = {"bgcolor": "rgba(0,0,0,0)",
                       "radialaxis": {"visible": True, "range": [0, 1], "gridcolor": PALETTE["border"]},
                       "angularaxis": {"gridcolor": PALETTE["border"]}}
    fig.update_layout(**layout)
    return fig
