"""Renderers: one per tool outcome type, each with View / Data / Export / Provenance.

Every card follows the same shape so the app feels consistent:

* a headline strip of metrics that answers "what happened",
* the visual (3D structure, molecule grid, pose, plot),
* a data table,
* downloads,
* a provenance block that states exactly which backend produced the numbers and
  whether they were simulated.

The provenance block is not decoration: it is the guarantee that nothing in the
UI can present a geometric model as if it were a NIM prediction.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import streamlit as st

from ..bio import chem as chem_mod
from ..bio import depict
from ..bio import sequence as seq_mod
from ..bio import structure as st_mod
from ..engine import export as export_mod
from . import plots, viewer3d
from .theme import PALETTE, SS_COLOURS, badge_html, card_html, metric_html


# --------------------------------------------------------------------------
# Shared pieces
# --------------------------------------------------------------------------


BACKEND_TONES = {
    "local": ("LOCAL NIM", "good"),
    "hosted": ("NVIDIA HOSTED", "good"),
    "framework": ("LOCAL RUNTIME", "violet"),
    "simulate": ("SIMULATED", "warn"),
    "local analysis": ("LOCAL ANALYSIS", "accent"),
}


def backend_badges(outcome: Any) -> str:
    """Status chips: which backend answered, which model, how long it took.

    Works from the live run object when there is one and from the recorded
    provenance otherwise, so a card re-rendered after a reload is labelled exactly
    as it was when it first ran. A simulated result must never re-appear as if a
    real service had answered it.
    """
    run = getattr(outcome, "run", None)
    provenance = getattr(outcome, "provenance", None) or {}
    chips: List[str] = []
    if run is not None:
        label, tone = run.badge()
        chips.append(badge_html(label, tone))
        chips.append(badge_html(run.spec_key, "accent", dot=False))
        if run.latency_ms:
            chips.append(badge_html(f"{run.latency_ms / 1000:.1f}s", "info", dot=False))
    else:
        backend = provenance.get("backend", "local analysis")
        label, tone = BACKEND_TONES.get(backend, (backend.upper(), "info"))
        chips.append(badge_html(label, tone))
        model = provenance.get("model_key")
        if model and model != outcome.tool:
            chips.append(badge_html(str(model), "accent", dot=False))
        latency = provenance.get("latency_ms")
        if latency:
            chips.append(badge_html(f"{float(latency) / 1000:.1f}s", "info", dot=False))
    chips.append(badge_html(f"{outcome.elapsed_s:.1f}s total", "info", dot=False))
    return " ".join(chips)


def warnings_block(outcome: Any) -> None:
    run = getattr(outcome, "run", None)
    warnings = list(getattr(run, "warnings", []) or [])
    if outcome.error:
        st.error(outcome.error, icon="⛔")
    for warning in warnings:
        st.markdown(f'<div class="bm-warning">{warning}</div>', unsafe_allow_html=True)


def provenance_block(outcome: Any, *, run_id: str = "") -> None:
    """The audit trail: request, backend, method, environment, replay command."""
    run = getattr(outcome, "run", None)
    if run is None:
        st.markdown(
            card_html(
                "Provenance",
                f'<div class="bm-summary">Computed locally by the console\'s own analysis code '
                f'(<span class="bm-mono">{outcome.runner}</span>) - no model call was made, so no '
                f"prediction error applies. Structural geometry and descriptors are exact.</div>",
            ),
            unsafe_allow_html=True,
        )
        return
    manifest = run.provenance()
    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.markdown("**Reproducibility manifest**")
        st.json(manifest, expanded=False)
    with col_b:
        st.markdown("**Replay this call yourself**")
        curl = export_mod.nim_request_curl(outcome)
        if curl:
            st.code(curl, language="bash")
            st.caption(
                "Point the URL at your own NIM container (default `http://localhost:8000`) and set "
                "`NGC_API_KEY` for the hosted service. The response shape is identical to what this "
                "card displays."
            )
        else:
            st.caption("This run was not an HTTP call.")
        if run.headers:
            st.markdown("**Response headers**")
            st.json(run.headers, expanded=False)


def downloads_block(outcome: Any, *, run_id: str = "") -> None:
    """Write artifacts for this outcome and offer them as downloads."""
    files = export_mod.outcome_files(outcome, run_id=run_id)
    if not files:
        st.caption("This result has no file export.")
        return
    columns = st.columns(min(4, len(files)))
    for index, record in enumerate(files):
        path = record["path"]
        try:
            data = open(path, "rb").read()
        except OSError:
            continue
        with columns[index % len(columns)]:
            st.download_button(
                f"{record['name']}",
                data=data,
                file_name=record["name"],
                key=f"dl_{outcome.tool}_{run_id}_{record['name']}",
                width="stretch",
                mime="text/plain",
            )
    st.caption(
        "Files are also kept under `.runtime/artifacts/` and listed on the History page, so a result "
        "can be retrieved after the session ends."
    )


# --------------------------------------------------------------------------
# Structure card
# --------------------------------------------------------------------------


def render_structure(outcome: Any, *, key: str = "") -> None:
    payload = outcome.payload or {}
    structure: Optional[st_mod.Structure] = payload.get("structure")
    pdb = payload.get("pdb") or ""
    pockets = payload.get("pockets") or (getattr(outcome.run, "meta", {}) or {}).get("pockets") or []

    if structure is None and pdb:
        try:
            structure = st_mod.parse_pdb(pdb)
        except Exception:
            structure = None

    if structure is None:
        warnings_block(outcome)
        st.info("No structure to display.")
        return

    plddt = structure.plddt()
    mean_plddt = float(np.mean(plddt)) if plddt is not None and len(plddt) else 0.0
    ss = structure.secondary_structure()
    ss_counts = {c: ss.count(c) for c in "HGEBTC"}
    label_note = "simulated model" if structure.simulated else "returned model"

    cols = st.columns(5)
    with cols[0]:
        st.markdown(metric_html("Residues", f"{structure.n_residues}",
                                f"{len(structure.chains())} chain(s)"), unsafe_allow_html=True)
    with cols[1]:
        st.markdown(metric_html("Radius of gyration", f"{structure.radius_of_gyration():.1f} Å",
                                f"{label_note}"), unsafe_allow_html=True)
    with cols[2]:
        confidence_label = "Mean confidence" if not structure.simulated else "Mean confidence*"
        st.markdown(metric_html(confidence_label, f"{mean_plddt:.1f}",
                                "pLDDT scale 0-100"), unsafe_allow_html=True)
    with cols[3]:
        helical = ss_counts.get("H", 0) + ss_counts.get("G", 0)
        st.markdown(metric_html("Helical", f"{helical}",
                                f"{100 * helical / max(1, len(ss)):.0f}% of chain"), unsafe_allow_html=True)
    with cols[4]:
        st.markdown(metric_html("Cavities", f"{len(pockets)}",
                                "detected" if pockets else "none found"), unsafe_allow_html=True)

    tabs = st.tabs(["3D view", "Cavities", "Confidence", "Sequence", "Export", "Provenance"])
    with tabs[0]:
        site_pdb = ""
        if pockets:
            pass
        viewer3d.show_viewer(
            pdb=structure.pdb_text,
            pocket_probes=[(p["centre"], max(1.2, (float(p.get("volume_a3") or 300) / 6.0) ** (1 / 3)),
                            p.get("id", "")) for p in pockets[:3]],
            height=470,
            caption=f"{structure.n_residues} residues · {'simulated' if structure.simulated else 'model'}",
            legend=[("chain", PALETTE["text_dim"]), ("helix", SS_COLOURS["H"]),
                    ("strand", SS_COLOURS["E"]), ("cavity", PALETTE["warn"])],
            style="cartoon",
        )
        if structure.simulated:
            st.caption(
                "*This structure was built by the console's own folding model (torsional Monte Carlo "
                "on a residue-level statistical potential, ideal backbone geometry). It is a physical "
                "model, not a learned prediction: treat the topology as a hypothesis and the geometry "
                "as approximate."
            )
        st.caption(
            "Controls: representation (cartoon / cartoon + binding-site sticks / sticks / "
            "space-filling / lines), colouring by secondary structure, chain, N→C rainbow, "
            "confidence (B-factor) or hydrophobicity, plus spin, PNG export and fullscreen."
        )

    with tabs[1]:
        if not pockets:
            st.markdown(
                '<div class="bm-empty">No cavity passed the enclosure and volume thresholds. '
                "A very small or very open protein legitimately has none.</div>",
                unsafe_allow_html=True,
            )
        else:
            figure = plots.pocket_bars(pockets)
            if figure:
                st.plotly_chart(figure, width="stretch", key=f"pocketbars_{key}")
            table = pd.DataFrame([
                {
                    "cavity": p.get("id"), "volume (Å³)": p.get("volume_a3"),
                    "max enclosure": p.get("max_enclosure"), "mean enclosure": p.get("mean_enclosure"),
                    "max depth (Å)": p.get("max_burial_depth"),
                    "burial fraction": p.get("ray_burial_fraction"),
                    "druggability": p.get("druggability_hint"),
                    "enclosed fraction": p.get("fully_enclosed_fraction"),
                    "lining residues": ", ".join(p.get("lining_residues", [])[:12]),
                    "centre (x,y,z)": ", ".join(str(round(v, 1)) for v in p.get("centre", [])),
                }
                for p in pockets
            ])
            st.dataframe(table, width="stretch", hide_index=True)
            st.caption(
                "Druggability hint blends enclosure, depth, burial and lining-residue count onto 0-1. "
                "It is a heuristic for triage, not a validated trained score. Depth is capped at 25 Å "
                "because the lattice is finite."
            )

    with tabs[2]:
        if plddt is not None and len(plddt):
            figure = plots.profile_track(
                plddt, title="Per-residue confidence", y_title="confidence",
                window=1, height=260, colour=PALETTE["accent"],
            )
            if figure:
                st.plotly_chart(figure, width="stretch", key=f"plddt_{key}")
            low = int((plddt < 70).sum())
            very_low = int((plddt < 50).sum())
            st.caption(
                f"{low} residue(s) below 70 and {very_low} below 50. Regions under 70 are the ones "
                "to treat cautiously - usually loops, termini or a wrong domain packing."
            )
        track = plots.ss_track(ss, predicted=str((getattr(outcome.run, "meta", {}) or {}).get("predicted_ss", "")),
                              height=170)
        if track:
            st.plotly_chart(track, width="stretch", key=f"sstrack_{key}")
        figure = plots.contact_map(structure.contact_map(), ss=ss, height=430)
        if figure:
            st.plotly_chart(figure, width="stretch", key=f"cmap_{key}")
            st.caption(
                "Long-range contacts (residues far apart in sequence but close in space) define the "
                "fold. Dense off-diagonal blocks are the domain interfaces."
            )

    with tabs[3]:
        sequence = structure.sequence()
        st.code(_wrap_sequence(sequence), language="text")
        report = seq_mod.analyse_protein(sequence)
        cols = st.columns(4)
        with cols[0]:
            st.markdown(metric_html("Length", f"{report.length}", "residues"), unsafe_allow_html=True)
        with cols[1]:
            st.markdown(metric_html("Mass", f"{report.molecular_weight / 1000:.1f} kDa",
                                    f"{report.molecular_weight:.0f} Da"), unsafe_allow_html=True)
        with cols[2]:
            st.markdown(metric_html("pI", f"{report.isoelectric_point:.2f}",
                                    f"charge {report.charge_pH7:+.1f} at pH 7"), unsafe_allow_html=True)
        with cols[3]:
            st.markdown(metric_html("GRAVY", f"{report.gravy:+.3f}",
                                    f"instability {report.instability_index:.0f}"), unsafe_allow_html=True)
        if report.motifs:
            st.markdown("**Motif hits**")
            st.dataframe(pd.DataFrame(report.motifs), width="stretch", hide_index=True)

    with tabs[4]:
        downloads_block(outcome, run_id=key)

    with tabs[5]:
        provenance_block(outcome, run_id=key)


def _wrap_sequence(sequence: str, width: int = 60) -> str:
    return "\n".join(sequence[i:i + width] for i in range(0, len(sequence), width))


# --------------------------------------------------------------------------
# Molecule card
# --------------------------------------------------------------------------


def render_molecules(outcome: Any, *, key: str = "") -> None:
    payload = outcome.payload or {}
    reports: List[Dict[str, Any]] = payload.get("reports") or []
    set_summary = payload.get("set_summary") or (getattr(outcome.run, "meta", {}) or {}).get("set_summary") or {}
    rows = (getattr(outcome.run, "meta", {}) or {}).get("rows") or []

    if not reports and rows:
        # A NIM result can arrive before local descriptors are attached.
        reports = [chem_mod.report_molecule(r["smiles"]).to_dict() for r in rows if r.get("smiles")]
        set_summary = chem_mod.summarise_set([r["canonical_smiles"] for r in reports]) if reports else {}

    if not reports:
        warnings_block(outcome)
        st.info("No molecules to show.")
        return

    alerts = sum(
        1 for r in reports for h in (r.get("filter_alerts") or []) if h.get("severity") == "alert"
    )
    notes = sum(
        1 for r in reports for h in (r.get("filter_alerts") or []) if h.get("severity") in ("note", "warn")
    )
    passing = sum(1 for r in reports if r.get("ro5_pass"))
    cols = st.columns(5)
    with cols[0]:
        st.markdown(metric_html("Molecules", f"{len(reports)}", "valid & embedded"), unsafe_allow_html=True)
    with cols[1]:
        st.markdown(metric_html("Mean QED", f"{set_summary.get('mean_qed', 0):.3f}",
                                "0-1, higher is better"), unsafe_allow_html=True)
    with cols[2]:
        st.markdown(metric_html("Mean cLogP", f"{set_summary.get('mean_clogp', 0):.2f}",
                                "Crippen"), unsafe_allow_html=True)
    with cols[3]:
        st.markdown(metric_html("Lipinski pass", f"{passing}/{len(reports)}",
                                f"Veber {int(100 * set_summary.get('veber_pass_rate', 0))}%"),
                    unsafe_allow_html=True)
    with cols[4]:
        st.markdown(metric_html("PAINS alerts", f"{alerts}",
                                f"{notes} notes · {set_summary.get('unique_scaffolds', 0)} scaffolds"),
                    unsafe_allow_html=True)

    tabs = st.tabs(["Structures", "Properties", "Set analysis", "Export", "Provenance"])

    with tabs[0]:
        _molecule_grid(reports)
        st.caption(
            "Depictions are generated by the console's own SVG renderer from RDKit's 2D layout - "
            "no Cairo or PIL needed, so the app runs on a bare server."
        )

    with tabs[1]:
        table = pd.DataFrame(reports)
        keep = [c for c in [
            "smiles", "name", "mw", "exact_mass", "clogp", "tpsa", "hbd", "hba",
            "rotatable_bonds", "rings", "aromatic_rings", "fraction_csp3", "qed", "sa_score",
            "predicted_solubility_logs", "violations", "ro5_pass", "veber_pass",
            "brenk_or_pains", "inchi_key",
        ] if c in table.columns]
        st.dataframe(table[keep], width="stretch", hide_index=True)
        st.caption(
            "QED is the Bickerton drug-likeness estimate; SA score is Ertl's synthetic accessibility "
            "(1 easy - 10 very hard); cLogP/TPSA/HBD/HBA are RDKit/Crippen implementations. Those "
            "descriptors are exact chemistry, whichever backend generated the molecules."
        )
        with st.expander("Alerts and flags per molecule"):
            flagged = [r for r in reports if r.get("filter_alerts") or r.get("ro5_violations")]
            if not flagged:
                st.write("No structural alerts or rule violations.")
            for report in flagged[:20]:
                st.markdown(
                    f"**`{report['canonical_smiles'][:70]}`**  \n"
                    f"alerts: {', '.join(report.get('filter_alerts') or ['none'])}  \n"
                    f"rule violations: {', '.join(report.get('ro5_violations') or ['none'])}"
                )

    with tabs[2]:
        left, right = st.columns(2)
        with left:
            figure = plots.molecule_set_summary(
                [{**r, "smiles": r.get("canonical_smiles", "")} for r in reports])
            if figure:
                st.plotly_chart(figure, width="stretch", key=f"molsum_{key}")
        with right:
            figure = plots.radar([{**r, "smiles": r.get("canonical_smiles", "")} for r in reports])
            if figure:
                st.plotly_chart(figure, width="stretch", key=f"radar_{key}")
        clusters = payload.get("clusters") or []
        if clusters:
            st.markdown("**Similarity clusters (Tanimoto, ECFP4)**")
            st.dataframe(pd.DataFrame([
                {
                    "cluster": c.get("cluster"),
                    "size": c.get("size"),
                    "mean intra-similarity": c.get("mean_similarity"),
                    "representative": (c.get("representative") or "")[:60],
                    "scaffolds": c.get("unique_scaffolds"),
                }
                for c in clusters
            ]), width="stretch", hide_index=True)
        if payload.get("scaffolds"):
            st.markdown("**Murcko scaffolds**")
            st.dataframe(pd.DataFrame(payload["scaffolds"]), width="stretch", hide_index=True)
        if payload.get("similarity_matrix"):
            labels = payload.get("similarity_labels") or []
            matrix = np.asarray(payload["similarity_matrix"], dtype=float)
            import plotly.graph_objects as go

            figure = go.Figure(go.Heatmap(
                z=matrix, x=labels, y=labels, zmin=0, zmax=1,
                colorscale=[[0, PALETTE["bg"]], [0.5, "#1d4ed8"], [1, PALETTE["accent"]]],
                colorbar={"title": "Tanimoto", "thickness": 12},
                hovertemplate="%{y} - %{x}<br>Tanimoto %{z:.2f}<extra></extra>",
            ))
            layout = plots.plotly_layout(420, "Pairwise similarity")
            figure.update_layout(**layout)
            st.plotly_chart(figure, width="stretch", key=f"simm_{key}")

    with tabs[3]:
        downloads_block(outcome, run_id=key)

    with tabs[4]:
        provenance_block(outcome, run_id=key)


def _molecule_grid(reports: Sequence[Dict[str, Any]], columns: int = 4) -> None:
    """Responsive grid of 2D depictions with key properties under each."""
    for start in range(0, len(reports), columns):
        row = reports[start:start + columns]
        cols = st.columns(columns)
        for column, report in zip(cols, row):
            smiles = report.get("canonical_smiles") or report.get("smiles") or ""
            with column:
                st.markdown(
                    depict.smiles_svg(
                        smiles,
                        width=300,
                        height=200,
                        subtitle=f"QED {report.get('qed', 0):.2f}  ·  cLogP "
                                 f"{report.get('clogp', 0):.2f}  ·  MW {report.get('mw', 0):.0f}",
                    ),
                    unsafe_allow_html=True,
                )
                flags = []
                if report.get("brenk_or_pains"):
                    flags.append("PAINS")
                if not report.get("ro5_pass"):
                    flags.append("Ro5 fail")
                if not report.get("veber_pass"):
                    flags.append("Veber fail")
                st.caption(f"`{smiles[:44]}`" + (f"  ·  {', '.join(flags)}" if flags else ""))


# --------------------------------------------------------------------------
# Docking card
# --------------------------------------------------------------------------


def render_poses(outcome: Any, *, key: str = "") -> None:
    payload = outcome.payload or {}
    run_meta = getattr(outcome.run, "meta", {}) or {}
    poses = payload.get("poses") or run_meta.get("poses") or []
    pockets = payload.get("pockets") or run_meta.get("pockets") or []
    protein_pdb = payload.get("pdb") or run_meta.get("protein_pdb") or ""
    ligand_smiles = payload.get("ligand") or run_meta.get("ligand_smiles") or ""

    if not poses:
        warnings_block(outcome)
        st.info("No poses were produced.")
        return

    best = poses[0]
    cols = st.columns(5)
    with cols[0]:
        st.markdown(metric_html("Poses", f"{len(poses)}",
                                f"diversity {run_meta.get('pose_diversity_mean_rmsd', 0):.1f} Å RMSD"),
                    unsafe_allow_html=True)
    with cols[1]:
        st.markdown(metric_html("Best confidence", f"{best.confidence:.2f}",
                                f"pose #{best.rank}"), unsafe_allow_html=True)
    with cols[2]:
        st.markdown(metric_html("H-bonds", f"{best.hbonds}",
                                f"{best.contacts} heavy-atom contacts"), unsafe_allow_html=True)
    with cols[3]:
        st.markdown(metric_html("Buried", f"{100 * best.buried_fraction:.0f}%",
                                "of ligand atoms"), unsafe_allow_html=True)
    with cols[4]:
        tone = "clean" if best.clashes == 0 else f"{best.clashes} overlap(s)"
        st.markdown(metric_html("Steric quality", tone, "vdW overlap > 0.55 Å"),
                    unsafe_allow_html=True)

    tabs = st.tabs(["Pose viewer", "Scores", "Interactions", "Export", "Provenance"])

    with tabs[0]:
        options = [f"#{p.rank}  confidence {p.confidence:.2f}  ({p.pocket_id or 'surface'})" for p in poses]
        selection = st.radio("Pose", options, index=0, horizontal=False, key=f"pose_sel_{key}",
                             label_visibility="collapsed")
        index = options.index(selection)
        if protein_pdb and ligand_smiles:
            viewer3d.show_pose_viewer(protein_pdb, ligand_smiles, poses, selected=poses[index].rank,
                                      pockets=pockets, height=480)
        else:
            st.info("Pose coordinates are available in the Scores tab; no receptor was supplied for 3D display.")
        if pockets:
            st.caption(
                "Cavity probes are drawn at the detected cavity centres. Only the single best cavity is "
                "shown per pose in the caption; dock into a specific cavity with the Workbench."
            )

    with tabs[1]:
        table = pd.DataFrame([p.to_dict() for p in poses])
        st.dataframe(table, width="stretch", hide_index=True)
        figure = plots.screening_scatter([
            {"smiles": ligand_smiles or f"pose {p.rank}", "best_confidence": p.confidence,
             "qed": None, "mw": None, "clashes": p.clashes, "rank": p.rank,
             "composite": p.contacts_per_atom}
            for p in poses
        ]) if False else None
        import plotly.graph_objects as go

        figure = go.Figure()
        figure.add_trace(go.Scatter(
            x=[p.rank for p in poses], y=[p.confidence for p in poses], mode="lines+markers",
            name="confidence", line={"color": PALETTE["accent"], "width": 2},
        ))
        figure.add_trace(go.Scatter(
            x=[p.rank for p in poses], y=[p.score for p in poses], mode="lines+markers",
            name="raw score", yaxis="y2", line={"color": PALETTE["info"], "width": 1.6,
                                                "dash": "dot"},
        ))
        layout = plots.plotly_layout(320, "Pose quality")
        layout["yaxis2"] = {"overlaying": "y", "side": "right", "showgrid": False,
                            "title": "raw score"}
        figure.update_layout(**layout)
        figure.update_xaxes(title_text="pose rank")
        st.plotly_chart(figure, width="stretch", key=f"posechart_{key}")
        st.caption(
            "Confidence is a logistic mapping of the raw score (steric penalty, contact density, "
            "hydrophobic fraction, H-bonds, burial) - monotone in pose quality, and attenuated by any "
            "remaining clashes."
        )

    with tabs[2]:
        for pose in poses[:6]:
            if pose.interaction_residues:
                st.markdown(
                    f"**Pose #{pose.rank}** - {pose.contacts} contacts across "
                    f"{len(pose.interaction_residues)} residues"
                    + (f", H-bonds to {', '.join(pose.hbond_residues[:6])}" if pose.hbond_residues else "")
                )
                st.markdown(
                    " ".join(f"`{r}`" for r in pose.interaction_residues[:20]), unsafe_allow_html=True
                )
        if not any(p.interaction_residues for p in poses):
            st.info("No residue-level contacts were recorded for these poses.")

    with tabs[3]:
        downloads_block(outcome, run_id=key)

    with tabs[4]:
        provenance_block(outcome, run_id=key)


# --------------------------------------------------------------------------
# Embeddings card
# --------------------------------------------------------------------------


def render_embeddings(outcome: Any, *, key: str = "") -> None:
    run = getattr(outcome, "run", None)
    meta = (getattr(run, "meta", {}) or {})
    canonical = getattr(run, "canonical", {}) or {}
    quality = meta.get("quality") or {}
    projection = meta.get("projection")
    sequence = meta.get("sequence") or ""
    embeddings = canonical.get("embeddings")

    if not projection and embeddings:
        from ..bio import embed as embed_mod

        wrapped = embed_mod.simulated_embeddings(sequence[:len(embeddings)]) if sequence else None
        if wrapped:
            meta = {**meta, **wrapped}
            projection = meta.get("projection")
            quality = meta.get("quality") or quality

    if not projection:
        warnings_block(outcome)
        st.info("No embedding matrix was returned.")
        return

    cols = st.columns(5)
    with cols[0]:
        st.markdown(metric_html("Residues", f"{quality.get('n_residues', len(projection))}",
                                "vectors"), unsafe_allow_html=True)
    with cols[1]:
        st.markdown(metric_html("Dimension", f"{quality.get('dim', '?')}",
                                "per-residue"), unsafe_allow_html=True)
    with cols[2]:
        st.markdown(metric_html("Mean cosine", f"{quality.get('mean_pairwise_cosine', 0):.3f}",
                                "pairwise similarity"), unsafe_allow_html=True)
    with cols[3]:
        st.markdown(metric_html("PC1+PC2", f"{100 * quality.get('explained_variance_2d', 0):.0f}%",
                                "variance retained"), unsafe_allow_html=True)
    with cols[4]:
        st.markdown(metric_html("Clusters", f"{len(meta.get('clusters') or [])}",
                                "k-means, k chosen by silhouette" if meta.get("clusters") else ""),
                    unsafe_allow_html=True)

    tabs = st.tabs(["Projection", "Profiles", "Neighbours", "Export", "Provenance"])
    with tabs[0]:
        highlight = []
        clusters = meta.get("clusters") or []
        if clusters:
            largest = max(clusters, key=lambda c: c.get("size", 0))
            highlight = largest.get("members", [])
        figure = plots.embedding_projection(
            projection,
            labels=[f"{i + 1}" for i in range(len(projection))],
            colour_values=[float(np.linalg.norm(np.asarray(v))) for v in projection],
            colour_label="vector norm",
            highlight=highlight,
            title="Per-residue embedding projection (PCA of the returned vectors)",
        )
        if figure:
            st.plotly_chart(figure, width="stretch", key=f"proj_{key}")
        if clusters:
            st.markdown("**Clusters**")
            st.dataframe(pd.DataFrame([
                {"cluster": c.get("cluster"), "size": c.get("size"),
                 "members": ", ".join(str(m + 1) for m in (c.get("members") or [])[:24]),
                 "silhouette": c.get("silhouette"),
                 "residue mix": c.get("residue_mix")}
                for c in clusters
            ]), width="stretch", hide_index=True)
        if getattr(run, "simulated", False) and not sequence:
            st.caption(
                "These vectors come from the simulated backend: a fixed random projection of "
                "physicochemical residue features. Useful for plumbing, clustering demos and UI "
                "development, but they are not ESM-2 embeddings and carry no evolutionary information."
            )

    with tabs[1]:
        if sequence and sequence[0].isalpha():
            try:
                report = seq_mod.analyse_protein(sequence[:len(projection)])
                figure = plots.profile_track(
                    report.hydropathy_profile, title="Hydropathy (Kyte-Doolittle, windowed)",
                    y_title="GRAVY", window=1, height=250,
                    labels=[f"{i + 1}:{aa}" for i, aa in enumerate(sequence)],
                )
                if figure:
                    st.plotly_chart(figure, width="stretch", key=f"hyd_{key}")
                figure = plots.profile_track(
                    report.flexibility_profile, title="Backbone flexibility (Vihinen)",
                    y_title="flexibility", height=230, colour=PALETTE["info"],
                )
                if figure:
                    st.plotly_chart(figure, width="stretch", key=f"flex_{key}")
            except Exception as exc:
                st.caption(f"Could not compute sequence profiles: {exc}")

    with tabs[2]:
        neighbours = meta.get("nearest_neighbours") or {}
        if isinstance(neighbours, dict) and neighbours:
            index = st.number_input("Residue", min_value=1, max_value=len(projection), value=1,
                                    key=f"nn_{key}")
            close = neighbours.get(index) or neighbours.get(index - 1)
            if close:
                st.dataframe(pd.DataFrame(close), width="stretch", hide_index=True)
        else:
            st.caption("Nearest-neighbour analysis is only available for locally computed embeddings.")

    with tabs[3]:
        if embeddings:
            import csv as _csv
            import io as _io

            buffer = _io.StringIO()
            writer = _csv.writer(buffer)
            writer.writerow(["residue", "amino_acid"] + [f"dim_{i}" for i in range(len(embeddings[0]))])
            for i, vector in enumerate(embeddings):
                writer.writerow([i + 1, sequence[i] if i < len(sequence) else ""] + list(vector))
            st.download_button(
                "embeddings.csv", data=buffer.getvalue(),
                file_name="embeddings.csv", key=f"emb_{key}", width="stretch",
            )
        downloads_block(outcome, run_id=key)

    with tabs[4]:
        provenance_block(outcome, run_id=key)


# --------------------------------------------------------------------------
# Sequence / DNA cards
# --------------------------------------------------------------------------


def render_sequence_report(outcome: Any, *, key: str = "") -> None:
    payload = outcome.payload or {}
    report = payload.get("report") or {}
    kind = payload.get("kind") or "protein"

    if not report:
        warnings_block(outcome)
        st.info("Nothing to analyse.")
        return

    if kind == "protein":
        cols = st.columns(5)
        metrics = [
            ("Length", f"{report.get('length')}", "residues"),
            ("Mass", f"{report.get('molecular_weight_da', 0) / 1000:.2f} kDa",
             f"{report.get('molecular_weight_da', 0):.0f} Da"),
            ("pI", f"{report.get('isoelectric_point', 0):.2f}",
             f"net {report.get('net_charge_pH7', 0):+.1f} at pH 7"),
            ("GRAVY", f"{report.get('gravy', 0):+.3f}", "Kyte-Doolittle"),
            ("Instability", f"{report.get('instability_index', 0):.0f}",
             str(report.get("instability_label", ""))),
        ]
        for column, (label, value, note) in zip(cols, metrics):
            with column:
                st.markdown(metric_html(label, value, note), unsafe_allow_html=True)
    else:
        cols = st.columns(5)
        metrics = [
            ("Length", f"{report.get('length')}", "nt"),
            ("GC content", f"{100 * report.get('gc_content', 0):.1f}%", "of all bases"),
            ("Tm (NN)", f"{report.get('tm_nn_c')} °C", "SantaLucia nearest-neighbour"),
            ("Effective codons", f"{report.get('effective_codons')}", "Wright Nc"),
            ("ORFs", f"{report.get('orf_count')}", f"longest {report.get('longest_orf_aa')} aa"),
        ]
        for column, (label, value, note) in zip(cols, metrics):
            with column:
                st.markdown(metric_html(label, value, note), unsafe_allow_html=True)

    tabs = st.tabs(["Report", "Profiles", "Export", "Provenance"])
    with tabs[0]:
        st.json(report, expanded=False)
    with tabs[1]:
        if kind == "protein":
            profiles = report.get("profiles") or {}
            figure = plots.profile_track(
                profiles.get("hydropathy") or report.get("hydropathy_profile") or [],
                title="Hydropathy", y_title="GRAVY", height=250,
                labels=[f"{i + 1}" for i in range(len(profiles.get("hydropathy") or []))],
            )
            if figure:
                st.plotly_chart(figure, width="stretch", key=f"seqhyd_{key}")
            figure = plots.profile_track(
                profiles.get("disorder") or report.get("disorder_profile") or [],
                title="Disorder propensity", y_title="score", height=240,
                colour=PALETTE["warn"],
            )
            if figure:
                st.plotly_chart(figure, width="stretch", key=f"seqdis_{key}")
            if profiles.get("hydrophobic_moment") or report.get("hydrophobic_moment_profile"):
                figure = plots.profile_track(
                    profiles.get("hydrophobic_moment") or report["hydrophobic_moment_profile"],
                    title="Hydrophobic moment (window 11)",
                    y_title="moment", height=240, colour=PALETTE["violet"],
                )
                if figure:
                    st.plotly_chart(figure, width="stretch", key=f"seqhm_{key}")
        else:
            composition = report.get("codon_counts") or report.get("codon_usage") or {}
            if composition:
                st.dataframe(pd.DataFrame(
                    [{"codon": k, "count": v.get("count", v) if isinstance(v, dict) else v}
                     for k, v in composition.items()]
                ), width="stretch", hide_index=True)
    with tabs[2]:
        downloads_block(outcome, run_id=key)
    with tabs[3]:
        provenance_block(outcome, run_id=key)


def render_dna_report(outcome: Any, *, key: str = "") -> None:
    meta = (getattr(outcome.run, "meta", {}) or {})
    canonical = (getattr(outcome.run, "canonical", {}) or {})
    analysis = meta.get("analysis") or {}
    generated = meta.get("generated") or canonical.get("generated") or canonical.get("sequence") or ""
    if not generated and not analysis:
        warnings_block(outcome)
        st.info("No sequence was generated.")
        return
    if analysis:
        cols = st.columns(5)
        metrics = [
            ("Generated", f"{len(generated)} nt", f"prompt {len(meta.get('prompt', ''))} nt"),
            ("GC content", f"{100 * analysis.get('gc_content', 0):.1f}%", "product only"),
            ("Longest ORF", f"{analysis.get('longest_orf_aa')} aa", "six frames"),
            ("ORFs > 30 aa", f"{analysis.get('orf_count')}", "candidate peptides"),
            ("Effective codons", f"{analysis.get('effective_codons')}", "codon bias"),
        ]
        for column, (label, value, note) in zip(cols, metrics):
            with column:
                st.markdown(metric_html(label, value, note), unsafe_allow_html=True)
    tabs = st.tabs(["Sequence", "Analysis", "Export", "Provenance"])
    with tabs[0]:
        st.code(_wrap_sequence(generated, 70), language="text")
        st.caption(f"Prompt (kept as generated context): `{meta.get('prompt', '')[:80]}`")
        if analysis.get("longest_orf_aa"):
            st.markdown(f"**Longest ORF translation** ({analysis.get('longest_orf_aa')} aa)")
            st.code(_wrap_sequence(analysis.get("longest_orf_sequence", ""), 60), language="text")
    with tabs[1]:
        st.json(analysis, expanded=False)
    with tabs[2]:
        downloads_block(outcome, run_id=key)
    with tabs[3]:
        provenance_block(outcome, run_id=key)


# --------------------------------------------------------------------------
# Design / comparison / screening cards
# --------------------------------------------------------------------------


def render_designs(outcome: Any, *, key: str = "") -> None:
    meta = (getattr(outcome.run, "meta", {}) or {})
    designs = outcome.payload.get("designs") or meta.get("designs") or []
    wild_type = meta.get("wild_type") or ""
    if not designs:
        warnings_block(outcome)
        st.info("No designs were produced.")
        return

    best = designs[0]
    cols = st.columns(5)
    with cols[0]:
        st.markdown(metric_html("Designs", f"{len(designs)}",
                                f"temperature {meta.get('temperature', '?')}"), unsafe_allow_html=True)
    with cols[1]:
        st.markdown(metric_html("Best recovery", f"{100 * best.get('recovery', 0):.0f}%",
                                "identical to wild type"), unsafe_allow_html=True)
    with cols[2]:
        st.markdown(metric_html("Mutations", f"{len(best.get('mutations') or [])}",
                                "in the best design"), unsafe_allow_html=True)
    with cols[3]:
        st.markdown(metric_html("Global score", f"{best.get('global_score', 0):.2f}",
                                "model log-likelihood"), unsafe_allow_html=True)
    with cols[4]:
        st.markdown(metric_html("Diversity", f"{_design_diversity(designs):.0f}%",
                                "positions varying across designs"), unsafe_allow_html=True)

    tabs = st.tabs(["Sequences", "Mutations", "Export", "Provenance"])
    with tabs[0]:
        frame = pd.DataFrame([
            {
                "design": f"#{i + 1}",
                "score": d.get("score"),
                "global_score": d.get("global_score"),
                "recovery": d.get("recovery"),
                "n_mutations": len(d.get("mutations") or []),
                "sequence": d.get("sequence", ""),
            }
            for i, d in enumerate(designs)
        ])
        st.dataframe(frame, width="stretch", hide_index=True)
        if wild_type:
            st.markdown("**Alignment against the wild type**")
            st.code(_alignment(wild_type, best.get("sequence", "")), language="text")
    with tabs[1]:
        for i, design in enumerate(designs[:8]):
            mutations = design.get("mutations") or []
            if not mutations:
                st.markdown(f"**Design #{i + 1}** - no substitutions")
                continue
            st.markdown(f"**Design #{i + 1}** - {len(mutations)} substitution(s)")
            st.dataframe(pd.DataFrame(mutations), width="stretch", hide_index=True)
    with tabs[2]:
        downloads_block(outcome, run_id=key)
    with tabs[3]:
        provenance_block(outcome, run_id=key)


def _design_diversity(designs: Sequence[Dict[str, Any]]) -> float:
    sequences = [d.get("sequence", "") for d in designs if d.get("sequence")]
    if len(sequences) < 2:
        return 0.0
    length = min(len(s) for s in sequences)
    varying = sum(1 for i in range(length) if len({s[i] for s in sequences}) > 1)
    return 100.0 * varying / max(1, length)


def _alignment(wild_type: str, design: str, width: int = 60) -> str:
    lines = []
    for start in range(0, min(len(wild_type), len(design)), width):
        wt = wild_type[start:start + width]
        ds = design[start:start + width]
        marks = "".join("|" if a == b else " " for a, b in zip(wt, ds))
        lines += [f"WT  {wt}", f"    {marks}", f"DS  {ds}", ""]
    return "\n".join(lines)


def render_comparison(outcome: Any, *, key: str = "") -> None:
    payload = outcome.payload or {}
    rmsd = payload.get("rmsd")
    overlap = payload.get("contact_map_overlap")
    cols = st.columns(4)
    with cols[0]:
        st.markdown(metric_html("CA RMSD", f"{rmsd:.2f} Å" if rmsd is not None else "-",
                                "Kabsch superposition"), unsafe_allow_html=True)
    with cols[1]:
        st.markdown(metric_html("Contact-map overlap",
                                f"{100 * overlap:.1f}%" if overlap is not None else "-",
                                "long-range |i-j| > 4"), unsafe_allow_html=True)
    n_a, n_b = (payload.get("n_residues") or [0, 0])[:2]
    with cols[2]:
        st.markdown(metric_html("Residues", f"{n_a} vs {n_b}", "structures A / B"),
                    unsafe_allow_html=True)
    with cols[3]:
        verdict = ("same fold" if (overlap or 0) > 0.6 and (rmsd or 99) < 3.0 else
                   "related" if (overlap or 0) > 0.35 else "different fold")
        st.markdown(metric_html("Verdict", verdict, "from overlap and RMSD"), unsafe_allow_html=True)
    tabs = st.tabs(["Overlay", "Maps", "Export", "Provenance"])
    with tabs[0]:
        pdb_a = payload.get("pdb_a", "")
        pdb_b = payload.get("pdb_b", "")
        if pdb_a and pdb_b:
            viewer3d.show_viewer(
                pdb=pdb_a,
                ligand_molblocks=None,
                height=430,
                caption="structure A (cartoon) and B (sticks overlay)",
                legend=[("A", PALETTE["accent"]), ("B", PALETTE["info"])],
                style="cartoon",
            )
            st.caption(
                "The viewer shows structure A; load B in the Structure Lab alongside to compare "
                "side by side, or download both files below."
            )
    with tabs[1]:
        structure_a = payload.get("structure_a")
        structure_b = payload.get("structure_b")
        column_a, column_b = st.columns(2)
        with column_a:
            if structure_a is not None:
                figure = plots.contact_map(structure_a.contact_map(),
                                           ss=structure_a.secondary_structure(), title="A",
                                           height=380)
                if figure:
                    st.plotly_chart(figure, width="stretch", key=f"cmp_a_{key}")
        with column_b:
            if structure_b is not None:
                figure = plots.contact_map(structure_b.contact_map(),
                                           ss=structure_b.secondary_structure(), title="B",
                                           height=380)
                if figure:
                    st.plotly_chart(figure, width="stretch", key=f"cmp_b_{key}")
    with tabs[2]:
        downloads_block(outcome, run_id=key)
    with tabs[3]:
        provenance_block(outcome, run_id=key)


def render_screening(outcome: Any, *, key: str = "") -> None:
    payload = outcome.payload or {}
    rows = payload.get("rows") or []
    pockets = payload.get("pockets") or []
    structure = payload.get("structure")
    if not rows:
        warnings_block(outcome)
        st.info("The screening produced no ranked candidates.")
        return

    top = rows[0]
    cols = st.columns(5)
    with cols[0]:
        st.markdown(metric_html("Candidates docked", f"{len(rows)}",
                                f"from {payload.get('n_molecules', '?')} generated"),
                    unsafe_allow_html=True)
    with cols[1]:
        st.markdown(metric_html("Best composite", f"{top.get('composite', 0):.3f}",
                                "see weighting below"), unsafe_allow_html=True)
    with cols[2]:
        st.markdown(metric_html("Best confidence", f"{top.get('best_confidence', 0):.2f}",
                                f"{top.get('hbonds')} H-bond(s)"), unsafe_allow_html=True)
    with cols[3]:
        st.markdown(metric_html("Top QED", f"{top.get('qed', 0):.3f}",
                                f"cLogP {top.get('clogp', 0):.2f}"), unsafe_allow_html=True)
    with cols[4]:
        st.markdown(metric_html("Cavities", f"{len(pockets)}",
                                f"docked into {top.get('pocket', 'surface')}"), unsafe_allow_html=True)

    tabs = st.tabs(["Ranking", "Top candidate", "All molecules", "Export", "Provenance"])
    with tabs[0]:
        frame = pd.DataFrame([
            {
                "rank": r.get("rank"), "SMILES": r.get("smiles"),
                "composite": r.get("composite"), "confidence": r.get("best_confidence"),
                "H-bonds": r.get("hbonds"), "contacts": r.get("contacts"),
                "clashes": r.get("clashes"), "QED": r.get("qed"), "cLogP": r.get("clogp"),
                "MW": r.get("mw"), "alerts": r.get("alerts"), "cavity": r.get("pocket"),
                "similarity to seed": r.get("similarity_to_seed"),
            }
            for r in rows
        ])
        st.dataframe(frame, width="stretch", hide_index=True)
        figure = plots.screening_scatter(rows)
        if figure:
            st.plotly_chart(figure, width="stretch", key=f"screen_{key}")
        st.markdown(
            "**Composite score** = 0.50 x pose confidence + 0.30 x QED + 0.15 x clash cleanliness "
            "+ 0.05 x alert-free. It is printed with every run so you can re-rank the CSV with your own "
            "weights. The docking stage never sees drug-likeness; the weighting is what combines them, "
            "and that split is deliberate."
        )
        failed = payload.get("failed_rows") or []
        if failed:
            with st.expander(f"{len(failed)} candidate(s) failed to dock"):
                st.dataframe(pd.DataFrame(failed), width="stretch", hide_index=True)

    with tabs[1]:
        left, right = st.columns([3, 2])
        with left:
            if structure is not None and top.get("smiles"):
                viewer3d.show_pose_viewer(
                    structure.pdb_text, top["smiles"], _pose_stubs(top), selected=1,
                    pockets=pockets, height=430,
                )
            else:
                st.info("No structure available for a 3D view of the top candidate.")
        with right:
            st.markdown(
                depict.smiles_svg(
                    top.get("smiles", ""), width=360, height=260,
                    subtitle=f"rank 1 · composite {top.get('composite', 0):.3f}",
                ),
                unsafe_allow_html=True,
            )
            report = top.get("descriptor") or {}
            figure = plots.descriptor_bars(report, height=300)
            if figure:
                st.plotly_chart(figure, width="stretch", key=f"topdesc_{key}")
            st.markdown("**Interactions**")
            st.markdown(" ".join(f"`{r}`" for r in (top.get("interaction_residues") or [])[:20]),
                        unsafe_allow_html=True)

    with tabs[2]:
        reports = []
        for r in rows:
            if r.get("descriptor"):
                reports.append({**r["descriptor"], "smiles": r["smiles"]})
        if reports:
            _molecule_grid(reports, columns=4)
        else:
            st.caption("Descriptor data was not retained for this run.")

    with tabs[3]:
        downloads_block(outcome, run_id=key)

    with tabs[4]:
        provenance_block(outcome, run_id=key)


def _pose_stubs(row: Dict[str, Any]):
    """Minimal pose objects for re-viewing a screening candidate's best pose."""
    from ..nims.docking import Pose

    confidences = row.get("pose_confidences") or [row.get("best_confidence") or 0]
    coords = row.get("pose_coords")
    if coords is None:
        return []
    poses = []
    for index, confidence in enumerate(confidences):
        poses.append(Pose(rank=index + 1, coords=np.asarray(coords[index] if isinstance(coords[0], list)
                                                            else coords, dtype=float),
                          confidence=float(confidence), score=float(row.get("best_score") or 0),
                          contacts=int(row.get("contacts") or 0),
                          clashes=int(row.get("clashes") or 0),
                          hbonds=int(row.get("hbonds") or 0),
                          pocket_id=row.get("pocket", ""),
                          interaction_residues=row.get("interaction_residues") or []))
    return poses


# --------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------

RENDERERS = {
    "structure_view": render_structure,
    "molecule_table": render_molecules,
    "pose_table": render_poses,
    "embedding_view": render_embeddings,
    "sequence_report": render_sequence_report,
    "dna_report": render_dna_report,
    "design_table": render_designs,
    "comparison": render_comparison,
    "screening_table": render_screening,
}


def render_outcome(outcome: Any, *, key: str = "", container=None, show_header: bool = True) -> None:
    """Draw one tool outcome as a card.

    ``container`` is an optional Streamlit container; when omitted, elements are
    written to the current context (the common case).
    """
    import contextlib

    target = contextlib.nullcontext() if container is None else container
    with target:
        if show_header:
            meta = f"{outcome.runner}"
            if getattr(outcome, "run", None) is not None and outcome.run.method:
                meta += f" · {outcome.run.method}"
            st.markdown(
                f'<div class="bm-card-head"><span class="bm-card-title">{outcome.title}</span>'
                f'<span>{backend_badges(outcome)}</span></div>'
                f'<div class="bm-card-meta">{meta}</div>',
                unsafe_allow_html=True,
            )
        warnings_block(outcome)
        renderer = RENDERERS.get(outcome.renderer)
        if renderer is None:
            st.markdown(f'<div class="bm-summary">{outcome.summary}</div>', unsafe_allow_html=True)
            st.json(outcome.to_dict(), expanded=False)
            return
        try:
            renderer(outcome, key=key)
        except Exception as exc:  # noqa: BLE001 - a rendering bug must not kill the chat
            st.error(f"This result could not be rendered ({type(exc).__name__}: {exc}).")
            with st.expander("Raw result"):
                st.json(outcome.to_dict(), expanded=False)
