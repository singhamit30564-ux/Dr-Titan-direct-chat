"""The non-chat pages: Structure Lab, Molecule Workbench, Screening, Administration, History.

The chat handles one-off requests. These pages are for the times a scientist
wants a form with every knob visible, a table they can sort, and a file they can
download -- the same engines, driven directly.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

from ..bio import chem as chem_mod
from ..bio import depict
from ..bio import sequence as seq_mod
from ..bio import structure as st_mod
from ..engine import export as export_mod
from ..engine import tools as tools_mod
from ..nims import catalog, local_runtime
from ..ui import cards, plots, viewer3d
from ..ui.theme import PALETTE, badge_html, card_html, empty_state, metric_html


# --------------------------------------------------------------------------
# Structure Lab
# --------------------------------------------------------------------------


def structure_lab(service: Any, store: Any, backend: str) -> None:
    st.markdown("## Structure Lab")
    st.caption(
        "Fold a sequence with a BioNeMo NIM (or the built-in model), then measure it: cavities, "
        "SASA, secondary structure, confidence and contacts. Everything here is the same engine the "
        "chat uses, with the knobs exposed."
    )

    tab_fold, tab_measure, tab_compare = st.tabs(["Fold & inspect", "Measure a PDB", "Compare two"])

    with tab_fold:
        left, right = st.columns([3, 2])
        with left:
            sequence = st.text_area(
                "Amino-acid sequence or FASTA",
                value=st.session_state.get("lab_sequence", ""),
                height=150,
                placeholder="MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPD…",
                key="lab_sequence_input",
            )
            fasta_file = st.file_uploader("…or load a FASTA", type=["fasta", "fa", "txt"],
                                           key="lab_fasta")
            if fasta_file is not None:
                records = seq_mod.parse_fasta(fasta_file.read().decode("utf-8", "replace"))
                if records:
                    sequence = records[0]["sequence"]
                    st.caption(f"Loaded {records[0]['id'] or 'sequence'} ({len(sequence)} residues)")
            st.session_state["lab_sequence"] = sequence
            if sequence:
                clean, warnings = seq_mod.clean_sequence(sequence)
                detected = seq_mod.detect_sequence_type(clean)
                st.markdown(
                    " ".join([badge_html(f"{len(clean)} residues", "accent", dot=False),
                              badge_html(detected or "unknown", "info", dot=False)])
                    + (f' <span class="bm-card-meta">{" ".join(warnings)}</span>' if warnings else ""),
                    unsafe_allow_html=True,
                )
        with right:
            model_options = ["esmfold", "alphafold2", "openfold3", "boltz2"]
            model = st.selectbox("Folder", model_options,
                                 format_func=lambda k: f"{catalog.get(k).name} ({catalog.get(k).vendor})",
                                 key="lab_model")
            spec = catalog.get(model)
            st.markdown(
                f'<div class="bm-card-meta">{spec.description}</div>',
                unsafe_allow_html=True,
            )
            with st.expander("Model parameters"):
                params = catalog.param_defaults(spec)
                for key, value in params.items():
                    if isinstance(value, bool):
                        params[key] = st.checkbox(key, value=value, key=f"lab_{model}_{key}")
                    elif isinstance(value, (int, float)) and not isinstance(value, bool):
                        params[key] = st.number_input(key, value=value, key=f"lab_{model}_{key}")
                    else:
                        params[key] = st.text_input(key, value=str(value), key=f"lab_{model}_{key}")
            with st.expander("Backend"):
                st.write(f"Requested: **{backend}**")
                st.caption(
                    "Set the default on the Administration page. Auto tries a local NIM, then NVIDIA's "
                    "hosted endpoint, then the built-in model, and reports which one answered."
                )

        if st.button("Fold", type="primary", disabled=not sequence, key="lab_fold_btn"):
            with st.spinner(f"Calling {model}…"):
                outcome = tools_mod.execute("fold_protein", {"sequence": sequence, "model": model},
                                            service, backend=backend)
            st.session_state["lab_outcome"] = outcome
            st.session_state["lab_key"] = f"lab_{int(time.time())}"
        outcome = st.session_state.get("lab_outcome")
        if outcome is not None:
            cards.render_outcome(outcome, key=st.session_state.get("lab_key", "lab"))

    with tab_measure:
        pdb_text = st.text_area("PDB content", height=170, key="lab_pdb_text",
                                placeholder="Paste ATOM records, or upload below…")
        pdb_file = st.file_uploader("…or upload a PDB", type=["pdb", "ent"], key="lab_pdb_file")
        if pdb_file is not None:
            pdb_text = pdb_file.read().decode("utf-8", "replace")
        find_pockets = st.checkbox("Detect cavities", value=True, key="lab_pockets")
        spacing = st.slider("Cavity lattice spacing (Å)", 0.8, 2.0, 1.2, 0.1, key="lab_spacing",
                            help="Smaller is more accurate and slower.")
        min_volume = st.slider("Minimum cavity volume (Å³)", 40, 400, 120, 10, key="lab_minvol")
        if st.button("Analyse", type="primary", disabled=not pdb_text, key="lab_measure_btn"):
            with st.spinner("Measuring…"):
                try:
                    structure = st_mod.parse_pdb(pdb_text)
                    pockets = structure.find_pockets(spacing=spacing, min_volume=float(min_volume),
                                                     top=6) if find_pockets else []
                    from ..engine.tools import ToolOutcome

                    st.session_state["lab_outcome"] = ToolOutcome(
                        tool="analyse_structure", runner="local analysis", ok=True,
                        title="Structure analysis", renderer="structure_view",
                        payload={"structure": structure, "pockets": pockets,
                                 "pdb": structure.pdb_text,
                                 "structure_summary": structure.to_dict(profiles=True)},
                        summary=(
                            f"{structure.n_residues} residues; Rg {structure.radius_of_gyration():.1f} Å; "
                            f"{len(pockets)} cavity(ies) at {spacing} Å lattice, "
                            f"min volume {min_volume} Å³."
                        ),
                    )
                    st.session_state["lab_key"] = f"measure_{int(time.time())}"
                except Exception as exc:
                    st.error(f"Could not analyse that structure: {exc}")
        outcome = st.session_state.get("lab_outcome")
        if outcome is not None and outcome.tool == "analyse_structure":
            cards.render_outcome(outcome, key=st.session_state.get("lab_key", "measure"))

    with tab_compare:
        columns = st.columns(2)
        with columns[0]:
            pdb_a = st.text_area("Structure A (PDB)", height=140, key="cmp_a")
        with columns[1]:
            pdb_b = st.text_area("Structure B (PDB)", height=140, key="cmp_b")
        if st.button("Compare", type="primary", disabled=not (pdb_a and pdb_b), key="cmp_btn"):
            from ..engine.tools import ToolOutcome

            with st.spinner("Superposing…"):
                try:
                    a, b = st_mod.parse_pdb(pdb_a), st_mod.parse_pdb(pdb_b)
                    rmsd = a.rmsd(b)
                    overlap = a.contact_map_overlap(b)
                    st.session_state["cmp_outcome"] = ToolOutcome(
                        tool="compare_structures", runner="local analysis", ok=True,
                        title="Structure comparison", renderer="comparison",
                        payload={"rmsd": rmsd, "contact_map_overlap": overlap,
                                 "n_residues": [a.n_residues, b.n_residues],
                                 "structure_a": a, "structure_b": b,
                                 "pdb_a": a.pdb_text, "pdb_b": b.pdb_text},
                        summary=(f"CA RMSD {rmsd:.2f} Å" if rmsd is not None else "No overlap") +
                                (f"; contact-map overlap {100 * overlap:.0f}%." if overlap is not None else "."),
                    )
                except Exception as exc:
                    st.error(f"Comparison failed: {exc}")
        outcome = st.session_state.get("cmp_outcome")
        if outcome is not None:
            cards.render_outcome(outcome, key="cmp")


# --------------------------------------------------------------------------
# Molecule Workbench
# --------------------------------------------------------------------------


def molecule_workbench(service: Any, store: Any, backend: str) -> None:
    st.markdown("## Molecule Workbench")
    st.caption(
        "Generate analogues with MolMIM/GenMol, then judge them locally: descriptors, drug-likeness "
        "rules, structural alerts, scaffolds, similarity and clustering. Generation is a model; the "
        "judging is exact chemistry."
    )

    left, right = st.columns([3, 2])
    with left:
        seed = st.text_input("Seed molecule (SMILES)",
                             value=st.session_state.get("wb_seed", "CC(=O)Oc1ccccc1C(=O)O"),
                             key="wb_seed_input")
        st.session_state["wb_seed"] = seed
        if seed:
            st.markdown(depict.smiles_svg(seed, width=340, height=210, title="seed"),
                        unsafe_allow_html=True)
    with right:
        model = st.selectbox("Generator", ["molmim", "genmol"],
                             format_func=lambda k: catalog.get(k).name, key="wb_model")
        num_molecules = st.slider("Molecules", 4, 60, 12, key="wb_n")
        objective = st.selectbox("Objective", ["QED", "plogP"], key="wb_obj")
        minimize = st.checkbox("Minimise the objective", value=False, key="wb_min")
        min_similarity = st.slider("Minimum similarity to seed", 0.0, 1.0, 0.3, 0.05, key="wb_sim")
        st.caption(catalog.get(model).description)

    if st.button("Generate & analyse", type="primary", disabled=not seed, key="wb_run"):
        progress = st.progress(0.0, text="Calling the generator…")
        outcome = tools_mod.execute(
            "generate_molecules",
            {"seed_smiles": seed, "num_molecules": num_molecules, "property_name": objective,
             "minimize": minimize, "min_similarity": min_similarity, "model": model},
            service, backend=backend,
        )
        progress.progress(0.7, text="Judging the molecules locally…")
        molecules = [m for m in (outcome.run.canonical.get("molecules") if outcome.run else []) or []
                     if m.get("smiles")]
        reports = [chem_mod.report_molecule(m["smiles"]).to_dict() for m in molecules]
        reports = [r for r in reports if r.get("canonical_smiles")]
        if reports:
            smiles_list = [r["canonical_smiles"] for r in reports]
            outcome.payload.update({
                "reports": reports,
                "set_summary": chem_mod.summarise_set(smiles_list),
                "clusters": chem_mod.cluster_molecules(smiles_list, threshold=0.4) if len(smiles_list) > 2 else [],
                "scaffolds": chem_mod.scaffold_table(smiles_list),
                "seed_smiles": seed,
            })
            labels, matrix = chem_mod.similarity_matrix(smiles_list)
            outcome.payload.update({"similarity_labels": labels, "similarity_matrix": matrix})
        progress.progress(1.0, text="done")
        progress.empty()
        st.session_state["wb_outcome"] = outcome
        st.session_state["wb_key"] = f"wb_{int(time.time())}"

    outcome = st.session_state.get("wb_outcome")
    if outcome is not None:
        cards.render_outcome(outcome, key=st.session_state.get("wb_key", "wb"))
    else:
        st.markdown(
            empty_state("Nothing generated yet.",
                        "Pick a seed molecule and press Generate — aspirin above is a good start."),
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("### Analyse a molecule set")
    st.caption("Paste SMILES (one per line or comma separated) to get descriptors, alerts and clustering "
               "without calling any model.")
    pasted = st.text_area("SMILES list", height=120, key="wb_paste",
                          placeholder="CCO\nCC(=O)Oc1ccccc1C(=O)O\nCN1C=NC2=C1C(=O)N(C(=O)N2C)C")
    if st.button("Analyse set", disabled=not pasted, key="wb_analyse"):
        with st.spinner("Computing descriptors…"):
            outcome = tools_mod.execute("analyse_molecules", {"smiles_list": pasted}, service,
                                        backend=backend)
        st.session_state["wb_set_outcome"] = outcome
    set_outcome = st.session_state.get("wb_set_outcome")
    if set_outcome is not None:
        cards.render_outcome(set_outcome, key="wb_set")


# --------------------------------------------------------------------------
# Screening pipeline
# --------------------------------------------------------------------------


def screening_page(service: Any, store: Any, queue: Any, backend: str, conversation_id: str) -> None:
    st.markdown("## Screening Pipeline")
    st.caption(
        "The generative virtual-screening chain, end to end: fold the target → detect cavities → "
        "generate analogues of a seed → dock every analogue → rank. This is the shape of NVIDIA's "
        "Generative Virtual Screening blueprint, run against whichever backends you have configured. "
        "It runs in the background, so you can keep working."
    )

    columns = st.columns([3, 2])
    with columns[0]:
        target = st.text_area("Target: sequence or PDB", height=150, key="scr_target",
                              value=st.session_state.get("scr_target_default", ""),
                              placeholder="Paste a sequence, or a PDB starting with ATOM/HETATM…")
        seed = st.text_input("Seed molecule (SMILES)",
                             value=st.session_state.get("scr_seed", "CC(=O)Oc1ccccc1C(=O)O"),
                             key="scr_seed_input")
        st.session_state["scr_seed"] = seed
    with columns[1]:
        num_molecules = st.slider("Analogues to generate", 4, 40, 8, key="scr_n")
        num_poses = st.slider("Poses per analogue", 1, 8, 3, key="scr_poses")
        objective = st.selectbox("Generation objective", ["QED", "plogP"], key="scr_obj")
        folder = st.selectbox("Folder for the target", ["esmfold", "alphafold2", "openfold3"],
                              key="scr_folder")
        st.markdown(
            f'<div class="bm-card-meta">Backend: <b>{backend}</b>. The fold, generation and docking '
            "stages each report their own backend in the provenance.</div>",
            unsafe_allow_html=True,
        )

    if st.button("Run pipeline", type="primary", disabled=not (target and seed), key="scr_run"):
        job_id = queue.submit(
            "screening",
            "Virtual screening",
            lambda ctx: _run_screening_job(ctx, service, target=target, seed=seed,
                                           num_molecules=num_molecules, num_poses=num_poses,
                                           objective=objective, folder=folder, backend=backend),
            params={"target_length": len(target), "seed": seed, "n": num_molecules},
            conversation_id=conversation_id,
        )
        st.session_state["scr_job"] = job_id
        st.success(f"Queued as job `{job_id}`. Progress appears below and in the chat's job panel.")

    job_id = st.session_state.get("scr_job")
    if job_id:
        record = queue.get(job_id)
        if record is not None:
            if record.record.is_active:
                st.progress(min(1.0, record.record.progress),
                            text=f"{record.record.status}: {record.record.message}")
                if st.button("Cancel", key="scr_cancel"):
                    queue.cancel(job_id)
                    st.rerun()
            else:
                st.markdown(
                    badge_html(record.record.status.upper(),
                               "good" if record.record.status == "done" else "bad"),
                    unsafe_allow_html=True,
                )
                if record.record.status == "done" and record.result is not None:
                    payload = record.result
                    outcome = payload.outcomes[0] if getattr(payload, "outcomes", None) else None
                    if outcome is not None:
                        cards.render_outcome(outcome, key=f"scr_{job_id}")
                elif record.record.error:
                    st.error(record.record.error)

    st.divider()
    st.markdown("### How the ranking is computed")
    st.markdown(
        "- **Fold** — the target comes from a folder NIM when the input is a sequence; a pasted PDB is used as-is.\n"
        "- **Cavities** — the console's lattice-based detector finds enclosed pockets (volume, enclosure, "
        "depth, druggability hint).\n"
        "- **Generate** — MolMIM/GenMol proposes analogues of the seed, optimising QED or plogP with a "
        "similarity floor.\n"
        "- **Dock** — every analogue is docked (10 poses by default per molecule in the full pipeline) "
        "and scored on steric quality, contact density, hydrophobic contact fraction, H-bonds and burial.\n"
        "- **Rank** — composite = 0.50 x pose confidence + 0.30 x QED + 0.15 x clash cleanliness + "
        "0.05 x alert-free. The CSV holds every column, so you can re-rank with your own weights."
    )


def _run_screening_job(job, service, *, target: str, seed: str, num_molecules: int, num_poses: int,
                       objective: str, folder: str, backend: str):
    """Background body for the screening page (mirrors the chat's pipeline tool)."""
    from ..engine.tools import ToolOutcome

    job.progress(0.05, "folding the target")
    outcome = tools_mod.execute(
        "virtual_screening",
        {"protein": target, "seed_smiles": seed, "num_molecules": num_molecules,
         "num_poses": num_poses, "property_name": objective},
        service, backend=backend,
        progress=lambda fraction, message: job.progress(0.05 + 0.9 * fraction, message),
    )
    job.progress(1.0, "done")

    class _Payload:
        def __init__(self):
            self.text = outcome.summary
            self.narration_source = "offline"
            self.outcomes = [outcome]
            self.kind = "result"

        def result_summary(self):
            return {"tools": [outcome.tool], "ok": outcome.ok, "text": outcome.summary[:600]}

    return _Payload()


# --------------------------------------------------------------------------
# Administration
# --------------------------------------------------------------------------


def administration(service: Any, store: Any, settings: Dict[str, Any], queue: Any) -> None:
    st.markdown("## Administration")
    st.caption(
        "What this machine can actually run, how the console is configured, and the commands that "
        "close any gap it finds. Nothing here is guessed: the environment probe runs on this host."
    )

    tabs = st.tabs(["Environment", "Backends", "NIM containers", "Language model",
                    "Performance", "Raw settings"])

    with tabs[0]:
        env = service.refresh_environment()
        columns = st.columns(4)
        with columns[0]:
            gpus = env["gpus"]
            st.markdown(metric_html("GPUs", str(gpus["count"]),
                                    (gpus["names"][0] if gpus["names"] else "none detected")),
                        unsafe_allow_html=True)
        with columns[1]:
            st.markdown(metric_html("Container runtime",
                                    "Docker" if env["containers"]["docker"] else
                                    ("Podman" if env["containers"]["podman"] else "none"),
                                    "nvidia toolkit " + ("present" if env["containers"]["nvidia_runtime"] else "missing")),
                        unsafe_allow_html=True)
        with columns[2]:
            st.markdown(metric_html("PyTorch", env["torch"] or "not installed",
                                    "transformers " + (env["transformers"] or "-")),
                        unsafe_allow_html=True)
        with columns[3]:
            st.markdown(metric_html("NGC key", "set" if env["ngc_key_present"] else "missing",
                                    "NGC_API_KEY / NVIDIA_API_KEY"), unsafe_allow_html=True)

        st.markdown("### What that means here")
        for tip in env["advice"]:
            st.markdown(f'<div class="bm-warning">{tip}</div>', unsafe_allow_html=True)

        with st.expander("Full environment record"):
            st.json(env, expanded=False)

    with tabs[1]:
        st.markdown("### Requested backend")
        options = ["auto", "local", "hosted", "framework", "simulate"]
        current = settings.get("prefer_backend", "auto")
        chosen = st.radio(
            "Default backend", options, index=options.index(current) if current in options else 0,
            format_func=lambda k: {
                "auto": "Auto — prefer real services, fall back only when they fail",
                "local": "Local NIM only (http://localhost:8000)",
                "hosted": "NVIDIA hosted NIM only (needs an API key)",
                "framework": "Local PyTorch/BioNeMo runtime",
                "simulate": "Built-in simulation only",
            }[k],
            key="admin_backend",
        )
        if chosen != current:
            store.set_setting("prefer_backend", chosen)
            st.success(f"Default backend set to **{chosen}**. Reload the page to apply.")

        st.markdown("### Endpoints")
        st.caption(
            "One base URL per model, so several NIM containers on different ports can be used at once. "
            "Leave a field empty to fall through to the hosted endpoint."
        )
        endpoints: Dict[str, str] = {}
        columns = st.columns(2)
        for index, spec in enumerate(catalog.all_models()):
            if spec.local_only and not spec.local_path:
                continue
            with columns[index % 2]:
                endpoints[spec.key] = st.text_input(
                    f"{spec.name} — {spec.vendor}",
                    value=service.client.local_endpoints.get(spec.key, "")
                    or (service.client.default_local if spec.key in ("esmfold", "molmim", "diffdock",
                                                                     "proteinmpnn", "esm2") else ""),
                    key=f"ep_{spec.key}",
                    placeholder="http://localhost:8000",
                )
        if st.button("Save endpoints", key="admin_save_eps"):
            service.client.local_endpoints.update({k: v for k, v in endpoints.items() if v})
            store.set_setting("local_endpoints", service.client.local_endpoints)
            st.success("Endpoints saved.")

        st.markdown("### Connectivity test")
        if st.button("Probe every configured endpoint", key="admin_probe"):
            rows = []
            for spec in catalog.all_models():
                base = service.client.base_url(spec, "local")
                if not base:
                    continue
                probe = service.client.probe(spec, "local")
                rows.append({"model": spec.key, "url": probe.url or base,
                             "reachable": probe.reachable, "ready": probe.ready,
                             "status": probe.status, "detail": probe.detail[:90]})
            if rows:
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            else:
                st.info("No local endpoints are configured, so there was nothing to probe.")
            hosted = [{"model": s.key, "url": service.client.url_for(s, "hosted")}
                      for s in catalog.all_models() if s.hosted_path and not s.local_only]
            if hosted:
                st.markdown("**Hosted endpoints these models would use** (not contacted)")
                st.dataframe(pd.DataFrame(hosted), width="stretch", hide_index=True)
        st.caption(
            "A probe asks `GET /v1/health/ready`, then `/v1/models` and `/v1/version`. See the next tab "
            "for the exact commands to start a NIM."
        )

    with tabs[2]:
        st.markdown("### Start a NIM locally")
        st.caption(
            "These are the real commands: they pull NVIDIA's containers from `nvcr.io`. Each needs a "
            "GPU visible to the container runtime and an NGC API key (free at build.nvidia.com)."
        )
        if not service.environment().get("can_run_nim_containers"):
            st.markdown(
                '<div class="bm-warning">This host cannot run NIM containers '
                f"(GPU: {'yes' if service.environment()['gpu_available'] else 'no'}, "
                f"toolkit: {'yes' if service.environment()['containers']['nvidia_runtime'] else 'no'}). "
                "Run these on the machine that has the GPU.</div>",
                unsafe_allow_html=True,
            )
        for entry in local_runtime.nim_run_commands():
            with st.expander(f"{entry['title']} — {entry['note']}"):
                st.code(entry["command"], language="bash")
                st.markdown("Verify it is up:")
                st.code(entry["verify"], language="bash")
                st.markdown(f"Register it here as `{entry['register']}`.")
        st.markdown("### Registry helpers")
        for entry in local_runtime.registry_commands():
            with st.expander(entry["title"]):
                st.code(entry["command"], language="bash")
                st.caption(entry["note"])

    with tabs[3]:
        st.markdown("### Language model (optional)")
        st.caption(
            "The console works with no language model: the router chooses tool calls deterministically "
            "and the deterministic narrator writes the result summaries. A model only improves the "
            "prose, and is never allowed to invent numbers."
        )
        llm_status = service.settings  # placeholder to keep the variable used
        client = st.session_state.get("_llm_client")
        if client is not None:
            status = client.status(refresh=True)
            st.markdown(
                badge_html("CONNECTED" if status.available else "OFFLINE",
                           "good" if status.available else "warn")
                + f' <span class="bm-card-meta">{status.message}</span>',
                unsafe_allow_html=True,
            )
            if status.models:
                st.caption("Models advertised by that endpoint:")
                st.code(", ".join(status.models[:30]))
        provider = st.selectbox(
            "Provider",
            ["off", "auto", "local", "ollama", "nvidia", "openai"],
            index=["off", "auto", "local", "ollama", "nvidia", "openai"].index(
                settings.get("llm_provider", "off") if settings.get("llm_provider", "off") != "off" else "off"),
            key="admin_llm_provider",
            help="'local' means any OpenAI-compatible server: vLLM, llama.cpp, LM Studio, a NIM LLM container.",
        )
        model = st.text_input("Model name", value=settings.get("llm_model", ""),
                              placeholder="meta/llama-3.1-8b-instruct, qwen2.5-7b-instruct, llama3.1:8b…",
                              key="admin_llm_model")
        base_url = st.text_input("Base URL (for 'local')",
                                 value=settings.get("llm_base_url", "http://localhost:8000/v1"),
                                 key="admin_llm_url")
        api_key = st.text_input("API key", value=settings.get("llm_api_key", ""), type="password",
                                key="admin_llm_key",
                                help="Stored in the local SQLite database only. Prefer the "
                                     "NVIDIA_API_KEY environment variable.")
        temperature = st.slider("Temperature", 0.0, 1.0, float(settings.get("llm_temperature", 0.2)),
                                0.05, key="admin_llm_temp")
        if st.button("Save and test", key="admin_llm_save"):
            store.set_settings({
                "llm_provider": provider, "llm_model": model, "llm_base_url": base_url,
                "llm_api_key": api_key, "llm_temperature": temperature,
            })
            st.success("Saved. Reload the page to connect.")
        st.caption(
            "Environment variables are read first: `NVIDIA_API_KEY` enables the hosted provider, "
            "`LOCAL_LLM_BASE_URL` points at a local server, `OPENAI_API_KEY` enables OpenAI."
        )

    with tabs[4]:
        st.markdown("### Recorded service calls")
        summary = store.benchmark_summary()
        if summary:
            st.dataframe(pd.DataFrame(summary), width="stretch", hide_index=True)
        else:
            st.markdown(empty_state("No service calls recorded yet.",
                                    "Run a model from the chat or a workbench to populate this."),
                        unsafe_allow_html=True)
        log = store.call_log(limit=120)
        if log:
            frame = pd.DataFrame(log)
            frame["when"] = pd.to_datetime(frame["created_at"], unit="s")
            st.dataframe(
                frame[["when", "model_key", "backend", "ok", "status", "latency_ms", "attempts", "url", "error"]],
                width="stretch", hide_index=True,
            )
        st.markdown("### Jobs")
        jobs = queue.list()
        if jobs:
            frame = pd.DataFrame([
                {"id": j["id"], "kind": j["kind"], "title": j["title"], "status": j["status"],
                 "progress": j["progress"], "message": j["message"], "duration_s": j["duration_s"],
                 "error": j["error"]}
                for j in jobs
            ])
            st.dataframe(frame, width="stretch", hide_index=True)
            if st.button("Clear finished jobs", key="admin_clear_jobs"):
                queue.clear_finished()
                st.rerun()
        else:
            st.caption("No jobs have been submitted in this process.")
        st.markdown("### Storage")
        st.json(store.stats(), expanded=False)

    with tabs[5]:
        st.markdown("### Effective settings")
        st.json(settings, expanded=False)
        st.caption(
            "Stored in `.runtime/console.db`. Environment variables (`NGC_API_KEY`, "
            "`BIONEMO_CONSOLE_HOME`, `LOCAL_LLM_BASE_URL`) take precedence over anything set here."
        )


# --------------------------------------------------------------------------
# History
# --------------------------------------------------------------------------


def history_page(store: Any) -> None:
    st.markdown("## History")
    st.caption(
        "Every run is recorded with the full provenance manifest: model, backend, URL, parameters, "
        "warnings and timing. This is what makes a result reproducible months later."
    )
    tabs = st.tabs(["Runs", "Artifacts", "Conversations", "Database"])

    with tabs[0]:
        runs = store.runs(limit=300)
        if not runs:
            st.markdown(empty_state("No runs recorded yet."), unsafe_allow_html=True)
        else:
            frame = pd.DataFrame([
                {
                    "when": pd.to_datetime(r["created_at"], unit="s"),
                    "model": r["model_key"], "backend": r["backend"],
                    "simulated": r["simulated"], "ok": r["ok"],
                    "latency_ms": round(r["latency_ms"], 1),
                    "warnings": len(r.get("warnings") or []),
                    "summary": (r.get("summary") or "")[:120],
                    "id": r["id"],
                }
                for r in runs
            ])
            st.dataframe(frame, width="stretch", hide_index=True)
            selected = st.selectbox("Inspect a run", [r["id"] for r in runs],
                                    format_func=lambda rid: next(
                                        (f"{r['model_key']} · {r['backend']} · {r['id']}"
                                         for r in runs if r["id"] == rid), rid),
                                    key="hist_run")
            run = store.run(selected) if selected else None
            if run:
                left, right = st.columns([3, 2])
                with left:
                    st.markdown("**Provenance manifest**")
                    st.json(run.get("provenance") or {}, expanded=True)
                with right:
                    st.markdown("**Summary**")
                    st.write(run.get("summary") or "(none)")
                    if run.get("warnings"):
                        st.markdown("**Warnings**")
                        for warning in run["warnings"]:
                            st.markdown(f'<div class="bm-warning">{warning}</div>', unsafe_allow_html=True)
                    if run.get("error"):
                        st.error(run["error"])
                    st.download_button(
                        "Download run JSON",
                        data=json.dumps(run, indent=2, default=str),
                        file_name=f"run_{run['id']}.json", key="hist_dl",
                        width="stretch",
                    )

    with tabs[1]:
        artifacts = store.artifacts(limit=300)
        if not artifacts:
            st.markdown(empty_state("No artifacts yet.",
                                    "Exports are written when you use a result's Export tab."),
                        unsafe_allow_html=True)
        else:
            st.dataframe(pd.DataFrame(artifacts), width="stretch", hide_index=True)
            for record in artifacts[:40]:
                path = record["path"]
                try:
                    data = open(path, "rb").read()
                except OSError:
                    continue
                st.download_button(f"{record['name']} ({record['size']} B)", data=data,
                                   file_name=record["name"], key=f"art_{record['id']}")

    with tabs[2]:
        conversations = store.list_conversations(limit=100)
        if not conversations:
            st.markdown(empty_state("No conversations yet."), unsafe_allow_html=True)
        else:
            st.dataframe(pd.DataFrame([
                {"title": c["title"], "messages": c["n_messages"],
                 "updated": pd.to_datetime(c["updated_at"], unit="s"), "id": c["id"]}
                for c in conversations
            ]), width="stretch", hide_index=True)
            selected = st.selectbox("Export a conversation",
                                    [c["id"] for c in conversations],
                                    format_func=lambda cid: next(
                                        (c["title"] for c in conversations if c["id"] == cid), cid),
                                    key="hist_conv")
            if selected:
                data = store.export_conversation(selected)
                messages = data.get("messages") or []
                markdown = export_mod.conversation_markdown(data.get("conversation") or {}, messages)
                columns = st.columns(2)
                with columns[0]:
                    st.download_button("conversation.md", data=markdown.encode("utf-8"),
                                       file_name=f"conversation_{selected}.md", key="hist_md",
                                       width="stretch")
                with columns[1]:
                    st.download_button("conversation.json",
                                       data=json.dumps(data, indent=2, default=str).encode("utf-8"),
                                       file_name=f"conversation_{selected}.json", key="hist_json",
                                       width="stretch")
                with st.expander("Preview the Markdown export"):
                    st.markdown(markdown[:6000])

    with tabs[3]:
        stats = store.stats()
        columns = st.columns(4)
        with columns[0]:
            st.markdown(metric_html("Conversations", str(stats["counts"]["conversations"])),
                        unsafe_allow_html=True)
        with columns[1]:
            st.markdown(metric_html("Runs", str(stats["counts"]["runs"]),
                                    f'{stats["real_runs"]} real · {stats["simulated_runs"]} simulated'),
                        unsafe_allow_html=True)
        with columns[2]:
            st.markdown(metric_html("Artifacts", str(stats["counts"]["artifacts"])), unsafe_allow_html=True)
        with columns[3]:
            st.markdown(metric_html("Database", f"{stats['db_size_kb']} KB", stats["db_path"]),
                        unsafe_allow_html=True)
