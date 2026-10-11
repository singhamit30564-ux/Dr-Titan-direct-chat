"""The tool registry: every action the console can take, in one place.

Each tool declares

* the NIM (or local analysis routine) it invokes,
* the argument schema -- used both for validation and for LLM function calling,
* the renderer the UI should use for its result.

The router (or a language model) proposes tools; :func:`execute` runs one,
records the run with full provenance in the store, and returns a
:class:`ToolOutcome` that the chat renderer turns into a rich card.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..bio import chem as chem_mod
from ..bio import sequence as seq_mod
from ..bio import structure as st_mod
from ..nims import catalog
from ..nims.service import NimService, RunResult

# Renderers the UI knows how to draw.
RENDERERS = [
    "structure_view",
    "molecule_table",
    "pose_table",
    "embedding_view",
    "sequence_report",
    "dna_report",
    "design_table",
    "comparison",
    "screening_table",
    "text",
]


@dataclass
class ToolSpec:
    name: str
    title: str
    description: str
    model_key: str
    renderer: str
    parameters: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    requires: List[str] = field(default_factory=list)
    category: str = "model"
    examples: List[str] = field(default_factory=list)

    def schema(self) -> Dict[str, Any]:
        """OpenAI-style function schema (used when an LLM plans the call)."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": [k for k, v in self.parameters.items() if v.get("required")],
                },
            },
        }


def _p(type_: str, description: str, *, required: bool = False, enum: Optional[List[str]] = None,
       default: Any = None) -> Dict[str, Any]:
    spec: Dict[str, Any] = {"type": type_, "description": description}
    if enum:
        spec["enum"] = enum
    if default is not None:
        spec["default"] = default
    if required:
        spec["required"] = True
    return spec


TOOLS: Dict[str, ToolSpec] = {}


def _register(spec: ToolSpec) -> ToolSpec:
    TOOLS[spec.name] = spec
    return spec


_register(ToolSpec(
    name="fold_protein",
    title="Predict protein structure",
    description="Fold an amino-acid sequence into a 3D structure with ESMFold, AlphaFold2 or OpenFold3.",
    model_key="esmfold",
    renderer="structure_view",
    parameters={
        "sequence": _p("string", "Amino-acid sequence (single letter) or FASTA.", required=True),
        "model": _p("string", "Which folder to use.", enum=["esmfold", "alphafold2", "openfold3", "boltz2"],
                    default="esmfold"),
    },
    requires=["sequence"],
    examples=["fold MKTAYIAKQRQISFVKSHFSRQLEER", "predict the structure of this sequence with alphafold2"],
))

_register(ToolSpec(
    name="predict_complex",
    title="Co-fold protein + ligand",
    description="Predict a protein-ligand complex (and optionally binding affinity) with Boltz-2 or OpenFold3.",
    model_key="boltz2",
    renderer="structure_view",
    parameters={
        "sequence": _p("string", "Protein sequence.", required=True),
        "ligand_smiles": _p("string", "Ligand SMILES.", required=True),
        "model": _p("string", "Model to use.", enum=["boltz2", "openfold3"], default="boltz2"),
    },
    requires=["sequence", "ligand_smiles"],
    examples=["co-fold this protein with ibuprofen and give the affinity"],
))

_register(ToolSpec(
    name="embed_sequence",
    title="Sequence embeddings",
    description="Compute per-residue protein language-model embeddings (ESM-2 family) for clustering or ML features.",
    model_key="esm2",
    renderer="embedding_view",
    parameters={
        "sequence": _p("string", "Amino-acid sequence.", required=True),
        "model": _p("string", "Embedding model.", enum=["esm2", "amplify", "geneformer"], default="esm2"),
    },
    requires=["sequence"],
    examples=["embed this sequence with esm2", "give me per-residue embeddings"],
))

_register(ToolSpec(
    name="generate_molecules",
    title="Generate analogue molecules",
    description="Generate chemically valid analogues of a seed molecule, optimising a chosen property (MolMIM / GenMol).",
    model_key="molmim",
    renderer="molecule_table",
    parameters={
        "seed_smiles": _p("string", "Seed SMILES.", required=True),
        "num_molecules": _p("integer", "How many molecules to generate.", default=12),
        "property_name": _p("string", "Objective to optimise.", enum=["QED", "plogP"], default="QED"),
        "minimize": _p("boolean", "Minimise instead of maximise the objective.", default=False),
        "min_similarity": _p("number", "Similarity floor to the seed (0-1).", default=0.3),
        "model": _p("string", "Generator to use.", enum=["molmim", "genmol"], default="molmim"),
    },
    requires=["seed_smiles"],
    examples=["generate 20 analogues of CC(=O)Oc1ccccc1C(=O)O optimising QED",
              "make molecules similar to caffeine with plogP maximised"],
))

_register(ToolSpec(
    name="dock_ligand",
    title="Dock a ligand",
    description="Predict binding poses of a small molecule against a protein (DiffDock), with per-pose confidence.",
    model_key="diffdock",
    renderer="pose_table",
    parameters={
        "protein": _p("string", "Protein as PDB text or as an amino-acid sequence.", required=True),
        "ligand": _p("string", "Ligand as SMILES or SDF.", required=True),
        "num_poses": _p("integer", "Number of poses to generate.", default=10),
        "model": _p("string", "Docking model.", enum=["diffdock"], default="diffdock"),
    },
    requires=["protein", "ligand"],
    examples=["dock aspirin into this protein", "run diffdock with 20 poses"],
))

_register(ToolSpec(
    name="design_sequences",
    title="Design sequences (inverse folding)",
    description="Propose new sequences that fold into a given backbone (ProteinMPNN).",
    model_key="proteinmpnn",
    renderer="design_table",
    parameters={
        "pdb": _p("string", "Backbone PDB text.", required=True),
        "num_sequences": _p("integer", "How many sequences to sample.", default=8),
        "sampling_temp": _p("number", "Sampling temperature; lower is more conservative.", default=0.1),
        "fixed_positions": _p("string", "Comma-separated residue numbers to keep fixed.", default=""),
        "model": _p("string", "Design model.", enum=["proteinmpnn", "esm2_generate"], default="proteinmpnn"),
    },
    requires=["pdb"],
    examples=["design 10 sequences for this backbone", "stabilise this protein with proteinmpnn"],
))

_register(ToolSpec(
    name="generate_dna",
    title="Generate / score DNA",
    description="Generate nucleotide sequence with Evo 2 (or score a variant), and analyse the product.",
    model_key="evo2",
    renderer="dna_report",
    parameters={
        "sequence": _p("string", "DNA prompt sequence.", default="ATG"),
        "num_tokens": _p("integer", "Number of nucleotides to generate.", default=100),
        "temperature": _p("number", "Sampling temperature.", default=0.7),
        "model": _p("string", "Genomics model.", enum=["evo2", "codonfm"], default="evo2"),
    },
    requires=[],
    examples=["generate 200 bp starting from ATGAAAGCT", "extend this DNA sequence"],
))

_register(ToolSpec(
    name="analyse_sequence",
    title="Analyse a sequence",
    description="Full property report for a protein or DNA sequence: MW, pI, GRAVY, instability, motifs, ORFs, Tm.",
    model_key="local",
    renderer="sequence_report",
    parameters={
        "sequence": _p("string", "Protein or DNA sequence.", required=True),
        "kind": _p("string", "Force interpretation.", enum=["auto", "protein", "dna"], default="auto"),
    },
    requires=["sequence"],
    category="analysis",
    examples=["analyse MKTAYIAKQRQ", "what is the pI of this protein?", "find ORFs in this DNA"],
))

_register(ToolSpec(
    name="analyse_molecules",
    title="Analyse molecules",
    description="Descriptor table, drug-likeness rules, PAINS/Brenk alerts, clustering and scaffolds for a set of SMILES.",
    model_key="local",
    renderer="molecule_table",
    parameters={
        "smiles_list": _p("string", "SMILES strings, one per line or comma separated.", required=True),
    },
    requires=["smiles_list"],
    category="analysis",
    examples=["analyse these molecules: CCO, c1ccccc1", "cluster these SMILES"],
))

_register(ToolSpec(
    name="analyse_structure",
    title="Analyse a structure",
    description="Secondary structure, SASA, contacts, cavities and quality flags for a PDB.",
    model_key="local",
    renderer="structure_view",
    parameters={
        "pdb": _p("string", "PDB text (or a sequence, which will be folded first).", required=True),
        "find_pockets": _p("boolean", "Run cavity detection.", default=True),
    },
    requires=["pdb"],
    category="analysis",
    examples=["analyse this structure", "find pockets in this PDB"],
))

_register(ToolSpec(
    name="compare_structures",
    title="Compare two structures",
    description="Superpose two structures and report CA RMSD plus contact-map overlap (fold similarity).",
    model_key="local",
    renderer="comparison",
    parameters={
        "pdb_a": _p("string", "First PDB.", required=True),
        "pdb_b": _p("string", "Second PDB.", required=True),
    },
    requires=["pdb_a", "pdb_b"],
    category="analysis",
    examples=["compare these two structures", "rmsd between these models"],
))

_register(ToolSpec(
    name="virtual_screening",
    title="Run the virtual screening pipeline",
    description=(
        "End-to-end generative virtual screening: fold the target, detect cavities, generate "
        "analogues of a seed, dock every analogue, then rank by pose quality and drug-likeness."
    ),
    model_key="pipeline",
    renderer="screening_table",
    parameters={
        "protein": _p("string", "Target as PDB or sequence.", required=True),
        "seed_smiles": _p("string", "Seed molecule to elaborate.", required=True),
        "num_molecules": _p("integer", "How many analogues to generate.", default=8),
        "num_poses": _p("integer", "Docking poses per molecule.", default=3),
        "property_name": _p("string", "Generation objective.", enum=["QED", "plogP"], default="QED"),
        "target_pocket": _p("string", "Restrict docking to one pocket id (empty = best ranked).", default=""),
    },
    requires=["protein", "seed_smiles"],
    category="pipeline",
    examples=["run a virtual screen of aspirin analogues against this protein",
              "virtual screening: target <pdb>, seed CC(=O)Oc1ccccc1C(=O)O"],
))


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------


@dataclass
class ToolOutcome:
    """What a tool produced, ready for rendering, storage and summarising."""

    tool: str
    runner: str
    ok: bool
    title: str
    renderer: str
    run: Optional[RunResult] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    steps: List[Dict[str, Any]] = field(default_factory=list)
    elapsed_s: float = 0.0
    error: str = ""
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    run_id: str = ""
    #: Provenance snapshot, so a card re-rendered after a reload still shows the
    #: correct backend badge and simulated flag.
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool": self.tool,
            "runner": self.runner,
            "ok": self.ok,
            "title": self.title,
            "renderer": self.renderer,
            "summary": self.summary,
            "steps": self.steps,
            "elapsed_s": round(self.elapsed_s, 2),
            "error": self.error,
            "artifacts": self.artifacts,
            "run_id": self.run_id,
            "provenance": self.provenance or (
                self.run.provenance() if self.run is not None
                else {"backend": "local analysis", "simulated": False, "method": self.runner}
            ),
            "payload": {
                k: v for k, v in self.payload.items()
                if isinstance(v, (str, int, float, bool, list, dict, type(None)))
            },
        }


def _clean_list(value: Any) -> List[str]:
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value or "")
    parts = [p.strip() for p in text.replace(";", ",").replace("\n", ",").split(",")]
    return [p for p in parts if p]


def execute(
    tool_name: str,
    args: Dict[str, Any],
    service: NimService,
    *,
    backend: str = "auto",
    context: Optional[Dict[str, Any]] = None,
    progress: Optional[Callable[[float, str], None]] = None,
) -> ToolOutcome:
    """Run one tool and return a renderable outcome."""
    spec = TOOLS.get(tool_name)
    if spec is None:
        return ToolOutcome(tool=tool_name, runner="none", ok=False, title="Unknown tool",
                           renderer="text", error=f"No such tool: {tool_name}")
    context = context or {}
    started = time.time()
    report = progress or (lambda *_a: None)

    try:
        if tool_name == "fold_protein":
            model = args.get("model") or "esmfold"
            report(0.15, f"Calling {model} ...")
            run = service.run(model, {"sequence": args.get("sequence", "")}, backend=backend)
            report(0.9, "post-processing structure")
            return _finish(spec, run, started, payload={"sequence": args.get("sequence", ""), "model": model})

        if tool_name == "predict_complex":
            model = args.get("model") or "boltz2"
            payload = {
                "sequence": args.get("sequence", ""),
                "ligand_smiles": args.get("ligand_smiles", ""),
                "diffusion_samples": int(args.get("diffusion_samples") or 1),
            }
            report(0.2, f"Co-folding with {model} ...")
            run = service.run(model, payload, backend=backend)
            return _finish(spec, run, started, payload=payload)

        if tool_name == "embed_sequence":
            model = args.get("model") or "esm2"
            payload = {"sequence": args.get("sequence", "")}
            report(0.2, f"Embedding with {model} ...")
            run = service.run(model, payload, backend=backend)
            return _finish(spec, run, started, payload=payload)

        if tool_name == "generate_molecules":
            model = args.get("model") or "molmim"
            payload = {
                "smi": args.get("seed_smiles", "") or args.get("smiles", ""),
                "num_molecules": int(args.get("num_molecules") or 12),
                "property_name": (args.get("property_name") or "QED"),
                "minimize": bool(args.get("minimize", False)),
                "min_similarity": float(args.get("min_similarity", 0.3)),
            }
            if model == "genmol":
                payload["smiles"] = payload.pop("smi")
                payload["temperature"] = float(args.get("temperature") or 2.0)
                payload["scoring"] = payload["property_name"]
            report(0.2, f"Generating {payload['num_molecules']} molecules with {model} ...")
            run = service.run(model, payload, backend=backend)
            return _finish(spec, run, started, payload=payload)

        if tool_name == "dock_ligand":
            payload = {
                "protein": args.get("protein", ""),
                "ligand": args.get("ligand", "") or args.get("smiles", ""),
                "num_poses": int(args.get("num_poses") or 10),
            }
            report(0.15, "Detecting cavities ...")
            report(0.3, "Docking poses ...")
            run = service.run("diffdock", payload, backend=backend)
            return _finish(spec, run, started, payload=payload)

        if tool_name == "design_sequences":
            model = args.get("model") or "proteinmpnn"
            payload = {
                "pdb": args.get("pdb", ""),
                "num_sequences": int(args.get("num_sequences") or 8),
                "sampling_temp": float(args.get("sampling_temp") or 0.1),
                "fixed_positions": args.get("fixed_positions", "") or "",
            }
            if model == "proteinmpnn" and not payload["pdb"]:
                source = args.get("structure") or context.get("last_structure_pdb") or ""
                if source:
                    payload["pdb"] = source
            if model == "esm2_generate":
                payload = {"sequence": args.get("pdb", "") or args.get("sequence", ""),
                           "num_tokens": int(args.get("num_sequences") or 100) * 20}
            report(0.25, f"Designing with {model} ...")
            run = service.run(model, payload, backend=backend)
            return _finish(spec, run, started, payload=payload)

        if tool_name == "generate_dna":
            model = args.get("model") or "evo2"
            payload = {
                "sequence": args.get("sequence") or "ATG",
                "num_tokens": int(args.get("num_tokens") or 100),
                "temperature": float(args.get("temperature") or 0.7),
                "enable_sampled_probs": True,
            }
            report(0.2, f"Generating with {model} ...")
            run = service.run(model, payload, backend=backend)
            return _finish(spec, run, started, payload=payload)

        if tool_name == "analyse_sequence":
            sequence = args.get("sequence", "")
            kind = (args.get("kind") or "auto").lower()
            report(0.4, "Computing sequence properties")
            result = _analyse_sequence_local(sequence, kind)
            return ToolOutcome(
                tool=spec.name, runner="local analysis", ok=bool(result.get("report")),
                title=spec.title, renderer=spec.renderer,
                payload={"kind": result.get("type"), "report": result.get("report"),
                         "sequence": result.get("sequence", sequence)[:20000]},
                summary=result.get("summary", ""),
                elapsed_s=time.time() - started,
                error="" if result.get("report") else "Nothing to analyse.",
            )

        if tool_name == "analyse_molecules":
            smiles = _clean_list(args.get("smiles_list") or args.get("smiles") or "")
            report(0.4, f"Analysing {len(smiles)} molecule(s)")
            return _analyse_molecules_local(spec, smiles, started)

        if tool_name == "analyse_structure":
            pdb = args.get("pdb") or context.get("last_structure_pdb") or ""
            report(0.3, "Parsing structure")
            return _analyse_structure_local(spec, pdb, bool(args.get("find_pockets", True)), started)

        if tool_name == "compare_structures":
            report(0.4, "Superposing structures")
            return _compare_structures_local(spec, args.get("pdb_a", ""), args.get("pdb_b", ""), started)

        if tool_name == "virtual_screening":
            return _virtual_screening(spec, args, service, backend=backend, context=context, report=report,
                                      started=started)

        return ToolOutcome(tool=tool_name, runner="none", ok=False, title=spec.title,
                           renderer=spec.renderer, error="Tool registered without an implementation.")
    except Exception as exc:  # noqa: BLE001 - the chat must survive any tool failure
        return ToolOutcome(
            tool=tool_name, runner="error", ok=False, title=spec.title, renderer=spec.renderer,
            error=f"{type(exc).__name__}: {exc}", elapsed_s=time.time() - started,
        )


def _enrich_molecules(run: RunResult, payload: Dict[str, Any]) -> None:
    """Attach local descriptor analysis to a generation result.

    The generator (MolMIM/GenMol, real or simulated) returns SMILES and its own
    score. Everything a chemist judges -- QED, SA, PAINS, scaffolds, clustering --
    is computed here with RDKit, so the same tables and exports exist regardless
    of which backend generated the molecules.
    """
    if not run or not run.canonical:
        return
    molecules = [m.get("smiles") for m in (run.canonical.get("molecules") or []) if m.get("smiles")]
    if not molecules:
        return
    reports = [r.to_dict() for r in (chem_mod.report_molecule(smi) for smi in molecules)
               if r.canonical_smiles]
    if not reports:
        return
    canonical = [r["canonical_smiles"] for r in reports]
    payload.setdefault("reports", reports)
    payload.setdefault("set_summary", chem_mod.summarise_set(canonical))
    payload.setdefault("seed_smiles", payload.get("smi") or payload.get("smiles") or "")
    if len(canonical) > 2:
        payload.setdefault("clusters", chem_mod.cluster_molecules(canonical, threshold=0.4))
    payload.setdefault("scaffolds", chem_mod.scaffold_table(canonical))
    labels, matrix = chem_mod.similarity_matrix(canonical)
    payload.setdefault("similarity_labels", labels)
    payload.setdefault("similarity_matrix", matrix)
    run.meta.setdefault("rows", [
        {"smiles": m.get("smiles"), "score": m.get("score")} for m in (run.canonical.get("molecules") or [])
        if m.get("smiles")
    ])
    run.meta.setdefault("set_summary", payload["set_summary"])


def _finish(spec: ToolSpec, run: RunResult, started: float, *, payload: Dict[str, Any]) -> ToolOutcome:
    from .llm import OfflineAnalyst

    if spec.model_key in ("molmim", "genmol"):
        _enrich_molecules(run, payload)
    if spec.model_key in ("proteinmpnn", "esm2_generate") and run is not None:
        # Inverse folding returns sequences under different keys depending on the
        # backend; the card always reads payload['designs'].
        designs = (run.meta or {}).get("designs") or (run.canonical or {}).get("sequences") or []
        if designs:
            payload.setdefault("designs", designs)
            if run.meta.get("wild_type"):
                payload.setdefault("wild_type", run.meta["wild_type"])
    return ToolOutcome(
        tool=spec.name,
        runner=f"{spec.model_key} via {run.backend}",
        ok=run.ok,
        title=spec.title,
        renderer=spec.renderer,
        run=run,
        payload=payload,
        summary=OfflineAnalyst.summarise_run(run),
        elapsed_s=time.time() - started,
        error=run.error,
        steps=[{"backend": b} for b in run.fallback_chain],
        provenance=run.provenance(),
    )


# --------------------------------------------------------------------------
# Local analysis tools (no model required)
# --------------------------------------------------------------------------


def _analyse_sequence_local(sequence: str, kind: str) -> Dict[str, Any]:
    seq, warnings = seq_mod.clean_sequence(sequence)
    if not seq:
        return {"report": None, "type": "unknown", "summary": "No sequence found in the message."}
    detected = seq_mod.detect_sequence_type(seq) if kind in ("", "auto") else kind
    if detected == "protein":
        rep = seq_mod.analyse_protein(seq)
        data = rep.to_dict(include_profiles=True)
        summary = (
            f"{rep.length} residues, {rep.molecular_weight:,.0f} Da, pI {rep.isoelectric_point:.2f}, "
            f"GRAVY {rep.gravy:+.3f}, instability index {rep.instability_index:.1f} ({rep.instability_label}). "
            f"Secondary-structure propensity {100 * rep.helix_fraction:.0f}% helix / "
            f"{100 * rep.sheet_fraction:.0f}% sheet; {len(rep.motifs)} motif hit(s); "
            f"mean disorder score {rep.mean_disorder:.2f}."
        )
        if warnings:
            summary += " " + " ".join(warnings)
        return {"report": data, "type": "protein", "summary": summary, "sequence": seq}
    if detected in ("dna", "rna"):
        data = seq_mod.analyse_dna(seq)
        summary = (
            f"{data['length']} nt, GC {100 * data['gc_content']:.1f}%, nearest-neighbour Tm "
            f"{data['tm_nn_c']} C, effective codon number {data['effective_codons']}, "
            f"{data['orf_count']} ORF(s) over 30 aa (longest {data['longest_orf_aa']} aa)."
        )
        return {"report": data, "type": detected, "summary": summary, "sequence": seq}
    return {"report": {"warnings": ["Unrecognised alphabet"]}, "type": "unknown",
            "summary": "The string does not look like a protein or nucleic-acid sequence.", "sequence": seq}


def _analyse_molecules_local(spec: ToolSpec, smiles: List[str], started: float) -> ToolOutcome:
    reports = [r.to_dict() for r in (chem_mod.report_molecule(s) for s in smiles) if r.canonical_smiles]
    if not reports:
        return ToolOutcome(tool=spec.name, runner="local analysis", ok=False, title=spec.title,
                           renderer=spec.renderer, error="No valid SMILES found.",
                           elapsed_s=time.time() - started)
    summary_stats = chem_mod.summarise_set([r["canonical_smiles"] for r in reports])
    clusters = chem_mod.cluster_molecules([r["canonical_smiles"] for r in reports], threshold=0.4) if len(reports) > 2 else []
    scaffolds = chem_mod.scaffold_table([r["canonical_smiles"] for r in reports])
    labels, matrix = (chem_mod.similarity_matrix([r["canonical_smiles"] for r in reports])
                      if len(reports) > 1 else ([], []))
    alerts = sum(len(r["filter_alerts"]) for r in reports)
    summary = (
        f"{len(reports)} molecule(s): mean MW {summary_stats['mean_mw']} Da, mean cLogP "
        f"{summary_stats['mean_clogp']}, mean QED {summary_stats['mean_qed']}, mean SA "
        f"{summary_stats['mean_sa']}; {int(100 * summary_stats['ro5_pass_rate'])}% pass Lipinski, "
        f"{int(100 * summary_stats['veber_pass_rate'])}% pass Veber, {alerts} alert(s), "
        f"{summary_stats['unique_scaffolds']} unique scaffold(s)."
    )
    return ToolOutcome(
        tool=spec.name, runner="local analysis", ok=True, title=spec.title, renderer=spec.renderer,
        payload={"reports": reports, "set_summary": summary_stats, "clusters": clusters,
                 "scaffolds": scaffolds, "similarity_labels": labels, "similarity_matrix": matrix},
        summary=summary, elapsed_s=time.time() - started,
    )


def _analyse_structure_local(spec: ToolSpec, pdb: str, find_pockets: bool, started: float) -> ToolOutcome:
    if not (pdb or "").strip():
        return ToolOutcome(tool=spec.name, runner="local analysis", ok=False, title=spec.title,
                           renderer=spec.renderer, error="No structure supplied.",
                           elapsed_s=time.time() - started)
    if "ATOM" in pdb[:4000] or "HETATM" in pdb[:4000]:
        structure = st_mod.parse_pdb(pdb)
    else:
        seq, _ = seq_mod.clean_sequence(pdb)
        structure = st_mod.fold_sequence(seq, seed=17, label="simulated")
    if structure.n_residues == 0:
        return ToolOutcome(tool=spec.name, runner="local analysis", ok=False, title=spec.title,
                           renderer=spec.renderer, error="The structure contained no residues.",
                           elapsed_s=time.time() - started)
    pockets = structure.find_pockets(top=5) if find_pockets else []
    sasa = structure.sasa()["residue_sasa"]
    ss = structure.secondary_structure()
    from collections import Counter

    comp = Counter(ss)
    summary = (
        f"{structure.n_residues} residues in {len(structure.chains())} chain(s); "
        f"Rg {structure.radius_of_gyration():.1f} A; total SASA {sasa.sum():,.0f} A^2; "
        f"SS: {comp.get('H', 0)}H {comp.get('G', 0)}G {comp.get('E', 0)}E {comp.get('C', 0) + comp.get('T', 0)}C/T; "
        f"{len(pockets)} cavity(ies) detected."
    )
    if structure.simulated:
        summary += " (The input was a sequence, so a simulated model was built first.)"
    return ToolOutcome(
        tool=spec.name, runner="local analysis", ok=True, title=spec.title, renderer=spec.renderer,
        payload={
            "structure": structure,
            "structure_summary": structure.to_dict(profiles=True),
            "pockets": pockets,
            "pdb": structure.pdb_text,
        },
        summary=summary, elapsed_s=time.time() - started,
    )


def _compare_structures_local(spec: ToolSpec, pdb_a: str, pdb_b: str, started: float) -> ToolOutcome:
    try:
        a = st_mod.parse_pdb(pdb_a) if "ATOM" in pdb_a else st_mod.fold_sequence(pdb_a[:500])
        b = st_mod.parse_pdb(pdb_b) if "ATOM" in pdb_b else st_mod.fold_sequence(pdb_b[:500])
    except Exception as exc:
        return ToolOutcome(tool=spec.name, runner="local analysis", ok=False, title=spec.title,
                           renderer=spec.renderer, error=str(exc), elapsed_s=time.time() - started)
    rmsd = a.rmsd(b)
    cmo = a.contact_map_overlap(b)
    summary = (
        f"CA RMSD over the {min(a.n_residues, b.n_residues)} overlapping residues: "
        f"{rmsd:.2f} A; contact-map overlap {100 * cmo:.1f}%." if rmsd is not None and cmo is not None
        else "The two structures could not be compared (too few or zero overlapping residues)."
    )
    return ToolOutcome(
        tool=spec.name, runner="local analysis", ok=True, title=spec.title, renderer=spec.renderer,
        payload={
            "rmsd": rmsd,
            "contact_map_overlap": cmo,
            "n_residues": [a.n_residues, b.n_residues],
            "structure_a": a,
            "structure_b": b,
            "pdb_a": a.pdb_text,
            "pdb_b": b.pdb_text,
        },
        summary=summary, elapsed_s=time.time() - started,
    )


# --------------------------------------------------------------------------
# The generative virtual screening blueprint
# --------------------------------------------------------------------------


def _virtual_screening(
    spec: ToolSpec,
    args: Dict[str, Any],
    service: NimService,
    *,
    backend: str,
    context: Dict[str, Any],
    report: Callable[[float, str], None],
    started: float,
) -> ToolOutcome:
    """fold -> cavities -> generate -> dock every analogue -> rank.

    This mirrors NVIDIA's Generative Virtual Screening blueprint (AlphaFold2 ->
    MolMIM -> DiffDock) with the models the user has configured, and keeps every
    intermediate result so the ranking can be audited.
    """
    from ..nims import docking as dock_mod

    protein_input = args.get("protein") or context.get("last_structure_pdb") or ""
    seed_smiles = args.get("seed_smiles") or ""
    n_mol = int(args.get("num_molecules") or 8)
    n_poses = int(args.get("num_poses") or 3)
    property_name = args.get("property_name") or "QED"
    steps: List[Dict[str, Any]] = []

    report(0.05, "Step 1/4: preparing the target")
    if "ATOM" in protein_input[:4000]:
        structure = st_mod.parse_pdb(protein_input, source="user-pdb")
        target_source = "supplied PDB"
        fold_run = None
    else:
        seq, _ = seq_mod.clean_sequence(protein_input)
        if not seq and context.get("last_sequence"):
            seq = context["last_sequence"]
        fold_run = service.run("esmfold", {"sequence": seq[:700]}, backend=backend)
        if not fold_run.ok or "structure" not in fold_run.meta:
            return ToolOutcome(
                tool=spec.name, runner="pipeline", ok=False, title=spec.title,
                renderer=spec.renderer, error=fold_run.error or "Target structure could not be prepared.",
                elapsed_s=time.time() - started,
            )
        structure = fold_run.meta["structure"]
        target_source = f"{fold_run.backend} fold"
        steps.append({"step": "fold", "backend": fold_run.backend, "model": "esmfold",
                      "latency_ms": fold_run.latency_ms, "simulated": fold_run.simulated})

    report(0.2, "Step 2/4: detecting cavities")
    pockets = structure.find_pockets(top=5)
    pockets_r = structure.find_pockets(top=5)
    steps.append({"step": "cavity_detection", "n_pockets": len(pockets_r), "top": pockets_r[0]["id"] if pockets_r else None})

    report(0.35, f"Step 3/4: generating {n_mol} analogues of the seed")
    gen_run = service.run(
        "molmim",
        {"smi": seed_smiles, "num_molecules": max(4, n_mol), "property_name": property_name,
         "min_similarity": 0.25},
        backend=backend,
    )
    molecules = [m for m in (gen_run.canonical.get("molecules") or []) if m.get("smiles")]
    if not molecules:
        return ToolOutcome(
            tool=spec.name, runner="pipeline", ok=False, title=spec.title, renderer=spec.renderer,
            error=gen_run.error or "Molecule generation produced nothing.",
            elapsed_s=time.time() - started, steps=steps,
        )
    steps.append({"step": "molecule_generation", "backend": gen_run.backend, "n": len(molecules),
                  "simulated": gen_run.simulated, "latency_ms": gen_run.latency_ms})

    report(0.5, "Step 4/4: docking candidates")
    pocket_target = args.get("target_pocket") or (pockets_r[0]["id"] if pockets_r else "")
    target_pocket = next((p for p in pockets_r if p["id"] == pocket_target), pockets_r[0] if pockets_r else None)
    pocket_list = [target_pocket] if target_pocket else None

    rows: List[Dict[str, Any]] = []
    for i, mol in enumerate(molecules):
        report(0.5 + 0.45 * (i + 1) / len(molecules), f"Docking candidate {i + 1}/{len(molecules)}")
        smiles = mol["smiles"]
        try:
            dock_result = dock_mod.dock(
                structure, smiles, num_poses=max(1, n_poses), seed=1000 + i,
                refine_iterations=90, pockets=pocket_list,
            )
        except Exception as exc:  # noqa: BLE001
            rows.append({"smiles": smiles, "error": str(exc), "rank": None})
            continue
        poses = dock_result.get("poses") or []
        best = poses[0] if poses else None
        descriptor = chem_mod.report_molecule(smiles)
        rows.append(
            {
                "smiles": smiles,
                "generator_score": mol.get("score"),
                "similarity_to_seed": next((r["similarity"] for r in (gen_run.meta.get("rows") or [])
                                            if r["smiles"] == smiles), None),
                "best_confidence": best.confidence if best else None,
                "best_score": best.score if best else None,
                "hbonds": best.hbonds if best else None,
                "contacts": best.contacts if best else None,
                "clashes": best.clashes if best else None,
                "pocket": best.pocket_id if best else "",
                "qed": descriptor.qed,
                "clogp": descriptor.clogp,
                "mw": descriptor.mw,
                "sa_score": descriptor.sa_score,
                "ro5_pass": descriptor.ro5_pass,
                "alerts": len(descriptor.filter_alerts),
                "descriptor": descriptor.to_dict(),
                "pose_confidences": [p.confidence for p in poses],
                "interaction_residues": best.interaction_residues if best else [],
            }
        )

    ok_rows = [r for r in rows if r.get("best_confidence") is not None]
    # A composite ranking that rewards a good pose, good drug-likeness and clean
    # chemistry, and penalises remaining steric clashes. Documented so the user
    # can disagree with it and re-sort in the table.
    for row in ok_rows:
        row["composite"] = round(
            0.5 * (row["best_confidence"] or 0)
            + 0.3 * (row["qed"] or 0)
            + 0.15 * (1.0 - min(1.0, (row["clashes"] or 0) / 12.0))
            + 0.05 * (0.0 if row["alerts"] else 1.0),
            4,
        )
    ok_rows.sort(key=lambda r: -r["composite"])
    for rank, row in enumerate(ok_rows, start=1):
        row["rank"] = rank

    top = ok_rows[0] if ok_rows else None
    summary = (
        f"Screened {len(molecules)} analogue(s) against {structure.n_residues} residues "
        f"({target_source}); {len(pockets_r)} cavity(ies) detected, docking into "
        f"{target_pocket['id'] if target_pocket else 'the whole surface'}."
    )
    if top:
        summary += (
            f" Best candidate {top['smiles']} (composite {top['composite']}, pose confidence "
            f"{top['best_confidence']:.2f}, QED {top['qed']:.2f}, {top['hbonds']} H-bond(s))."
        )
    return ToolOutcome(
        tool=spec.name, runner="pipeline", ok=True, title=spec.title, renderer=spec.renderer,
        payload={
            "rows": ok_rows,
            "failed_rows": [r for r in rows if r.get("best_confidence") is None],
            "pockets": pockets_r,
            "structure": structure,
            "seed_smiles": seed_smiles,
            "property_name": property_name,
            "target_source": target_source,
            "generator": gen_run.backend,
            "n_molecules": len(molecules),
            "pdb": structure.pdb_text,
        },
        summary=summary, steps=steps, elapsed_s=time.time() - started,
    )


def tool_schemas(names: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """Function-calling schemas for the planner."""
    specs = [TOOLS[n] for n in names] if names else list(TOOLS.values())
    return [s.schema() for s in specs]


def catalogue_for_prompt() -> str:
    """Short human-readable tool list for the planner prompt."""
    lines = []
    for spec in TOOLS.values():
        params = ", ".join(
            f"{k}{'*' if v.get('required') else ''}" for k, v in spec.parameters.items()
        )
        lines.append(f"- {spec.name}({params}): {spec.description}")
    return "\n".join(lines)
