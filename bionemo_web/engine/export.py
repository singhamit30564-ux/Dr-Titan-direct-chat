"""Turning results into files a scientist can actually use.

Everything the console computes can leave the app in a standard format:

* structures -> PDB (and a JSON summary)
* molecule sets -> SDF (with 3D coordinates and properties), CSV of descriptors
* docking -> SDF per pose plus a CSV of scores/confidences
* screenings -> a ranked CSV, an SDF of the shortlist and a Markdown report
* any run -> a JSON provenance manifest

Files are written under ``.runtime/artifacts/<run_id>/`` and registered in the
store, so the History page can serve them again later.
"""

from __future__ import annotations

import csv
import datetime as _dt
import io
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..bio import chem as chem_mod
from ..bio import structure as st_mod
from .store import ARTIFACT_DIR

TEXT_FORMATS = ["pdb", "sdf", "csv", "json", "md", "fasta", "txt"]


def _slug(text: str, limit: int = 60) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")
    return clean[:limit] or "result"


def _stamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def artifact_dir(run_id: str = "", conversation_id: str = "") -> Path:
    folder = ARTIFACT_DIR / (run_id or conversation_id or _stamp())
    folder.mkdir(parents=True, exist_ok=True)
    return folder


# --------------------------------------------------------------------------
# Individual writers
# --------------------------------------------------------------------------


def write_text(name: str, content: str, *, run_id: str = "") -> Path:
    path = artifact_dir(run_id) / _slug(name)
    path.write_text(content, encoding="utf-8")
    return path


def write_json(name: str, payload: Any, *, run_id: str = "") -> Path:
    path = artifact_dir(run_id) / (name if name.endswith(".json") else name + ".json")
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def write_csv(name: str, rows: Sequence[Dict[str, Any]], *, run_id: str = "") -> Path:
    path = artifact_dir(run_id) / (name if name.endswith(".csv") else name + ".csv")
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    # Union of keys, keeping first-seen order.
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _cell(v) for k, v in row.items() if k in keys})
    return path


def _cell(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return "; ".join(str(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, default=str)
    if isinstance(value, (np.floating, float)):
        return round(float(value), 4)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def write_pdb(name: str, structure: "st_mod.Structure", *, run_id: str = "") -> Path:
    text = structure.pdb_text if structure is not None else ""
    path = artifact_dir(run_id) / (name if name.endswith(".pdb") else name + ".pdb")
    path.write_text(text, encoding="utf-8")
    return path


def write_sdf(name: str, molecules: Sequence[Tuple[str, Optional[np.ndarray]]], *,
              properties: Optional[List[Dict[str, Any]]] = None, run_id: str = "") -> Path:
    """Write an SDF. ``molecules`` is a sequence of (SMILES, optional 3D coords).

    Coordinates are embedded on demand when absent, so every record is 3D and
    re-dockable. Properties (score, confidence, descriptor columns) are attached
    as SD tags, which is what downstream tools expect.
    """
    path = artifact_dir(run_id) / (name if name.endswith(".sdf") else name + ".sdf")
    chunks: List[str] = []
    for index, (smiles, coords) in enumerate(molecules):
        if coords is not None:
            block = _mol_block_from_coords(smiles, np.asarray(coords))
        else:
            block = chem_mod.mol_block_3d(smiles)
        if not block:
            continue
        props = properties[index] if properties and index < len(properties) else {}
        tags = "".join(
            f">  <{k}>\n{_cell(v)}\n\n" for k, v in (props or {}).items()
            if not isinstance(v, (dict, list)) or True
        )
        chunks.append(block.rstrip("\n") + "\n" + tags + "$$$$\n")
    path.write_text("".join(chunks), encoding="utf-8")
    return path


def _mol_block_from_coords(smiles: str, coords: np.ndarray) -> str:
    """A V2000 mol block for the SMILES topology carrying externally supplied 3D coords."""
    mol, warnings = chem_mod.parse_smiles(smiles)
    if mol is None:
        return ""
    try:
        from rdkit import Chem
        from rdkit.Geometry import Point3D

        m = Chem.RWMol(mol)
        conf = Chem.Conformer(m.GetNumAtoms())
        for i in range(m.GetNumAtoms()):
            x, y, z = (coords[i] if i < len(coords) else (0.0, 0.0, 0.0))
            conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
        m.AddConformer(conf, assignId=True)
        out = Chem.MolToMolBlock(m.GetMol(), kekulize=True)
        return out
    except Exception:
        return chem_mod.mol_block_3d(smiles)


# --------------------------------------------------------------------------
# Outcome-level exporters
# --------------------------------------------------------------------------


def record_artifacts(records: Sequence[Dict[str, Any]], *, run_id: str = "",
                     conversation_id: str = "") -> None:
    """Register written files in the store so History can serve them again.

    Failures are swallowed on purpose: an export that cannot be indexed is still a
    usable download, and losing the index entry must never break the run.
    """
    if not records:
        return
    try:
        from .store import get_store

        store = get_store()
        for record in records:
            store.add_artifact(
                kind=record.get("kind", "file"),
                name=record.get("name", "artifact"),
                path=record.get("path", ""),
                run_id=run_id,
                conversation_id=conversation_id,
                meta={"tool": record.get("tool", "")},
            )
    except Exception:  # pragma: no cover
        pass


def outcome_files(
    outcome: Any, *, run_id: str = "", record: bool = True, conversation_id: str = ""
) -> List[Dict[str, Any]]:
    """Write every sensible file for one tool outcome; return artifact records."""
    written: List[Dict[str, Any]] = []
    payload = getattr(outcome, "payload", {}) or {}
    name_base = f"{outcome.tool}_{_stamp()}"

    def add(path: Path, kind: str) -> None:
        if path and Path(path).exists():
            written.append({"kind": kind, "name": Path(path).name, "path": str(path),
                            "size": Path(path).stat().st_size, "tool": outcome.tool})

    if payload.get("pdb") and isinstance(payload["pdb"], str):
        path = artifact_dir(run_id) / f"{name_base}.pdb"
        path.write_text(payload["pdb"], encoding="utf-8")
        add(path, "pdb")
    if payload.get("structure") is not None:
        add(write_json(f"{name_base}_structure.json",
                       payload["structure"].to_dict(profiles=True), run_id=run_id), "json")
        pockets = payload.get("pockets") or []
        if pockets:
            add(write_csv(f"{name_base}_pockets.csv", pockets, run_id=run_id), "csv")

    if payload.get("reports"):
        reports = payload["reports"]
        flat = [{k: v for k, v in rep.items() if not isinstance(v, (list, dict))} for rep in reports]
        add(write_csv(f"{name_base}_descriptors.csv", flat, run_id=run_id), "csv")
        add(write_sdf(f"{name_base}_molecules.sdf",
                      [(rep["canonical_smiles"], None) for rep in reports],
                      properties=flat, run_id=run_id), "sdf")

    if payload.get("poses"):
        poses = payload["poses"]
        add(write_csv(f"{name_base}_poses.csv", [p.to_dict() for p in poses], run_id=run_id), "csv")

    if payload.get("designs"):
        rows = []
        for design in payload["designs"]:
            if isinstance(design, dict):
                rows.append({
                    "sequence": design.get("sequence", ""),
                    "score": design.get("score"),
                    "global_score": design.get("global_score"),
                    "recovery": design.get("recovery"),
                    "n_mutations": len(design.get("mutations") or []),
                    "mutations": ", ".join(
                        f"{m.get('wt')}{m.get('position')}{m.get('designed')}"
                        for m in (design.get("mutations") or [])
                    )[:900],
                })
        add(write_csv(f"{name_base}_designs.csv", rows, run_id=run_id), "csv")
        fasta = "\n".join(
            f">{outcome.tool}|design_{i + 1}|score={d.get('score')}\n{d.get('sequence', '')}"
            for i, d in enumerate(payload["designs"]) if isinstance(d, dict)
        )
        add(write_text(f"{name_base}_designs.fasta", fasta, run_id=run_id), "fasta")

    if payload.get("rows") and outcome.tool == "virtual_screening":
        rows = payload["rows"]
        flat = [{k: v for k, v in row.items() if not isinstance(v, (dict, list))} for row in rows]
        add(write_csv(f"{name_base}_ranked.csv", flat, run_id=run_id), "csv")
        add(write_sdf(
            f"{name_base}_top_candidates.sdf",
            [(row["smiles"], None) for row in rows],
            properties=flat,
            run_id=run_id,
        ), "sdf")
        add(write_json(f"{name_base}_screening.json",
                       {"rows": rows, "pockets": payload.get("pockets", []),
                        "seed_smiles": payload.get("seed_smiles", ""),
                        "generator": payload.get("generator", "")}, run_id=run_id), "json")
        brief = screening_report(outcome)
        add(write_text(f"{name_base}_report.md", brief, run_id=run_id), "md")

    if payload.get("generated"):
        add(write_text(f"{name_base}_generated.fasta", payload["generated"], run_id=run_id), "fasta")
    if payload.get("report") is not None:
        add(write_json(f"{name_base}_report.json", payload["report"], run_id=run_id), "json")
    if getattr(outcome, "run", None) is not None:
        add(write_json(f"{name_base}_provenance.json", outcome.run.provenance(), run_id=run_id), "json")
    if record:
        record_artifacts(written, run_id=run_id or outcome.run_id, conversation_id=conversation_id)
    return written


def screening_report(outcome: Any) -> str:
    """A short Markdown brief for a screening run."""
    payload = outcome.payload or {}
    rows = payload.get("rows") or []
    pockets = payload.get("pockets") or []
    lines = [
        "# Virtual screening report",
        "",
        f"- Generated: {_dt.datetime.now().isoformat(timespec='seconds')}",
        f"- Target: {payload.get('target_source', 'unknown')} "
        f"({payload.get('structure').n_residues if payload.get('structure') else '?'} residues)",
        f"- Seed molecule: `{payload.get('seed_smiles', '')}`",
        f"- Generator backend: {payload.get('generator', '')}",
        f"- Analogues docked: {payload.get('n_molecules', len(rows))}",
        "",
        "## Detected cavities",
        "",
        "| cavity | volume (A^3) | max enclosure | max depth (A) | druggability |",
        "| --- | --- | --- | --- | --- |",
    ]
    for pocket in pockets[:6]:
        lines.append(
            f"| {pocket.get('id')} | {pocket.get('volume_a3')} | {pocket.get('max_enclosure')} "
            f"| {pocket.get('max_burial_depth')} | {pocket.get('druggability_hint')} |"
        )
    lines += [
        "",
        "## Ranking",
        "",
        "_Composite = 0.5 x pose confidence + 0.3 x QED + 0.15 x clash cleanliness "
        "+ 0.05 x alert-free. Re-sort in the app if you weight these differently._",
        "",
        "| rank | SMILES | composite | confidence | H-bonds | QED | cLogP | MW | alerts |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('rank')} | `{row.get('smiles')}` | {row.get('composite')} "
            f"| {_fmt(row.get('best_confidence'))} | {row.get('hbonds')} | {_fmt(row.get('qed'))} "
            f"| {_fmt(row.get('clogp'))} | {_fmt(row.get('mw'), 1)} | {row.get('alerts')} |"
        )
    lines += ["", "## Provenance", "", f"- {outcome.summary or 'no summary'}", ""]
    if getattr(outcome, "run", None) is not None and outcome.run.warnings:
        lines.append("### Warnings")
        lines += [f"- {w}" for w in outcome.run.warnings]
    lines.append("")
    return "\n".join(lines)


def _fmt(value: Any, digits: int = 2) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def conversation_markdown(conversation: Dict[str, Any], messages: Sequence[Dict[str, Any]]) -> str:
    """Export a whole conversation, results included, as Markdown."""
    conv = conversation or {}
    lines = [
        f"# {conv.get('title', 'Conversation')}",
        "",
        f"- Conversation id: `{conv.get('id', '')}`",
        f"- Created: {_fmt_time(conv.get('created_at'))}",
        f"- Exported: {_dt.datetime.now().isoformat(timespec='seconds')}",
        "- App: Dr-Titan BioNeMo Web Console",
        "",
        "---",
        "",
    ]
    for message in messages:
        role = message.get("role", "user")
        icon = "You" if role == "user" else "Console"
        lines += [f"## {icon}", "", str(message.get("content", "")).strip(), ""]
        for run in message.get("runs") or []:
            lines += [
                f"<details><summary>run `{run.get('id')}` - {run.get('model_key')} "
                f"({run.get('backend')}{', simulated' if run.get('simulated') else ''})</summary>",
                "",
                "```json",
                json.dumps(run.get("provenance") or {}, indent=2, default=str)[:4000],
                "```",
                "",
                "</details>",
                "",
            ]
    return "\n".join(lines)


def _fmt_time(value: Any) -> str:
    try:
        return _dt.datetime.fromtimestamp(float(value)).isoformat(timespec="seconds")
    except (TypeError, ValueError):
        return "unknown"


def dataframe_rows(outcome: Any) -> List[Dict[str, Any]]:
    """Tabular view of an outcome, for the UI's table renderer."""
    payload = getattr(outcome, "payload", {}) or {}
    if payload.get("rows"):
        return payload["rows"]
    if payload.get("reports"):
        return [{k: v for k, v in rep.items() if not isinstance(v, (list, dict))}
                for rep in payload["reports"]]
    if payload.get("poses"):
        return [pose.to_dict() for pose in payload["poses"]]
    if payload.get("designs"):
        return [d for d in payload["designs"] if isinstance(d, dict)]
    if payload.get("pockets"):
        return payload["pockets"]
    return []


def nim_request_curl(outcome: Any) -> str:
    """A replayable curl command for the NIM call behind this outcome.

    Shipping this with every result means a user can reproduce the exact HTTP
    request against their own container without reading the source.
    """
    run = getattr(outcome, "run", None)
    if run is None:
        return ""
    payload = json.dumps(run.request_payload or {}, indent=2)
    if len(payload) > 1500:
        payload = json.dumps({k: (v if not isinstance(v, str) or len(v) < 200 else v[:200] + "...")
                              for k, v in (run.request_payload or {}).items()}, indent=2)
    url = run.url or "http://localhost:8000/<model-path>"
    return (
        f"curl -s -X POST '{url}' \\\n"
        f"  -H 'Content-Type: application/json' \\\n"
        f"  -H 'Authorization: Bearer $NGC_API_KEY' \\\n"
        f"  -d '{payload}'"
    )


def write_markdown(name: str, content: str, *, run_id: str = "") -> Path:
    return write_text(name if name.endswith(".md") else name + ".md", content, run_id=run_id)
