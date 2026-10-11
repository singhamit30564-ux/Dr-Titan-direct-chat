"""Turns a sentence into a plan of tool calls.

Design principle: **the router is deterministic**.  Model calls are chosen by
pattern matching, entity extraction and explicit slash commands, not by asking a
language model to guess -- so the same sentence always triggers the same
scientific action, and a wrong call is a bug you can read in this file rather
than a temperature setting.

A language model is used only as a *planner of last resort*: when nothing
matches confidently but the message clearly asks for work, the model is shown the
tool catalogue and asked for JSON.  Its answer is validated against the same
schemas, so a hallucinated tool or argument is rejected.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..nims import catalog
from . import tools as tools_mod
from .llm import LLMClient, extract_json

AA_ONLY = set("ACDEFGHIKLMNPQRSTVWY")
DNA_ONLY = set("ACGTU")
SMILES_HINTS = set("abcdefghijklmnopqrstuvwxyz0123456789()=#[]@+-/\\%.$:*\"'")

#: Characters that can legitimately appear in a SMILES string.
SMILES_ALPHABET = set("BCNOFPSIHKVLAGTUYREWDMXZbcno psihk[]()=#@+-0123456789./\\%$:*\"'")
SMILES_ALPHABET = {c for c in SMILES_ALPHABET if c != " "}

#: Characters that only ever appear in notation, never in a plain English word.
SMILES_DISTINCTIVE = set("()=#[]@+-/\\%.*0123456789")
#: Single-character element symbols valid in SMILES, sensitive to case:
#: uppercase is an aliphatic atom, lowercase an aromatic one.
_ONE_CHAR_ELEMENTS = set("BCNOFPSIKVYUWHAsebcno")
#: Two-character element symbols whose first letter must be uppercase.
_TWO_CHAR_ELEMENTS = {
    "He", "Li", "Be", "Ne", "Na", "Mg", "Al", "Si", "Cl", "Ar", "Ca", "Sc", "Ti", "Cr",
    "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr",
    "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm",
    "Yb", "Lu", "Hf", "Ta", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po",
    "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf",
}


def _letters_spell_elements(token: str) -> bool:
    """True if a bare-letters token can be read as a run of element symbols.

    This is the cheap test that keeps English out of RDKit's parser. SMILES is
    case-sensitive on purpose: ``CCO`` is an aliphatic carbon, carbon, oxygen,
    while ``backbone`` fails at the second character (``a`` is not an element) and
    ``is`` fails at the first (``i`` is not an element symbol; ``I`` is iodine).
    """
    index = 0
    length = len(token)
    while index < length:
        two = token[index:index + 2]
        if len(two) == 2 and two in _TWO_CHAR_ELEMENTS:
            index += 2
            continue
        one = token[index]
        if one in _ONE_CHAR_ELEMENTS:
            index += 1
            continue
        return False
    return index == length

#: keyword -> weight, per tool.  Order of evaluation is by total weight.
INTENT_KEYWORDS: Dict[str, List[Tuple[str, float]]] = {
    "fold_protein": [
        ("fold", 3.0), ("structure prediction", 3.0), ("predict the structure", 3.0),
        ("predict structure", 3.0), ("esmfold", 3.0), ("alphafold2", 3.0), ("alphafold", 3.0),
        ("af2", 2.5), ("openfold", 3.0), ("3d structure", 2.0), ("model the structure", 2.5),
        ("fold it", 2.5), ("folding", 2.0),
    ],
    "predict_complex": [
        ("co-fold", 3.0), ("cofold", 3.0), ("complex with", 2.5), ("binding affinity", 2.5),
        ("boltz", 3.0), ("affinity of", 2.0), ("protein-ligand complex", 3.0),
    ],
    "embed_sequence": [
        ("embed", 2.5), ("embedding", 2.5), ("esm2", 2.5), ("esm-2", 2.5), ("amplify", 2.0),
        ("geneformer", 2.0), ("vectorise", 2.0), ("vectorize", 2.0), ("latent representation", 2.0),
    ],
    "generate_molecules": [
        ("generate molecule", 3.0), ("generate molecules", 3.0), ("analogue", 2.5), ("analog", 2.0),
        ("molmim", 3.0), ("genmol", 3.0), ("generate compound", 2.5), ("make molecules", 3.0),
        ("molecules similar", 2.5), ("optimise", 1.5), ("optimize", 1.5), ("qed", 1.5),
        ("plogp", 2.0), ("derivatives", 2.0), ("new molecules", 2.5),
    ],
    "dock_ligand": [
        ("dock", 3.0), ("docking", 3.0), ("diffdock", 3.0), ("binding pose", 3.0),
        ("poses", 2.0), ("bind to", 2.0), ("binds to", 2.0), ("into the protein", 2.0),
    ],
    "design_sequences": [
        ("design", 2.5), ("proteinmpnn", 3.0), ("mpnn", 2.5), ("inverse folding", 3.0),
        ("redesign", 3.0), ("stabilise", 2.0), ("stabilize", 2.0), ("solubilise", 2.0),
        ("solubilize", 2.0), ("new sequences for", 2.5), ("sequence design", 3.0),
    ],
    "generate_dna": [
        ("dna", 2.0), ("nucleotide", 2.5), ("genome", 2.0), ("genomic", 2.5), ("evo2", 3.0),
        ("evo 2", 3.0), ("codon", 2.5), ("mrna", 2.5), ("base pair", 2.0), ("bp", 1.5),
    ],
    "analyse_sequence": [
        ("analyse", 1.5), ("analyze", 1.5), ("properties", 2.0), ("isoelectric", 3.0), ("pi of", 2.5),
        ("molecular weight", 2.5), ("gravy", 3.0), ("instability", 2.5), ("motif", 2.0),
        ("orf", 2.5), ("what is the pi", 3.0), ("hydrophob", 1.5), ("composition", 1.5),
    ],
    "analyse_molecules": [
        ("lipinski", 3.0), ("pains", 3.0), ("drug-like", 2.5), ("druglike", 2.5), ("descriptor", 2.5),
        ("cluster these", 2.5), ("scaffold", 2.5), ("sa score", 3.0), ("tpsa", 3.0),
        ("analyse these molecules", 3.0), ("tanimoto", 3.0),
    ],
    "analyse_structure": [
        ("pocket", 3.0), ("cavity", 3.0), ("sasa", 3.0), ("secondary structure", 2.5),
        ("contacts", 1.5), ("quality of", 1.5), ("analyse this structure", 3.0), ("buried", 2.0),
    ],
    "compare_structures": [
        ("compare", 3.0), ("rmsd", 3.0), ("superpose", 3.0), ("superimpose", 3.0),
        ("align", 2.0), ("difference between", 2.5),
    ],
    "virtual_screening": [
        ("virtual screen", 4.0), ("screening", 3.0), ("screen candidates", 3.5),
        ("pipeline", 3.0), ("hit finding", 3.0), ("end to end", 2.0), ("full workflow", 2.0),
        ("blueprint", 2.5),
    ],
}

SLASH_COMMANDS: Dict[str, str] = {
    "/fold": "fold_protein",
    "/structure": "fold_protein",
    "/complex": "predict_complex",
    "/embed": "embed_sequence",
    "/molecules": "generate_molecules",
    "/molmim": "generate_molecules",
    "/dock": "dock_ligand",
    "/diffdock": "dock_ligand",
    "/design": "design_sequences",
    "/mpnn": "design_sequences",
    "/dna": "generate_dna",
    "/evo2": "generate_dna",
    "/analyse": "analyse_sequence",
    "/analyze": "analyse_sequence",
    "/molecule-report": "analyse_molecules",
    "/structure-report": "analyse_structure",
    "/compare": "compare_structures",
    "/screen": "virtual_screening",
}

#: Verbs that make a molecule request a *generation* request. Without one of
#: these, "what is the QED of caffeine?" is a property question, not a request to
#: invent analogues.
GENERATION_VERBS = (
    "generate", "make", "create", "design", "propose", "suggest", "enumerate",
    "explore", "analog", "analogue", "similar", "derivative", "optimis", "optimize",
    "improve", "new molecule", "screen",
)

#: Verb forms that mark a read-only analysis request.
ANALYSIS_VERBS = (
    "analyse", "analyze", "what is", "what's", "how much", "how many", "calculate",
    "compute", "report", "give me", "tell me", "show me", "check", "properties",
    "descriptors", "predict the p", "isoelectric",
)

NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "twelve": 12, "twenty": 20, "fifty": 50,
    "hundred": 100,
}


@dataclass
class ToolPlan:
    """One planned tool call."""

    tool: str
    args: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    rationale: str = ""
    needs: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"tool": self.tool, "args": self.args, "confidence": round(self.confidence, 2),
                "rationale": self.rationale, "needs": self.needs}


@dataclass
class Plan:
    """The router's reading of a user turn."""

    steps: List[ToolPlan] = field(default_factory=list)
    entities: Dict[str, Any] = field(default_factory=dict)
    intents: Dict[str, float] = field(default_factory=dict)
    chat_only: bool = False
    clarifications: List[str] = field(default_factory=list)
    planner: str = "rules"
    notes: List[str] = field(default_factory=list)

    @property
    def is_actionable(self) -> bool:
        return bool(self.steps)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "steps": [s.to_dict() for s in self.steps],
            "entities": {k: (v[:80] + "..." if isinstance(v, str) and len(v) > 80 else v)
                         for k, v in self.entities.items()},
            "intents": {k: round(v, 2) for k, v in sorted(self.intents.items(), key=lambda kv: -kv[1])[:6]},
            "chat_only": self.chat_only,
            "planner": self.planner,
            "notes": self.notes,
        }


# --------------------------------------------------------------------------
# Entity extraction
# --------------------------------------------------------------------------


def _looks_like_smiles(token: str) -> Tuple[bool, int]:
    """Return (is plausible SMILES, heavy atom count), validated with RDKit."""
    if len(token) < 2 or len(token) > 400:
        return False, 0
    if not any(c in SMILES_HINTS for c in token):
        return False, 0
    if not set(token) <= SMILES_ALPHABET:
        return False, 0
    if not any(c.isalpha() for c in token):
        # Every SMILES names at least one atom, so a digit-only token such as the
        # "10" in "generate 10 analogues" can never be one. RDKit would answer
        # with a parse error on stderr, which looks like a crash to a user.
        return False, 0
    if not any(c in SMILES_DISTINCTIVE for c in token):
        # No brackets, digits, charges or bond symbols: the token is a bare letter
        # run, so it has to spell element symbols or it is just a word.
        if not _letters_spell_elements(token):
            return False, 0
    try:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(token)
    except Exception:
        return False, 0
    if mol is None:
        return False, 0
    heavy = mol.GetNumHeavyAtoms()
    if heavy < 2:
        return False, 0
    return True, heavy


def extract_entities(text: str) -> Dict[str, Any]:
    """Pull sequences, SMILES, PDB blocks and parameters out of a message."""
    entities: Dict[str, Any] = {}
    raw = text or ""

    # PDB block (multi-line)
    if "ATOM" in raw or "HETATM" in raw:
        lines = [ln for ln in raw.splitlines()
                 if ln.startswith(("ATOM", "HETATM", "TER", "HEADER", "REMARK", "CRYST1", "SEQRES"))]
        if lines:
            entities["pdb"] = "\n".join(lines) + "\nEND\n"

    tokens = re.split(r"[\s,;]+", raw)
    dna_tokens, protein_tokens, smiles_tokens = [], [], []
    for token in tokens:
        cleaned = token.strip().strip(".,:;()[]\"'")
        if not cleaned:
            continue
        upper = cleaned.upper()
        if 8 <= len(cleaned) <= 20000 and set(upper) <= DNA_ONLY and len(cleaned) >= 10:
            dna_tokens.append(cleaned)
            continue
        if 10 <= len(cleaned) <= 20000 and set(upper) <= AA_ONLY:
            protein_tokens.append(cleaned)
            continue
        ok, heavy = _looks_like_smiles(cleaned)
        if ok and heavy >= 3:
            smiles_tokens.append(cleaned)

    if dna_tokens:
        entities["dna"] = max(dna_tokens, key=len)
    if protein_tokens:
        entities["sequence"] = max(protein_tokens, key=len)
    if smiles_tokens:
        entities["smiles"] = max(smiles_tokens, key=len)
        entities["smiles_all"] = smiles_tokens
    else:
        # No SMILES in the message: check for a molecule named in words, so that
        # "dock aspirin into this protein" works without pasting a structure.
        from ..bio import chem as chem_mod

        lowered = raw.lower()
        for name, smiles in chem_mod.COMMON_MOLECULES.items():
            if len(name) >= 4 and re.search(rf"\b{re.escape(name)}\b", lowered):
                entities["smiles"] = smiles
                entities["smiles_all"] = [smiles]
                entities["smiles_name"] = name
                break

    # Numbers with a nearby noun
    patterns = {
        "num_molecules": r"(\d{1,3})\s*(?:molecules|analogues|analogs|compounds|derivatives|candidates)",
        "num_poses": r"(\d{1,3})\s*(?:poses|docking poses)",
        "num_sequences": r"(\d{1,3})\s*(?:sequences|designs|variants)",
        "num_tokens": r"(\d{1,5})\s*(?:nt|bp|bases|nucleotides|tokens)",
        "num_designs": r"(\d{1,3})\s*designs",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, raw, re.I)
        if match:
            entities[key] = int(match.group(1))
    temp = re.search(r"temperature\s*(?:of|=|:)?\s*([0-9]*\.?[0-9]+)", raw, re.I)
    if temp:
        try:
            entities["temperature"] = float(temp.group(1))
        except ValueError:
            pass
    sim = re.search(r"similarity\s*(?:of|>=|>|=|at least)?\s*([0-9]*\.?[0-9]+)", raw, re.I)
    if sim:
        try:
            entities["min_similarity"] = float(sim.group(1))
        except ValueError:
            pass
    for word, value in NUMBER_WORDS.items():
        if re.search(rf"\b{word}\s+(?:molecules|analogues|analogs|compounds|poses|sequences|designs)\b", raw, re.I):
            if "molecules" in raw.lower() or "compound" in raw.lower() or "analog" in raw.lower():
                entities.setdefault("num_molecules", value)
    if re.search(r"\bminimi[sz]e\b", raw, re.I):
        entities["minimize"] = True
    if re.search(r"\bplogp\b", raw, re.I):
        entities["property_name"] = "plogP"
    elif re.search(r"\bqed\b", raw, re.I):
        entities["property_name"] = "QED"
    if re.search(r"\bfixed positions?\s*[:=]?\s*([0-9,\s]+)", raw, re.I):
        match = re.search(r"\bfixed positions?\s*[:=]?\s*([0-9,\s]+)", raw, re.I)
        entities["fixed_positions"] = match.group(1).strip()

    # Explicit model mentions
    for spec in catalog.all_models():
        for needle in (spec.key, spec.name.lower(), spec.name.lower().split(" (")[0]):
            if needle and re.search(rf"\b{re.escape(needle)}\b", raw, re.I):
                entities.setdefault("model_mentions", []).append(spec.key)
                break
    return entities


# --------------------------------------------------------------------------
# Intent scoring
# --------------------------------------------------------------------------


def score_intents(text: str) -> Dict[str, float]:
    lowered = (text or "").lower()
    scores: Dict[str, float] = {}
    for tool, keywords in INTENT_KEYWORDS.items():
        score = 0.0
        for phrase, weight in keywords:
            if phrase in lowered:
                score += weight
        if score:
            scores[tool] = score
    # A slash command is decisive.
    first = lowered.strip().split()[0] if lowered.strip() else ""
    if first in SLASH_COMMANDS:
        scores = {SLASH_COMMANDS[first]: 10.0}
    return scores


def _disambiguate(scores: Dict[str, float], entities: Dict[str, Any], text: str) -> None:
    """Apply the rules that keyword weighting alone gets wrong.

    Three corrections, each motivated by a real sentence:

    * "what is the QED of caffeine?" shares the word *QED* with "generate analogues
      optimising QED", but only the second is a generation request -- so a
      generation score with no generation verb is reassigned to analysis.
    * "analyse the molecule paracetamol" contains *analyse*, which both analysis
      tools claim; the payload decides: a resolved molecule means molecule
      analysis, a sequence means sequence analysis.
    * a message carrying both (a protein and a ligand) should keep whatever the
      keyword scoring chose, so nothing is moved when both entities are present.

    Slash commands are explicit, so they are never second-guessed: ``/molecules``
    means generate molecules even though the text contains no verb at all.
    """
    lowered = text.lower()
    first = lowered.strip().split()[0] if lowered.strip() else ""
    if first in SLASH_COMMANDS:
        return
    has_verb = any(verb in lowered for verb in GENERATION_VERBS)
    if "generate_molecules" in scores and not has_verb:
        scores["analyse_molecules"] = scores.get("analyse_molecules", 0.0) + scores.pop(
            "generate_molecules"
        )

    molecule_present = bool(entities.get("smiles"))
    sequence_present = bool(entities.get("sequence") or entities.get("dna"))
    if molecule_present and not sequence_present:
        if "analyse_sequence" in scores:
            scores["analyse_molecules"] = scores.get("analyse_molecules", 0.0) + scores.pop(
                "analyse_sequence"
            )
    elif sequence_present and not molecule_present:
        if "analyse_molecules" in scores:
            scores["analyse_sequence"] = scores.get("analyse_sequence", 0.0) + scores.pop(
                "analyse_molecules"
            )
    # Nothing is pruned for missing payloads: the router sees one message at a
    # time, so "now fold it" legitimately carries no sequence and still has to
    # route to a tool that will pick the protein up from the conversation.


def _model_args(entities: Dict[str, Any], tool: str, default: str) -> Dict[str, Any]:
    mentions = entities.get("model_mentions") or []
    compatible = {
        "fold_protein": {"esmfold", "alphafold2", "openfold3", "boltz2"},
        "predict_complex": {"boltz2", "openfold3"},
        "embed_sequence": {"esm2", "amplify", "geneformer"},
        "generate_molecules": {"molmim", "genmol"},
        "dock_ligand": {"diffdock"},
        "design_sequences": {"proteinmpnn", "esm2_generate"},
        "generate_dna": {"evo2", "codonfm"},
    }.get(tool, set())
    for key in mentions:
        if key in compatible:
            return {"model": key}
    return {"model": default}


# --------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------


def plan(
    text: str,
    *,
    context: Optional[Dict[str, Any]] = None,
    llm: Optional[LLMClient] = None,
    allow_llm: bool = True,
) -> Plan:
    """Build a plan for a user turn."""
    context = context or {}
    entities = extract_entities(text)
    stripped = (text or "").strip()
    command_args = _command_arguments(stripped)
    entities.update({k: v for k, v in command_args.items() if v not in (None, "")})

    scores = score_intents(stripped)
    _disambiguate(scores, entities, stripped)
    result = Plan(entities=entities, intents=scores)
    if not stripped:
        result.chat_only = True
        return result

    # Explicit analysis of pasted molecules / structures wins when the payload is
    # unambiguous, because it is cheap and always useful.
    if "pdb" in entities and not scores:
        result.steps.append(ToolPlan("analyse_structure", {"pdb": entities["pdb"]}, 0.7,
                                     "a PDB block was pasted"))
        return result

    if not scores:
        # Nothing matched: let the LLM propose a tool call if one is available.
        if allow_llm and llm is not None and llm.status().available:
            planned = _llm_plan(stripped, entities, context, llm)
            if planned.steps:
                result.steps = planned.steps
                result.planner = "llm"
                result.notes.extend(planned.notes)
                return result
        result.chat_only = True
        return result

    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    primary, primary_score = ranked[0]
    total = sum(scores.values()) or 1.0
    confidence = min(0.95, primary_score / max(total, 1.0) + min(0.3, primary_score / 10.0))

    # Chained workflows -------------------------------------------------
    wants_fold = scores.get("fold_protein", 0) >= 2.0
    wants_dock = scores.get("dock_ligand", 0) >= 2.0
    wants_design = scores.get("design_sequences", 0) >= 2.0
    wants_molecules = scores.get("generate_molecules", 0) >= 2.0

    if scores.get("virtual_screening", 0) >= 3.0 or (
        wants_fold and wants_dock and wants_molecules
    ):
        args = {
            "protein": entities.get("pdb") or entities.get("sequence") or context.get("last_structure_pdb", ""),
            "seed_smiles": entities.get("smiles", ""),
            "num_molecules": entities.get("num_molecules", 8),
            "num_poses": entities.get("num_poses", 3),
            "property_name": entities.get("property_name", "QED"),
        }
        result.steps.append(ToolPlan("virtual_screening", args, 0.8,
                                     "screening keywords plus generation and docking entities"))
        result.planner = "rules:workflow"
        _add_missing(result, "virtual_screening", args, context)
        return result

    if wants_fold and (wants_dock or wants_design) and not entities.get("pdb"):
        base = {
            "protein": entities.get("pdb") or entities.get("sequence") or context.get("last_structure_pdb", ""),
            "seed_smiles": entities.get("smiles", ""),
            "num_molecules": entities.get("num_molecules", 10),
            "num_poses": entities.get("num_poses", 10),
            "property_name": entities.get("property_name", "QED"),
        }
        result.steps.append(ToolPlan("virtual_screening", base, 0.7,
                                     "a fold plus a follow-up step needs a structure first"))
        result.planner = "rules:workflow"
        _add_missing(result, "virtual_screening", base, context)
        return result

    # Single best tool ---------------------------------------------------
    builder = {
        "fold_protein": _fold_args,
        "predict_complex": _complex_args,
        "embed_sequence": _embed_args,
        "generate_molecules": _molecule_args,
        "dock_ligand": _dock_args,
        "design_sequences": _design_args,
        "generate_dna": _dna_args,
        "analyse_sequence": _analyse_seq_args,
        "analyse_molecules": _analyse_mol_args,
        "analyse_structure": _analyse_structure_args,
        "compare_structures": _compare_args,
    }.get(primary)
    if builder is None:
        result.chat_only = True
        return result
    args = builder(entities, context, text)
    result.steps.append(ToolPlan(primary, args, confidence, f"matched {primary.replace('_', ' ')} keywords"))

    # A secondary, non-conflicting intent that has its own payload can run too.
    if len(ranked) > 1:
        secondary, secondary_score = ranked[1]
        if (secondary_score >= 2.5 and secondary in ("analyse_sequence", "analyse_molecules")
                and primary not in ("analyse_sequence", "analyse_molecules")):
            built = ({"sequence": entities.get("sequence", "")} if secondary == "analyse_sequence"
                     else {"smiles_list": entities.get("smiles_all") or [entities.get("smiles", "")]})
            if all(built.values()):
                result.steps.append(ToolPlan(secondary, built, 0.4, "secondary analysis intent"))
    _add_missing(result, primary, args, context)
    return result


def _command_arguments(text: str) -> Dict[str, Any]:
    """Parse ``--key value`` flags from slash commands."""
    if not text.startswith("/"):
        return {}
    args: Dict[str, Any] = {}
    for key, value in re.findall(r"--([a-zA-Z_]+)[=\s]+([^\s]+)", text):
        try:
            args[key] = float(value) if re.match(r"^\d+\.\d+$", value) else int(value)
        except ValueError:
            args[key] = value
    return args


def _add_missing(result: Plan, tool: str, args: Dict[str, Any], context: Dict[str, Any]) -> None:
    """Record what the tool still needs, so the UI can prompt for it."""
    spec = tools_mod.TOOLS.get(tool)
    if not spec:
        return
    missing = [name for name in spec.requires if not args.get(name)]
    if missing:
        result.clarifications.append(
            "Missing input for " + tool + ": " + ", ".join(missing)
        )
        if result.steps:
            result.steps[-1].needs = missing


# -- argument builders -------------------------------------------------


def _fold_args(e: Dict[str, Any], ctx: Dict[str, Any], text: str) -> Dict[str, Any]:
    args = {"sequence": e.get("sequence") or ctx.get("last_sequence", "")}
    if e.get("pdb"):
        args["pdb"] = e["pdb"]
    args.update(_model_args(e, "fold_protein", "esmfold"))
    return args


def _complex_args(e: Dict[str, Any], ctx: Dict[str, Any], text: str) -> Dict[str, Any]:
    args = {
        "sequence": e.get("sequence") or ctx.get("last_sequence", ""),
        "ligand_smiles": e.get("smiles", ""),
    }
    args.update(_model_args(e, "predict_complex", "boltz2"))
    return args


def _embed_args(e: Dict[str, Any], ctx: Dict[str, Any], text: str) -> Dict[str, Any]:
    args = {"sequence": e.get("sequence") or ctx.get("last_sequence", "")}
    args.update(_model_args(e, "embed_sequence", "esm2"))
    return args


def _molecule_args(e: Dict[str, Any], ctx: Dict[str, Any], text: str) -> Dict[str, Any]:
    args = {
        "seed_smiles": e.get("smiles") or ctx.get("last_smiles", ""),
        "num_molecules": int(e.get("num_molecules") or 12),
        "property_name": e.get("property_name", "QED"),
        "minimize": bool(e.get("minimize", False)),
        "min_similarity": float(e.get("min_similarity", 0.3)),
    }
    args.update(_model_args(e, "generate_molecules", "molmim"))
    return args


def _dock_args(e: Dict[str, Any], ctx: Dict[str, Any], text: str) -> Dict[str, Any]:
    protein = e.get("pdb") or ctx.get("last_structure_pdb") or e.get("sequence", "")
    return {
        "protein": protein,
        "ligand": e.get("smiles") or ctx.get("last_smiles", ""),
        "num_poses": int(e.get("num_poses") or 10),
    }


def _design_args(e: Dict[str, Any], ctx: Dict[str, Any], text: str) -> Dict[str, Any]:
    args = {
        "pdb": e.get("pdb") or ctx.get("last_structure_pdb", ""),
        "num_sequences": int(e.get("num_sequences") or e.get("num_designs") or 8),
        "sampling_temp": float(e.get("temperature") or 0.1),
        "fixed_positions": e.get("fixed_positions", ""),
    }
    args.update(_model_args(e, "design_sequences", "proteinmpnn"))
    return args


def _dna_args(e: Dict[str, Any], ctx: Dict[str, Any], text: str) -> Dict[str, Any]:
    args = {
        "sequence": e.get("dna") or "ATG",
        "num_tokens": int(e.get("num_tokens") or 100),
        "temperature": float(e.get("temperature") or 0.7),
    }
    args.update(_model_args(e, "generate_dna", "evo2"))
    return args


def _analyse_seq_args(e: Dict[str, Any], ctx: Dict[str, Any], text: str) -> Dict[str, Any]:
    return {"sequence": e.get("sequence") or e.get("dna") or ctx.get("last_sequence", ""), "kind": "auto"}


def _analyse_mol_args(e: Dict[str, Any], ctx: Dict[str, Any], text: str) -> Dict[str, Any]:
    smiles = e.get("smiles_all") or ([e["smiles"]] if e.get("smiles") else [])
    return {"smiles_list": ", ".join(smiles)}


def _analyse_structure_args(e: Dict[str, Any], ctx: Dict[str, Any], text: str) -> Dict[str, Any]:
    return {"pdb": e.get("pdb") or ctx.get("last_structure_pdb", ""), "find_pockets": True}


def _compare_args(e: Dict[str, Any], ctx: Dict[str, Any], text: str) -> Dict[str, Any]:
    return {"pdb_a": e.get("pdb", ""), "pdb_b": ctx.get("previous_structure_pdb", "")}


# -- LLM planner (last resort) -------------------------------------------------


def _llm_plan(text: str, entities: Dict[str, Any], context: Dict[str, Any], llm: LLMClient) -> Plan:
    """Ask a model for one tool call, then validate it against the registry."""
    system = (
        "You choose tool calls for a BioNeMo protein/molecule console. Reply with a single "
        "JSON object: {\"tool\": <name>, \"arguments\": {...}, \"rationale\": <short string>}. "
        "Choose exactly one tool. Use only these tools and argument names:\n"
        + tools_mod.catalogue_for_prompt()
        + "\nIf the message is a question that needs no computation, reply {\"tool\": null}."
    )
    user = json.dumps(
        {
            "message": text,
            "detected_entities": {k: (v if not isinstance(v, str) or len(v) < 400 else v[:400] + "...")
                                  for k, v in entities.items()},
            "session_context": {
                "has_structure": bool(context.get("last_structure_pdb")),
                "has_sequence": bool(context.get("last_sequence")),
                "has_molecules": bool(context.get("last_smiles")),
            },
        },
        indent=2,
        default=str,
    )[:6000]
    reply = llm.complete([{"role": "system", "content": system}, {"role": "user", "content": user}],
                         json_mode=True, temperature=0.0)
    data = extract_json(reply)
    result = Plan(entities=entities)
    if not data or not data.get("tool"):
        result.chat_only = True
        result.planner = "llm:none"
        return result
    name = str(data["tool"])
    spec = tools_mod.TOOLS.get(name)
    if spec is None:
        result.chat_only = True
        result.notes.append(f"The planner proposed an unknown tool '{name}'; ignored.")
        return result
    args = data.get("arguments") or {}
    if not isinstance(args, dict):
        result.chat_only = True
        result.notes.append("The planner returned malformed arguments; ignored.")
        return result
    # Only keep declared parameters, and merge in anything the extractor found.
    clean = {k: v for k, v in args.items() if k in spec.parameters}
    for key in spec.requires:
        if not clean.get(key):
            inferred = {
                "sequence": entities.get("sequence"),
                "protein": entities.get("pdb") or entities.get("sequence"),
                "seed_smiles": entities.get("smiles"),
                "ligand": entities.get("smiles"),
                "pdb": entities.get("pdb"),
                "smiles_list": entities.get("smiles_all"),
            }.get(key)
            if inferred:
                clean[key] = inferred
    result.steps.append(ToolPlan(name, clean, 0.55, str(data.get("rationale", ""))[:200]))
    result.planner = "llm"
    _add_missing(result, name, clean, context)
    return result
