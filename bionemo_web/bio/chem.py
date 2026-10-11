"""Cheminformatics for the console.

Everything here is computed with RDKit, i.e. with the same algorithms a medicinal
chemist would use: Crippen logP, Ertl synthetic accessibility, TPSA, QED, Murcko
scaffolds, Morgan/ECFP Tanimoto similarity, Butina clustering, PAINS/Brenk
pan-assay-interference filters and ETKDG conformer generation.

The module also owns the *analogue generator* that the simulated MolMIM /
GenMol backends use. Rather than returning random strings, it performs real,
chemically valid edits on the seed molecule (BRICS recombination, R-group
enumeration, bioisosteric swaps) and filters the products by similarity, so a
simulated molecule-generation run produces molecules you could actually order.
"""

from __future__ import annotations

import math
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from rdkit import Chem, RDLogger

# RDKit is chatty about deprecated-but-working APIs; the console logs its own way.
RDLogger.DisableLog("rdApp.info")
RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.debug")

from rdkit.Chem import AllChem, BRICS, Crippen, Descriptors, Lipinski, QED, rdMolDescriptors
from rdkit.Chem import DataStructs, FilterCatalog, rdFMCS, rdMolAlign
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.SimDivFilters import rdSimDivPickers

try:  # Ertl's synthetic-accessibility score ships in RDKit's Contrib directory.
    from rdkit.Chem import RDConfig

    _sa_dir = os.path.join(RDConfig.RDContribDir, "SA_Score")
    if _sa_dir not in sys.path:
        sys.path.append(_sa_dir)
    import sascorer  # type: ignore

    _HAS_SA = True
except Exception:  # pragma: no cover - optional component
    _HAS_SA = False

# --------------------------------------------------------------------------
# Filters
# --------------------------------------------------------------------------

_FILTER_CATALOG: Optional[FilterCatalog.FilterCatalog] = None

#: Common warhead / reactive-group SMARTS that flag a compound as problematic.
REACTIVE_SMARTS: List[Tuple[str, str]] = [
    ("Micheal acceptor", "[CX3]=[CX3][CX3]=[OX1]"),
    ("Acyl halide", "[CX3](=O)[F,Cl,Br,I]"),
    ("Epoxide", "[OX2]1[CX4][CX4]1"),
    ("Aziridine", "[NX3]1[CX4][CX4]1"),
    ("Aldehyde", "[CX3H1](=O)[#6]"),
    ("Alkyl halide", "[CX4][Cl,Br,I]"),
    ("Nitroso", "[NX2]=[OX1]"),
    ("Azo", "[#6][NX2]=[NX2][#6]"),
    ("Peroxide", "[OX2][OX2]"),
    ("Sulfonyl halide", "S(=O)(=O)[F,Cl,Br,I]"),
    ("Isocyanate", "[NX2]=[CX2]=[OX1]"),
    ("Anhydride", "[CX3](=O)[OX2][CX3](=O)"),
    ("Thiol", "[SX2H]"),
    ("Hydrazine", "[NX3][NX3]"),
]


def _catalog(name: str):
    """Build (and cache) one RDKit filter catalog by name."""
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(getattr(FilterCatalog.FilterCatalogParams.FilterCatalogs, name))
    return FilterCatalog.FilterCatalog(params)


_CATALOGS: Dict[str, Any] = {}


def filter_hits(mol: Chem.Mol) -> List[Dict[str, str]]:
    """Structural alerts, with severity.

    The three RDKit catalogs mean different things, and collapsing them into one
    "alert" bucket is how tools end up flagging aspirin (which Brenk lists for its
    labile phenol ester) as if it were a screening artefact. So:

    ``alert``  PAINS -- frequent-hitter / assay-interference risk. Treat seriously.
    ``note``   Brenk & NIH -- undesirable functionality; often present in drugs.
    ``warn``   the local reactive-group list (peroxides, isocyanates, ...).
    """
    hits: List[Dict[str, str]] = []
    if mol is None:
        return hits
    severity = {"PAINS": "alert", "BRENK": "note", "NIH": "note"}
    for name in ("PAINS", "BRENK", "NIH"):
        if name not in _CATALOGS:
            try:
                _CATALOGS[name] = _catalog(name)
            except Exception:  # pragma: no cover - RDKit without that catalog
                _CATALOGS[name] = None
        catalog = _CATALOGS[name]
        if catalog is None:
            continue
        try:
            for match in catalog.GetMatches(mol):
                hits.append(
                    {
                        "source": name.lower(),
                        "name": match.GetDescription(),
                        "severity": severity[name],
                        "note": _CATALOG_NOTES.get(match.GetDescription(), ""),
                    }
                )
        except Exception:  # pragma: no cover
            continue
    for name, smarts in REACTIVE_SMARTS:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is not None and mol.HasSubstructMatch(pattern):
            hits.append({"source": "reactive-group", "name": name, "severity": "warn", "note": ""})
    return hits


#: Explanations for the catalog entries users ask about most.
_CATALOG_NOTES = {
    "phenol_ester": "labile ester; the alert is about stability, not about being a bad drug "
                    "(aspirin and many prodrugs are flagged)",
    "quinone": "redox cycler and frequent-hitter risk",
    "michael_acceptor": "covalent modifier risk if the alkene is unsubstituted",
    "catechol": "metal chelation and assay interference",
}


#: A small dictionary of molecules people type by name.
#:
#: It exists because "dock aspirin into this protein" is the first thing a
#: chemist types, and requiring them to paste a SMILES for a household drug is
#: needless friction. Canonical SMILES are from the public ChEMBL/PubChem
#: records; the list is deliberately short and unambiguous -- anything
#: unfamiliar stays unresolved rather than being guessed at.
COMMON_MOLECULES: Dict[str, str] = {
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "acetylsalicylic acid": "CC(=O)Oc1ccccc1C(=O)O",
    "ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "acetaminophen": "CC(=O)Nc1ccc(O)cc1",
    "caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "nicotine": "CN1CCCC1c1cccnc1",
    "ethanol": "CCO",
    "methanol": "CO",
    "glucose": "OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O",
    "fructose": "OC[C@@H](O)[C@@H](O)[C@H](O)C(=O)CO",
    "sucrose": "OC[C@H]1O[C@@](CO)(O[C@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)[C@@H](O)[C@@H]1O",
    "penicillin g": "CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O",
    "amoxicillin": "CC1(C)SC2C(NC(=O)C(N)c3ccc(O)cc3)C(=O)N2C1C(=O)O",
    "ciprofloxacin": "O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O",
    "metformin": "CN(C)C(=N)NC(=N)N",
    "warfarin": "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O",
    "diazepam": "CN1c2ccc(Cl)cc2C(c2ccccc2)=NCC1=O",
    "morphine": "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5",
    "dopamine": "NCCc1ccc(O)c(O)c1",
    "serotonin": "NCCc1c[nH]c2ccc(O)cc12",
    "histamine": "NCCc1c[nH]cn1",
    "adrenaline": "CNC[C@H](O)c1ccc(O)c(O)c1",
    "epinephrine": "CNC[C@H](O)c1ccc(O)c(O)c1",
    "urea": "NC(N)=O",
    "glycerol": "OCC(O)CO",
    "benzene": "c1ccccc1",
    "toluene": "Cc1ccccc1",
    "phenol": "Oc1ccccc1",
    "aniline": "Nc1ccccc1",
    "pyridine": "c1ccncc1",
    "imidazole": "c1c[nH]cn1",
    "indole": "c1ccc2[nH]ccc2c1",
    "naphthalene": "c1ccc2ccccc2c1",
    "urea cycle": "NC(N)=O",
}


def smiles_from_name(name: str) -> Optional[str]:
    """Resolve a common molecule name to a SMILES, or ``None`` if unknown."""
    if not name:
        return None
    key = str(name).strip().lower().replace("-", " ")
    return COMMON_MOLECULES.get(key)


# --------------------------------------------------------------------------
# Descriptors
# --------------------------------------------------------------------------

RULE_OF_FIVE = {
    "mw": (">", 500.0, "Molecular weight (Da)"),
    "clogp": (">", 5.0, "Crippen logP"),
    "hbd": (">", 5, "H-bond donors"),
    "hba": (">", 10, "H-bond acceptors"),
}

VEBER_RULES = {"rotatable_bonds": (">", 10, "Rotatable bonds"), "tpsa": (">", 140.0, "TPSA (A^2)")}


@dataclass
class MoleculeReport:
    """Descriptor + rule-of-thumb report for one molecule."""

    smiles: str
    canonical_smiles: str = ""
    name: str = ""
    formula: str = ""
    inchi_key: str = ""
    mw: float = 0.0
    exact_mass: float = 0.0
    clogp: float = 0.0
    tpsa: float = 0.0
    hbd: int = 0
    hba: int = 0
    rotatable_bonds: int = 0
    rings: int = 0
    aromatic_rings: int = 0
    heavy_atoms: int = 0
    fraction_csp3: float = 0.0
    formal_charge: int = 0
    qed: float = 0.0
    sa_score: Optional[float] = None
    molar_refractivity: float = 0.0
    num_stereocenters: int = 0
    num_amide_bonds: int = 0
    num_heteroatoms: int = 0
    logp_class: str = ""
    violations: List[str] = field(default_factory=list)
    ro5_pass: bool = True
    veber_pass: bool = True
    filter_alerts: List[Dict[str, str]] = field(default_factory=list)
    murcko_scaffold: str = ""
    generic_scaffold: str = ""
    fragments: List[str] = field(default_factory=list)
    total_fragments: int = 0
    brenk_or_pains: int = 0
    predicted_solubility_logs: float = 0.0
    ligand_efficiency_hint: float = 0.0
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "smiles": self.smiles,
            "canonical_smiles": self.canonical_smiles,
            "name": self.name,
            "formula": self.formula,
            "inchi_key": self.inchi_key,
            "mw": round(self.mw, 2),
            "exact_mass": round(self.exact_mass, 4),
            "clogp": round(self.clogp, 3),
            "tpsa": round(self.tpsa, 2),
            "hbd": self.hbd,
            "hba": self.hba,
            "rotatable_bonds": self.rotatable_bonds,
            "rings": self.rings,
            "aromatic_rings": self.aromatic_rings,
            "heavy_atoms": self.heavy_atoms,
            "fraction_csp3": round(self.fraction_csp3, 3),
            "formal_charge": self.formal_charge,
            "qed": round(self.qed, 4),
            "sa_score": round(self.sa_score, 2) if self.sa_score is not None else None,
            "molar_refractivity": round(self.molar_refractivity, 2),
            "stereocenters": self.num_stereocenters,
            "amide_bonds": self.num_amide_bonds,
            "heteroatoms": self.num_heteroatoms,
            "ro5_pass": self.ro5_pass,
            "veber_pass": self.veber_pass,
            "violations": self.violations,
            "filter_alerts": self.filter_alerts,
            "murcko_scaffold": self.murcko_scaffold,
            "generic_scaffold": self.generic_scaffold,
            "fragments": self.fragments[:40],
            "predicted_solubility_logs": round(self.predicted_solubility_logs, 2),
            "warnings": self.warnings,
        }


def parse_smiles(text: str) -> Tuple[Optional[Chem.Mol], List[str]]:
    """Parse a SMILES/SDF/InChI-ish string tolerantly; returns (mol, warnings)."""
    warnings: List[str] = []
    if not text:
        return None, ["Empty input."]
    raw = text.strip()
    if raw.startswith("InChI="):
        mol = Chem.MolFromInchi(raw)
        if mol is None:
            warnings.append("Could not parse the InChI string.")
        return mol, warnings
    if "\n" in raw and ("V2000" in raw or "V3000" in raw or raw.startswith("$$$$")):
        supplier = Chem.SDMolSupplier()
        supplier.SetData(raw, sanitize=True, removeHs=False)
        mols = [m for m in supplier if m is not None]
        if mols:
            return mols[0], warnings
        warnings.append("SDF block present but no molecule could be read from it.")
    mol = Chem.MolFromSmiles(raw)
    if mol is None:
        relaxed = Chem.MolFromSmiles(raw, sanitize=False)
        if relaxed is not None:
            try:
                Chem.SanitizeMol(relaxed, catchErrors=True)
                mol = relaxed
                warnings.append("SMILES required relaxed sanitisation; check valence/stereochemistry.")
            except Exception:
                pass
    if mol is None:
        warnings.append("Invalid SMILES: RDKit could not parse the molecule.")
    return mol, warnings


def report_molecule(smiles: str, name: str = "") -> MoleculeReport:
    """Full descriptor report for a SMILES string."""
    mol, warnings = parse_smiles(smiles)
    rep = MoleculeReport(smiles=smiles, name=name or smiles[:40], warnings=list(warnings))
    if mol is None:
        return rep

    rep.canonical_smiles = Chem.MolToSmiles(mol)
    rep.formula = rdMolDescriptors.CalcMolFormula(mol)
    try:
        rep.inchi_key = Chem.MolToInchiKey(mol)
    except Exception:
        rep.inchi_key = ""
    rep.mw = Descriptors.MolWt(mol)
    rep.exact_mass = Descriptors.ExactMolWt(mol)
    rep.clogp = Crippen.MolLogP(mol)
    rep.tpsa = rdMolDescriptors.CalcTPSA(mol)
    rep.hbd = Lipinski.NumHDonors(mol)
    rep.hba = Lipinski.NumHAcceptors(mol)
    rep.rotatable_bonds = Lipinski.NumRotatableBonds(mol)
    rep.rings = rdMolDescriptors.CalcNumRings(mol)
    rep.aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    rep.heavy_atoms = mol.GetNumHeavyAtoms()
    rep.fraction_csp3 = rdMolDescriptors.CalcFractionCSP3(mol)
    rep.formal_charge = Chem.GetFormalCharge(mol)
    rep.qed = QED.qed(mol)
    rep.molar_refractivity = Crippen.MolMR(mol)
    rep.num_stereocenters = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False))
    rep.num_amide_bonds = rdMolDescriptors.CalcNumAmideBonds(mol)
    rep.num_heteroatoms = rdMolDescriptors.CalcNumHeteroatoms(mol)
    if _HAS_SA:
        try:
            rep.sa_score = sascorer.calculateScore(mol)
        except Exception:
            rep.sa_score = None
    rep.murcko_scaffold = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol)) if mol.GetNumAtoms() > 2 else ""
    try:
        rep.generic_scaffold = Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(mol)) if mol.GetNumAtoms() > 2 else ""
    except Exception:
        rep.generic_scaffold = ""

    for frag in BRICS.BRICSDecompose(mol):
        # BRICSDecompose yields SMILES strings in current RDKit releases.
        rep.fragments.append(frag[1] if isinstance(frag, tuple) else frag)
    rep.fragments = sorted(set(rep.fragments))
    rep.total_fragments = len(rep.fragments)

    rep.filter_alerts = filter_hits(mol)
    rep.brenk_or_pains = sum(1 for h in rep.filter_alerts if h["source"] == "catalog")

    # Rule sets -------------------------------------------------------
    checks = {
        "mw": rep.mw,
        "clogp": rep.clogp,
        "hbd": rep.hbd,
        "hba": rep.hba,
        "rotatable_bonds": rep.rotatable_bonds,
        "tpsa": rep.tpsa,
    }
    for key, (op, limit, label) in RULE_OF_FIVE.items():
        if checks[key] > limit:
            rep.violations.append(f"Lipinski: {label} = {checks[key]:.2f} > {limit}")
    rep.ro5_pass = len(rep.violations) == 0
    for key, (op, limit, label) in VEBER_RULES.items():
        if checks[key] > limit:
            rep.veber_pass = False
            rep.violations.append(f"Veber: {label} = {checks[key]:.2f} > {limit}")

    # A well-known transparent solubility heuristic (ESOL-style):
    # logS ~= 0.16 - 0.63*clogP - 0.0062*MW + 0.066*RB - 0.74*(aromatic proportion)
    aromatic_fraction = (
        sum(int(a.GetIsAromatic()) for a in mol.GetAtoms()) / max(1, mol.GetNumHeavyAtoms())
    )
    rep.predicted_solubility_logs = (
        0.16 - 0.63 * rep.clogp - 0.0062 * rep.mw + 0.066 * rep.rotatable_bonds - 0.74 * aromatic_fraction
    )
    # Ligand-efficiency style hint (QED per heavy atom, scaled) -- a screen, not a promise.
    rep.ligand_efficiency_hint = rep.qed / max(1, rep.heavy_atoms) * 10
    rep.logp_class = (
        "very hydrophobic" if rep.clogp > 4 else "hydrophobic" if rep.clogp > 2 else
        "balanced" if rep.clogp > 0 else "hydrophilic"
    )
    return rep


def descriptors_frame(smiles_list: Sequence[str]) -> List[Dict]:
    """Batch descriptor table, skipping unparsable entries."""
    out = []
    for smi in smiles_list:
        rep = report_molecule(smi)
        if rep.canonical_smiles:
            out.append(rep.to_dict())
    return out


# --------------------------------------------------------------------------
# Similarity, clustering, scaffold analysis
# --------------------------------------------------------------------------


def fingerprint(mol: Chem.Mol, kind: str = "morgan", radius: int = 2, n_bits: int = 2048):
    """Return an RDKit fingerprint object (``morgan`` | ``rdkit`` | ``maccs`` | ``atom-pair``)."""
    kind = (kind or "morgan").lower()
    if kind in ("morgan", "ecfp", "ecfp4"):
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    if kind in ("rdkit", "rdk"):
        return Chem.RDKFingerprint(mol, fpSize=n_bits)
    if kind == "maccs":
        from rdkit.Chem import MACCSkeys

        return MACCSkeys.GenMACCSKeys(mol)
    if kind in ("atom-pair", "atompair", "ap"):
        return rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=n_bits)
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def tanimoto(smiles_a: str, smiles_b: str, kind: str = "morgan") -> Optional[float]:
    a, _ = parse_smiles(smiles_a)
    b, _ = parse_smiles(smiles_b)
    if a is None or b is None:
        return None
    fp = fingerprint(a, kind)
    fp2 = fingerprint(b, kind)
    if kind == "morgan":
        return DataStructs.TanimotoSimilarity(fp, fp2)
    return DataStructs.TanimotoSimilarity(fp, fp2)


def similarity_matrix(smiles_list: Sequence[str], kind: str = "morgan") -> Tuple[List[str], List[List[float]]]:
    mols = []
    labels = []
    for smi in smiles_list:
        m, _ = parse_smiles(smi)
        if m is not None:
            mols.append(m)
            labels.append(Chem.MolToSmiles(m))
    fps = [fingerprint(m, kind) for m in mols]
    n = len(fps)
    matrix = [[1.0 if i == j else DataStructs.TanimotoSimilarity(fps[i], fps[j]) for j in range(n)] for i in range(n)]
    return labels, matrix


def cluster_molecules(smiles_list: Sequence[str], threshold: float = 0.35, kind: str = "morgan") -> List[Dict]:
    """Butina clustering on a Tanimoto distance matrix (Taylor-Butina algorithm)."""
    mols, labels = [], []
    for smi in smiles_list:
        m, _ = parse_smiles(smi)
        if m is not None:
            mols.append(m)
            labels.append(Chem.MolToSmiles(m))
    n = len(mols)
    if n == 0:
        return []
    if n == 1:
        return [{"cluster": 1, "size": 1, "members": labels, "centroid": labels[0]}]
    fps = [fingerprint(m, kind) for m in mols]
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(1.0 - DataStructs.TanimotoSimilarity(fps[i], fps[j]))
    picker = rdSimDivPickers.LeaderPicker()
    try:
        picks = picker.LazyBitVectorPick(fps, n, threshold)
    except Exception:
        picks = []
    clusters: Dict[int, List[int]] = {}
    if not picks:
        # Fallback: greedy single-linkage over the distance matrix.
        picks = list(range(min(n, max(1, n // 4))))
    for idx in range(n):
        best, best_d = None, 2.0
        for c, rep in enumerate(picks):
            d = 1.0 - DataStructs.TanimotoSimilarity(fps[idx], fps[rep])
            if d < best_d:
                best, best_d = c, d
        clusters.setdefault(best, []).append(idx)
    out = []
    for rank, (_c, members) in enumerate(sorted(clusters.items(), key=lambda kv: -len(kv[1])), start=1):
        out.append(
            {
                "cluster": rank,
                "size": len(members),
                "members": [labels[i] for i in members],
                "centroid": labels[members[0]],
                "mean_intra_similarity": round(
                    sum(
                        1 - DataStructs.TanimotoSimilarity(fps[i], fps[j])
                        for i in members for j in members if i < j
                    ) / max(1, len(members) * (len(members) - 1) / 2),
                    3,
                ),
            }
        )
    return out


def mcs_similarity(smiles_a: str, smiles_b: str, timeout: int = 5) -> Dict:
    """Maximum common substructure between two molecules."""
    a, _ = parse_smiles(smiles_a)
    b, _ = parse_smiles(smiles_b)
    if a is None or b is None:
        return {}
    res = rdFMCS.FindMCS([a, b], timeout=timeout, completeRingsOnly=False, ringMatchesRingOnly=False)
    return {
        "smarts": res.smartsString,
        "num_atoms": res.numAtoms,
        "num_bonds": res.numBonds,
        "fraction_a": res.numAtoms / max(1, a.GetNumHeavyAtoms()),
        "fraction_b": res.numAtoms / max(1, b.GetNumHeavyAtoms()),
    }


def scaffold_table(smiles_list: Sequence[str], top: int = 25) -> List[Dict]:
    """Group molecules by Murcko scaffold, most populated first."""
    groups: Dict[str, List[str]] = {}
    for smi in smiles_list:
        m, _ = parse_smiles(smi)
        if m is None:
            continue
        try:
            scaf = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m))
        except Exception:
            scaf = ""
        groups.setdefault(scaf or "(acyclic)", []).append(Chem.MolToSmiles(m))
    rows = [
        {"scaffold": k, "count": len(v), "members": v[:12]} for k, v in groups.items()
    ]
    rows.sort(key=lambda r: -r["count"])
    return rows[:top]


# --------------------------------------------------------------------------
# Analogue generation (used by the simulated MolMIM / GenMol backends)
# --------------------------------------------------------------------------

#: Small, drug-like R-groups appended to an attachment point.
R_GROUPS = ["F", "Cl", "C", "O", "N", "C#N", "OC", "C(F)(F)F", "NC(=O)C", "S(C)(=O)=O", "N(C)C", "CO"]

#: Bioisosteric replacement pairs (as SMARTS-free SMILES substrings on the R-group level).
_BIOISOSTERES = [
    ("c1ccccc1", "c1ccncc1"),      # benzene -> pyridine
    ("C(=O)O", "C(=O)N"),          # acid -> amide
    ("O", "N"),                    # ether -> amine
    ("F", "Cl"),
    ("C", "N"),
]

MAX_ANALOGUES = 600


def _substituent_sites(mol: Chem.Mol) -> List[str]:
    """SMARTS for positions where a hydrogen can reasonably be replaced."""
    return [
        "[cH]",
        "[CH3]",
        "[CH2]",
        "[NH2]",
        "[nH]",
        "[OH]",
    ]


def _safe(smi: str) -> Optional[Chem.Mol]:
    if not smi:
        return None
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    try:
        Chem.SanitizeMol(m, catchErrors=True)
    except Exception:
        return None
    return m


def enumerate_r_group_analogues(seed: Chem.Mol, max_products: int = 150) -> List[Chem.Mol]:
    """Attach small R-groups at every substituent site, one edit at a time."""
    products: List[Chem.Mol] = []
    for smarts in _substituent_sites(seed):
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            continue
        for atom_idx in seed.GetSubstructMatches(patt)[:6]:
            for grp in R_GROUPS:
                try:
                    combo = Chem.RWMol(seed)
                    target = combo.GetAtomWithIdx(atom_idx[0])
                    if target.GetTotalNumHs() == 0:
                        continue
                    frag = Chem.MolFromSmiles(grp)
                    if frag is None:
                        continue
                    merged = Chem.RWMol(Chem.CombineMols(combo, frag))
                    new_idx = combo.GetNumAtoms()
                    # Bond the fragment's first atom to the target atom.
                    merged.AddBond(atom_idx[0], new_idx, Chem.BondType.SINGLE)
                    m = merged.GetMol()
                    Chem.SanitizeMol(m, catchErrors=True)
                    if m.GetNumAtoms() == seed.GetNumAtoms():
                        continue
                    products.append(m)
                except Exception:
                    continue
                if len(products) >= max_products:
                    return products
    return products


def brics_analogues(seed: Chem.Mol, max_products: int = 200, max_depth: int = 2) -> List[Chem.Mol]:
    """Recombine the seed's own BRICS fragments into new molecules."""
    try:
        pieces = list(BRICS.BRICSDecompose(seed))
    except Exception:
        return []
    if not pieces:
        return []
    frag_mols = [f for f in (Chem.MolFromSmiles(p) for p in pieces) if f is not None]
    if not frag_mols:
        return []
    try:
        builder = BRICS.BRICSBuild(frag_mols, maxDepth=max_depth)
    except Exception:
        return []
    out: List[Chem.Mol] = []
    for mol in builder:
        try:
            mol.UpdatePropertyCache(strict=False)
            Chem.SanitizeMol(mol, catchErrors=True)
            if mol.GetNumHeavyAtoms() == 0:
                continue
            out.append(mol)
        except Exception:
            continue
        if len(out) >= max_products:
            break
    return out


def bioisostere_analogues(seed: Chem.Mol, max_products: int = 80) -> List[Chem.Mol]:
    """Whole-molecule bioisosteric swaps that stay synthetically plausible."""
    out: List[Chem.Mol] = []
    smi = Chem.MolToSmiles(seed)
    for a, b in _BIOISOSTERES:
        for frm, to in ((a, b), (b, a)):
            if frm in smi:
                cand = _safe(smi.replace(frm, to, 1))
                if cand is not None and cand.GetNumHeavyAtoms() > 3:
                    out.append(cand)
            if len(out) >= max_products:
                return out
    return out


def generate_analogues(
    seed_smiles: str,
    n: int = 12,
    *,
    min_similarity: float = 0.3,
    max_similarity: float = 1.0,
    seed: int = 0xC0FFEE,
    max_heavy_atoms: int = 60,
    property_name: str = "QED",
    minimize: bool = False,
    scaffold_hop: bool = False,
) -> List[Dict]:
    """Generate ranked, chemically valid analogues of ``seed_smiles``.

    Returns dicts with the SMILES, descriptor report, similarity to the seed and
    the optimisation score.  Products are deduplicated by canonical SMILES.
    """
    seed_mol, warnings = parse_smiles(seed_smiles)
    if seed_mol is None:
        return []
    rng = random.Random(seed)

    candidates: List[Chem.Mol] = []
    candidates.extend(brics_analogues(seed_mol, max_products=260, max_depth=2 if not scaffold_hop else 3))
    candidates.extend(enumerate_r_group_analogues(seed_mol, max_products=220))
    candidates.extend(bioisostere_analogues(seed_mol))
    rng.shuffle(candidates)

    seed_fp = fingerprint(seed_mol)
    seen: Dict[str, Dict] = {}
    for mol in candidates:
        if mol is None or mol.GetNumHeavyAtoms() > max_heavy_atoms:
            continue
        try:
            can = Chem.MolToSmiles(mol)
        except Exception:
            continue
        if can in seen:
            continue
        try:
            sim = DataStructs.TanimotoSimilarity(seed_fp, fingerprint(mol))
        except Exception:
            continue
        if sim < min_similarity or sim > max_similarity:
            continue
        rep = report_molecule(can)
        if not rep.canonical_smiles:
            continue
        score = _property_value(rep, property_name)
        seen[can] = {
            "smiles": can,
            "similarity": round(sim, 4),
            "score": round(score, 4),
            "property": property_name,
            "mmff_energy": None,
            "report": rep.to_dict(),
        }
        if len(seen) >= MAX_ANALOGUES:
            break

    ranked = sorted(seen.values(), key=lambda d: d["score"], reverse=not minimize)[: max(1, n)]
    for i, row in enumerate(ranked, start=1):
        row["rank"] = i
        row["generation_seed"] = seed
    if not ranked:
        # Guarantee a non-empty answer: report the seed itself as rank 0.
        rep = report_molecule(Chem.MolToSmiles(seed_mol))
        ranked = [
            {
                "smiles": rep.canonical_smiles,
                "similarity": 1.0,
                "score": round(_property_value(rep, property_name), 4),
                "property": property_name,
                "rank": 1,
                "generation_seed": seed,
                "note": "no analogues satisfied the similarity window; the seed is shown",
                "report": rep.to_dict(),
            }
        ]
    return ranked


def _property_value(rep: MoleculeReport, property_name: str) -> float:
    prop = (property_name or "QED").upper()
    if prop == "QED":
        return rep.qed
    if prop in ("PLOGP", "LOGP", "CLOGP"):
        # plogP-style objective: penalise both extremes of logP.
        return -abs(rep.clogp - 2.5)
    if prop in ("SA", "SASCORE", "SYNTH"):
        return -(rep.sa_score if rep.sa_score is not None else 5.0)
    if prop in ("TPSA",):
        return -abs(rep.tpsa - 75.0)
    if prop in ("MW",):
        return -abs(rep.mw - 350.0)
    if prop in ("SOLUBILITY", "LOGS"):
        return rep.predicted_solubility_logs
    return rep.qed


# --------------------------------------------------------------------------
# 3D geometry (shared by molecule viewer + docking engine)
# --------------------------------------------------------------------------


def embed_3d(
    smiles: str,
    *,
    n_confs: int = 1,
    optimize: str = "MMFF",
    seed: int = 42,
    max_attempts: int = 8,
) -> Optional[Chem.Mol]:
    """Generate and (optionally) optimise 3D conformers with ETKDGv3."""
    mol, _ = parse_smiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed) % (2**31 - 1)
    params.enforceChirality = True
    params.useSmallRingTorsions = True
    params.pruneRmsThresh = 0.25
    conf_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=max(1, n_confs), params=params))
    if not conf_ids:
        for attempt in range(max_attempts):
            params.randomSeed = (int(seed) + attempt * 7919) % (2**31 - 1)
            conf_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=max(1, n_confs), params=params))
            if conf_ids:
                break
    if not conf_ids:
        return None
    try:
        if optimize == "MMFF" and AllChem.MMFFHasAllMoleculeParams(mol):
            AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=400, numThreads=1)
        elif optimize == "UFF":
            AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=400, numThreads=1)
    except Exception:
        pass
    return mol


def mol_to_pdb_block(mol: Chem.Mol, conf_id: int = 0, res_name: str = "LIG") -> str:
    """Minimal PDB writer so ligands can be handed to PDB-consuming tools."""
    conf = mol.GetConformer(conf_id) if mol.GetNumConformers() else None
    lines = []
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx()) if conf is not None else None
        x, y, z = (pos.x, pos.y, pos.z) if pos is not None else (0.0, 0.0, 0.0)
        elem = atom.GetSymbol().upper()
        name = f"{elem}{(atom.GetIdx() + 1) % 1000:>3}"
        lines.append(
            "HETATM{serial:5d} {name:<4}{alt:1}{res:>3} {chain:1}{resid:4d}    "
            "{x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{b:6.2f}          {elem:>2}".format(
                serial=(atom.GetIdx() + 1) % 100000,
                name=name[:4],
                alt=" ",
                res=res_name[:3],
                chain="L",
                resid=1,
                x=x, y=y, z=z, occ=1.0, b=0.0, elem=elem,
            )
        )
    return "\n".join(lines) + "\nEND\n"


def mol_block_3d(smiles: str, *, seed: int = 42, optimize: str = "MMFF") -> Optional[str]:
    """Return a 3D MOL block (for 3Dmol.js) or None."""
    mol = embed_3d(smiles, seed=seed, optimize=optimize)
    if mol is None:
        return None
    return Chem.MolToMolBlock(mol, confId=0)


def summarise_set(smiles_list: Sequence[str]) -> Dict:
    """Aggregate property statistics for a molecule set (used in dashboards)."""
    reports = [r for r in (report_molecule(s) for s in smiles_list) if r.canonical_smiles]
    if not reports:
        return {"n": 0}
    numeric = lambda key: [getattr(r, key) for r in reports if getattr(r, key) is not None]  # noqa: E731
    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0  # noqa: E731
    return {
        "n": len(reports),
        "mean_mw": round(mean(numeric("mw")), 1),
        "mean_clogp": round(mean(numeric("clogp")), 2),
        "mean_tpsa": round(mean(numeric("tpsa")), 1),
        "mean_qed": round(mean(numeric("qed")), 3),
        "mean_sa": round(mean([r.sa_score for r in reports if r.sa_score]), 2),
        "ro5_pass_rate": round(sum(r.ro5_pass for r in reports) / len(reports), 3),
        "veber_pass_rate": round(sum(r.veber_pass for r in reports) / len(reports), 3),
        "mean_solubility_logs": round(mean([r.predicted_solubility_logs for r in reports]), 2),
        "unique_scaffolds": len({r.murcko_scaffold for r in reports}),
        "alerts": sum(
            1 for r in reports for h in (r.filter_alerts or []) if h.get("severity") == "alert"
        ),
        "notes": sum(
            1 for r in reports for h in (r.filter_alerts or []) if h.get("severity") in ("note", "warn")
        ),
        "mol_weight_range": [round(min(numeric("mw")), 1), round(max(numeric("mw")), 1)],
        "logp_range": [round(min(numeric("clogp")), 2), round(max(numeric("clogp")), 2)],
    }
