"""A geometric docking engine used by the simulated DiffDock backend.

This is not a learned docking model and the console never claims it is: it is a
physically motivated, fully inspectable search that produces *real* geometry --

* the ligand is embedded with RDKit's ETKDGv3 and optimised with MMFF94;
* pocket centres come from the console's own cavity detection;
* poses are sampled as rigid-body transforms (uniform SO(3) rotations, pocket
  centred translations) and refined by hill climbing on a transparent score:
  steric overlap, contact density, hydrophobic contact fraction, hydrogen bonds
  and ligand burial;
* every pose is reported with its contacts, H-bonds, buried fraction and a
  confidence mapped from the raw score, so the ranking can be argued with.

Every term in the score is normalised or saturating. That detail matters: a raw
contact count grows with ligand size and saturates inside a cavity, so without
normalisation every pose scores the same and the ranking carries no information.

Real DiffDock/Boltz-2 NIMs are one configuration switch away; this engine exists
so the console is useful on a laptop with no GPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..bio import structure as st_mod
from ..bio.chem import embed_3d, parse_smiles

#: van der Waals radii (A) for the elements we care about.
VDW = {
    "H": 1.10, "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "F": 1.47,
    "CL": 1.75, "BR": 1.85, "I": 1.98, "P": 1.80,
}

HYDROPHOBIC_ELEMENTS = {"C", "S", "F", "CL", "BR", "I"}
HBOND_DONORS = {"N", "O", "S"}
HBOND_ACCEPTORS = {"N", "O", "S", "F"}

#: Score constants (see :func:`_score_pose`): packing saturation point and
#: weights, all in the same arbitrary energy-like unit.
PACKING_MAX = 40.0
PACKING_SATURATION = 12.0     # contacts per ligand atom
HYDROPHOBIC_MAX = 25.0
HBOND_WEIGHT = 3.0
HBOND_CAP = 6
BURIAL_WEIGHT = 6.0
CLASH_WEIGHT = 8.0
OVERLAP_WEIGHT = 6.0
#: A fragment with few heavy atoms cannot establish a confident binding
#: hypothesis no matter how snugly it sits in a crevice, so the whole score is
#: scaled by ``1 - exp(-n_atoms / BREADTH_SCALE)`` (0.53 at 9 atoms, 0.83 at 21,
#: 0.97 at 40). This keeps "small molecule that touches everything" from
#: outranking a genuine hit; drug-likeness itself is a separate pipeline stage.
BREADTH_SCALE = 12.0
#: Confidence mapping: logistic( (score - CENTRE) * SLOPE ) on the scaled score.
#: Calibrated against the observed score range of the engine (roughly 9-32 for
#: drug-sized ligands in a detected cavity), so a good pose lands near 0.7 and a
#: poor one near 0.15 instead of every pose saturating at either end.
CONFIDENCE_CENTRE = 22.0
CONFIDENCE_SLOPE = 0.18


@dataclass
class Pose:
    """A single docking pose with everything needed to judge it."""

    rank: int
    coords: np.ndarray                     # (n_atoms, 3) heavy-atom coordinates
    score: float = 0.0
    confidence: float = 0.0
    contacts: int = 0
    clashes: int = 0
    hydrophobic_contacts: int = 0
    hbonds: int = 0
    buried_fraction: float = 0.0
    contacts_per_atom: float = 0.0
    pocket_id: str = ""
    pocket_centre: Optional[List[float]] = None
    interaction_residues: List[str] = field(default_factory=list)
    hbond_residues: List[str] = field(default_factory=list)
    rmsd_to_centroid: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "rank": self.rank,
            "score": round(float(self.score), 3),
            "confidence": round(float(self.confidence), 3),
            "contacts": int(self.contacts),
            "contacts_per_atom": round(float(self.contacts_per_atom), 2),
            "clashes": int(self.clashes),
            "hydrophobic_contacts": int(self.hydrophobic_contacts),
            "hbonds": int(self.hbonds),
            "buried_fraction": round(float(self.buried_fraction), 3),
            "pocket_id": self.pocket_id,
            "pocket_centre": self.pocket_centre,
            "interaction_residues": self.interaction_residues,
            "hbond_residues": self.hbond_residues,
            "rmsd_to_centroid": round(float(self.rmsd_to_centroid), 2),
        }


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Uniformly distributed rotation matrix (QR of a Gaussian matrix, det=+1)."""
    a = rng.normal(size=(3, 3))
    q, r = np.linalg.qr(a)
    q *= np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1.0
    return q


def _small_rotation(rng: np.random.Generator, degrees: float) -> np.ndarray:
    """A near-identity rotation, used to step the hill climber."""
    m = np.eye(3) + (rng.normal(size=(3, 3)) - np.eye(3)) * (math.radians(degrees) / math.pi)
    u, _s, vt = np.linalg.svd(m)
    rot = u @ vt
    if np.linalg.det(rot) < 0:
        rot[:, 0] *= -1
    return rot


def _ligand_atom_arrays(mol) -> Tuple[np.ndarray, List[str], List[bool], List[bool]]:
    conf = mol.GetConformer()
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())], dtype=float)
    elements = [a.GetSymbol().upper() for a in mol.GetAtoms()]
    donors = [
        bool(a.GetTotalNumHs() > 0 and a.GetSymbol().upper() in HBOND_DONORS) for a in mol.GetAtoms()
    ]
    acceptors = [
        a.GetSymbol().upper() in HBOND_ACCEPTORS and a.GetFormalCharge() <= 0 for a in mol.GetAtoms()
    ]
    return coords, elements, donors, acceptors


def _protein_arrays(structure: "st_mod.Structure"):
    coords, elements, residues, atom_names = [], [], [], []
    for res in structure.residues:
        for name, xyz in res.atoms.items():
            if name.startswith("H"):
                continue
            coords.append(xyz)
            elements.append(name[0].upper())
            residues.append(res)
            atom_names.append(name)
    return (
        np.array(coords, dtype=float) if coords else np.zeros((0, 3)),
        elements,
        residues,
        atom_names,
    )


def _score_pose(
    ligand_coords: np.ndarray,
    elements: Sequence[str],
    donors: Sequence[bool],
    acceptors: Sequence[bool],
    protein_coords: np.ndarray,
    protein_elements: Sequence[str],
    protein_residues: Sequence,
) -> Dict:
    """Transparent pose score: packing, hydrophobic contacts, H-bonds, burial, clashes."""
    if len(protein_coords) == 0 or len(ligand_coords) == 0:
        return {"score": -100.0, "contacts": 0, "clashes": 0, "hbonds": 0,
                "hydrophobic_contacts": 0, "buried": 0.0, "residues": [],
                "hbond_residues": [], "contacts_per_atom": 0.0}

    d = np.linalg.norm(ligand_coords[:, None, :] - protein_coords[None, :, :], axis=-1)
    r_lig = np.array([VDW.get(e, 1.7) for e in elements])
    r_prot = np.array([VDW.get(e, 1.7) for e in protein_elements])

    # Steric clash: overlap beyond 0.55 A of the summed vdW radii.
    overlap = (r_lig[:, None] + r_prot[None, :]) - d
    clash_mask = overlap > 0.55
    clashes = int(clash_mask.sum())
    overlap_sum = float(np.clip(overlap[clash_mask], 0, None).sum()) if clashes else 0.0

    contact_mask = d < 4.5
    contacts = int(contact_mask.sum())
    hydrophobic_contacts = 0
    for i in range(len(ligand_coords)):
        if elements[i] not in HYDROPHOBIC_ELEMENTS:
            continue
        for j in np.where(contact_mask[i])[0]:
            if protein_elements[j] in HYDROPHOBIC_ELEMENTS:
                hydrophobic_contacts += 1
    hydrophobic_fraction = hydrophobic_contacts / max(1, contacts)

    # Hydrogen bonds: donor/acceptor pairs at 2.4-3.6 A.
    hbonds = 0
    hbond_residues: List[str] = []
    for i in range(len(ligand_coords)):
        if not (donors[i] or acceptors[i]):
            continue
        near = np.where((d[i] > 2.4) & (d[i] < 3.6))[0]
        for j in near:
            if protein_elements[j] not in HBOND_DONORS:
                continue
            if (donors[i] and protein_elements[j] in HBOND_ACCEPTORS) or (
                acceptors[i] and protein_elements[j] in HBOND_DONORS
            ):
                hbonds += 1
                hbond_residues.append(f"{protein_residues[j].one_letter}{protein_residues[j].resseq}")

    # Burial: fraction of ligand atoms with >= 14 protein heavy atoms within 7 A.
    within7 = (d < 7.0).sum(1)
    buried = float((within7 >= 14).mean())

    residue_keys: List[str] = []
    seen = set()
    for j in np.where((d < 4.5).any(0))[0]:
        res = protein_residues[j]
        key = f"{res.one_letter}{res.resseq}"
        if key not in seen:
            seen.add(key)
            residue_keys.append(key)

    # --- pose score -----------------------------------------------------
    # Size-normalised and saturating by construction: a raw contact count grows
    # with ligand size and saturates inside a cavity, so an un-normalised score
    # would rank every pose identically.
    n_atoms = max(1, len(ligand_coords))
    contacts_per_atom = contacts / n_atoms
    packing = PACKING_MAX * min(1.0, contacts_per_atom / PACKING_SATURATION)
    hydrophobic_term = HYDROPHOBIC_MAX * hydrophobic_fraction * min(1.0, contacts_per_atom / 10.0)
    hbond_term = HBOND_WEIGHT * min(hbonds, HBOND_CAP)
    burial_term = BURIAL_WEIGHT * buried
    # Steric overlap is disqualifying: a pose that interpenetrates the protein
    # must never outrank a clean one, so clashes are penalised hard and linearly.
    clash_term = CLASH_WEIGHT * clashes + OVERLAP_WEIGHT * overlap_sum

    breadth = 1.0 - math.exp(-n_atoms / BREADTH_SCALE)
    score = (packing + hydrophobic_term + hbond_term + burial_term - clash_term) * breadth
    return {
        "score": float(score),
        "contacts": contacts,
        "clashes": clashes,
        "hbonds": hbonds,
        "hydrophobic_contacts": hydrophobic_contacts,
        "buried": buried,
        "contacts_per_atom": round(contacts_per_atom, 2),
        "breadth": round(breadth, 3),
        "residues": residue_keys[:24],
        "hbond_residues": sorted(set(hbond_residues))[:12],
    }


def _confidence_from_score(score: float, stats: Dict) -> float:
    """Map the raw score onto a 0-1 confidence.

    A logistic on the (already size-normalised) score, centred where a pose
    becomes credible -- packing saturated, buried, few clashes -- then attenuated
    by any remaining steric overlap. Deliberately monotone and documented so the
    number can be argued with rather than trusted blindly.
    """
    clashes = stats.get("clashes", 0)
    raw = (score - CONFIDENCE_CENTRE) * CONFIDENCE_SLOPE
    confidence = 1.0 / (1.0 + math.exp(-raw))
    if clashes:
        confidence *= math.exp(-clashes / 6.0)
    return float(max(0.0, min(1.0, confidence)))


def _refine(
    coords: np.ndarray,
    elements: Sequence[str],
    donors: Sequence[bool],
    acceptors: Sequence[bool],
    protein_coords: np.ndarray,
    protein_elements: Sequence[str],
    protein_residues: Sequence,
    centre: np.ndarray,
    rng: np.random.Generator,
    iterations: int = 120,
) -> Tuple[np.ndarray, Dict]:
    """Hill-climb on the rigid-body transform to improve the pose score."""
    best = coords.copy()
    best_stats = _score_pose(best, elements, donors, acceptors,
                             protein_coords, protein_elements, protein_residues)
    step_r, step_t = 6.0, 0.5
    for _ in range(iterations):
        rot = _small_rotation(rng, step_r)
        shift = rng.normal(0.0, step_t, size=3)
        candidate = (rot @ (best - centre).T).T + centre + shift
        stats = _score_pose(candidate, elements, donors, acceptors,
                            protein_coords, protein_elements, protein_residues)
        if stats["score"] > best_stats["score"]:
            best, best_stats = candidate, stats
        else:
            step_r = max(3.0, step_r * 0.97)
            step_t = max(0.15, step_t * 0.97)
    return best, best_stats


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------


def dock(
    protein: "st_mod.Structure",
    ligand_smiles: str,
    *,
    num_poses: int = 10,
    seed: int = 4242,
    refine_iterations: int = 120,
    pockets: Optional[List[Dict]] = None,
) -> Dict:
    """Dock ``ligand_smiles`` into ``protein`` and return ranked poses.

    Returns ``{"poses": [Pose], "pockets": [...], ...}`` with coordinates in the
    protein's frame, ready for the 3D viewer.
    """
    mol = embed_3d(ligand_smiles, n_confs=1, optimize="MMFF", seed=seed)
    if mol is None:
        return {"poses": [], "error": "Could not build 3D coordinates for the ligand.",
                "pockets": pockets or []}
    coords0, elements, donors, acceptors = _ligand_atom_arrays(mol)
    coords0 = coords0 - coords0.mean(0)

    protein_coords, protein_elements, protein_residues, _names = _protein_arrays(protein)
    if len(protein_coords) == 0:
        return {"poses": [], "error": "The protein has no atoms to dock against.",
                "pockets": pockets or []}

    if pockets is None:
        pockets = protein.find_pockets(top=max(3, min(6, num_poses // 2 or 3)))
    if not pockets:
        # No cavity detected: fall back to the chain centroid so the user still
        # gets an answer, and say so in the returned metadata.
        pockets = [
            {
                "id": "surface_fallback",
                "centre": [float(v) for v in protein.ca_coords().mean(0)],
                "volume_a3": 0.0,
                "druggability_hint": 0.0,
                "note": "no cavity detected; used the chain centroid",
            }
        ]

    rng = np.random.default_rng(seed)
    poses: List[Pose] = []
    ligand_radius = float(np.linalg.norm(coords0, axis=1).max())

    for pose_index in range(max(1, num_poses)):
        pocket = pockets[pose_index % len(pockets)]
        centre = np.array(pocket["centre"], dtype=float)
        # Place the ligand near the pocket centre with a spread comparable to the
        # ligand's own radius, which keeps every pose inside the cavity shell.
        offset = rng.normal(0.0, max(1.0, ligand_radius * 0.5), size=3)
        rot = _random_rotation(rng)
        start = (rot @ coords0.T).T + centre + offset
        refined, stats = _refine(
            start, elements, donors, acceptors,
            protein_coords, protein_elements, protein_residues,
            centre, rng, iterations=refine_iterations,
        )
        poses.append(
            Pose(
                rank=0,
                coords=refined,
                score=stats["score"],
                confidence=_confidence_from_score(stats["score"], stats),
                contacts=stats["contacts"],
                clashes=stats["clashes"],
                hydrophobic_contacts=stats["hydrophobic_contacts"],
                hbonds=stats["hbonds"],
                buried_fraction=stats["buried"],
                contacts_per_atom=stats.get("contacts_per_atom", 0.0),
                pocket_id=pocket.get("id", ""),
                pocket_centre=[round(float(x), 2) for x in centre],
                interaction_residues=stats["residues"],
                hbond_residues=stats.get("hbond_residues", []),
                rmsd_to_centroid=float(np.linalg.norm(refined.mean(0) - centre)),
            )
        )

    poses.sort(key=lambda p: -p.score)
    for i, pose in enumerate(poses, start=1):
        pose.rank = i

    diversity = 0.0
    if len(poses) > 1:
        d = [
            float(np.linalg.norm(poses[i].coords - poses[j].coords, axis=1).mean())
            for i in range(len(poses)) for j in range(i + 1, len(poses))
        ]
        diversity = float(np.mean(d)) if d else 0.0

    return {
        "poses": poses,
        "pockets": pockets,
        "ligand_atom_count": int(len(coords0)),
        "ligand_radius": round(ligand_radius, 2),
        "pose_diversity_mean_rmsd": round(diversity, 2),
    }


def poses_to_nim_format(poses: Sequence[Pose], element_count: int) -> Dict:
    """Flatten poses into the DiffDock response layout.

    NVIDIA's DiffDock NIM returns ``ligand_positions`` as one flat
    ``[x1, y1, z1, x2, y2, z2, ...]`` list per pose plus ``position_confidence``,
    so the simulated backend matches that exactly and the UI code path is
    identical for real and simulated runs.
    """
    flat = []
    conf = []
    for pose in poses:
        flat.append([round(float(v), 3) for v in pose.coords.reshape(-1)])
        conf.append(round(float(pose.confidence), 4))
    return {"ligand_positions": flat, "position_confidence": conf}


def unflatten_positions(flat: Sequence[float], n_atoms: int) -> np.ndarray:
    """Inverse of :func:`poses_to_nim_format` (also parses nested xyz lists)."""
    arr = np.asarray(flat, dtype=float)
    if arr.ndim == 2 and arr.shape[1] == 3:
        return arr
    return arr.reshape(-1, 3)[:n_atoms]


def score_existing_pose(
    protein: "st_mod.Structure",
    ligand_smiles: str,
    ligand_coords: np.ndarray,
) -> Dict:
    """Score an externally supplied pose (e.g. one returned by a real NIM)."""
    mol3d = embed_3d(ligand_smiles, optimize="MMFF")
    if mol3d is None:
        mol, _ = parse_smiles(ligand_smiles)
        if mol is None:
            return {}
        return {}
    _c, elements, donors, acceptors = _ligand_atom_arrays(mol3d)
    protein_coords, protein_elements, protein_residues, _ = _protein_arrays(protein)
    stats = _score_pose(ligand_coords, elements, donors, acceptors,
                        protein_coords, protein_elements, protein_residues)
    stats["confidence"] = _confidence_from_score(stats["score"], stats)
    return stats
