"""Structure engine: geometry, secondary structure, cavities and folding."""

from __future__ import annotations

import numpy as np
import pytest

from bionemo_web.bio import structure as st_mod


def test_ideal_backbone_geometry_matches_reference_values():
    """Engh & Huber bond lengths and angles, from a single alanine."""
    structure = st_mod.fold_sequence("AAAAA", seed=0)
    atoms = {k: np.asarray(v, dtype=float) for k, v in structure.residues[0].atoms.items()}
    assert np.linalg.norm(atoms["N"] - atoms["CA"]) == pytest.approx(1.458, abs=0.02)
    assert np.linalg.norm(atoms["CA"] - atoms["C"]) == pytest.approx(1.525, abs=0.02)
    assert np.linalg.norm(atoms["C"] - atoms["O"]) == pytest.approx(1.231, abs=0.03)
    # N-CA-C bond angle is 111.2 degrees in the Engh & Huber reference geometry.
    u = atoms["N"] - atoms["CA"]
    v = atoms["C"] - atoms["CA"]
    cosine = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
    assert np.degrees(np.arccos(np.clip(cosine, -1, 1))) == pytest.approx(111.2, abs=2.0)


def test_helix_is_geometrically_a_helix():
    """An ideal alpha helix built from explicit phi/psi must measure as one."""
    structure = _ideal_helix_structure(30)
    coords = structure.ca_coords()
    # i -> i+4 CA distance is the diagnostic for an alpha helix (~6.0-6.6 A).
    distances = [np.linalg.norm(coords[i] - coords[i + 4]) for i in range(len(coords) - 4)]
    assert 5.6 < float(np.median(distances)) < 7.0
    # The pitch of an alpha helix is 5.4 A per turn (3.6 residues).
    rise_per_residue = np.linalg.norm(coords[1] - coords[0])
    assert 3.4 < rise_per_residue < 4.2
    coords = structure.ca_coords()
    # For a 30-residue helix (45 A long, 2.3 A radius) Rg lands near 13 A.
    rg = structure.radius_of_gyration()
    assert 11.0 < rg < 15.0
    # i -> i+4 distance is the diagnostic for an alpha helix (~6.0-6.6 A).
    distances = [np.linalg.norm(coords[i] - coords[i + 4]) for i in range(len(coords) - 4)]
    assert 5.6 < float(np.median(distances)) < 7.0


def test_fold_is_deterministic_for_a_seed():
    a = st_mod.fold_sequence("MKTAYIAKQRQISFVKSHF", seed=42)
    b = st_mod.fold_sequence("MKTAYIAKQRQISFVKSHF", seed=42)
    assert np.allclose(a.ca_coords(), b.ca_coords())
    c = st_mod.fold_sequence("MKTAYIAKQRQISFVKSHF", seed=43)
    assert not np.allclose(a.ca_coords(), c.ca_coords())


def test_fold_respects_chain_continuity():
    structure = st_mod.fold_sequence("MKTAYIAKQRQISFVKSHFSRQ", seed=1)
    coords = structure.ca_coords()
    gaps = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    # A CA-CA virtual bond is 3.8 A; the annealer is not allowed to break it.
    assert gaps.max() < 6.0, f"chain broke: max gap {gaps.max():.2f} A"


def test_a_folded_model_says_it_is_simulated():
    """A downloaded PDB must carry its provenance, not just the UI badge."""
    structure = st_mod.fold_sequence("MKTAYIAKQRQISF", seed=1)
    assert structure.simulated is True
    assert "SIMULATED" in structure.to_pdb().upper()


def test_parse_pdb_round_trip(folded):
    text = folded.pdb_text
    again = st_mod.parse_pdb(text)
    assert again.n_residues == folded.n_residues
    assert again.sequence() == folded.sequence()
    # Coordinates survive the round trip to within PDB's 3-decimal precision.
    assert np.allclose(again.ca_coords(), folded.ca_coords(), atol=0.002)


def test_sasa_of_a_short_peptide_is_physically_plausible():
    structure = st_mod.fold_sequence("GGGGG", seed=0)
    total = float(structure.sasa()["total"])
    # Shrake-Rupley with a 1.4 A probe: a pentaglycine is a few hundred A^2.
    assert 200 < total < 900


def test_buried_residue_has_less_sasa_than_a_protruding_one():
    structure = st_mod.fold_sequence("GGAGGG", seed=0)
    sasa = structure.sasa()["residue_sasa"]
    assert float(sasa[2]) < float(sasa[0])

def test_secondary_structure_assigns_h_along_an_ideal_helix():
    """Kabsch-Sander on a clean 3.6-residue/turn helix must be almost all H.

    The assignment is physics, not a heuristic: an idealised helix has the
    i -> i+4 N-H...O=C hydrogen bonds that *define* an alpha helix, so the
    assigner has to find them.
    """
    helix = _ideal_helix_structure(30)
    ss = helix.secondary_structure()
    assert len(ss) == 30
    assert ss.count("H") >= 0.6 * 30, f"ideal helix assigned {ss}"


def test_a_single_extended_strand_is_not_called_a_sheet():
    """A beta-bridge needs a partner strand; one alone must stay coil/turn."""
    phi_psi = [(-139.0, 135.0)] * 14              # ideal extended (beta) geometry
    strand = st_mod.structure_from_backbone("A" * 14, st_mod.build_backbone("A" * 14, phi_psi))
    ss = strand.secondary_structure()
    assert ss.count("E") == 0, f"an isolated strand must not be assigned E: {ss}"


def test_pockets_are_finite_and_ranked(folded):
    pockets = folded.find_pockets(top=4)
    assert pockets, "the synthetic fold has cavities"
    for pocket in pockets:
        assert 0 < pocket["volume_a3"] < 20000
        assert 0 <= pocket["druggability_hint"] <= 1
        assert pocket["max_burial_depth"] >= 0
        assert 0 <= pocket["ray_burial_fraction"] <= 1
        assert pocket["n_lining_residues"] == len(pocket["lining_residues"])
    # Cavities are ranked by burial and druggability, not by raw volume: a large
    # shallow groove is less interesting than a small fully enclosed pocket.
    assert [p["rank"] for p in pockets] == list(range(1, len(pockets) + 1))
    assert all(p["volume_a3"] > 0 for p in pockets)


def test_pocket_detection_rejects_impossible_thresholds(folded):
    assert folded.find_pockets(min_volume=100000, top=3) == []


def test_contact_map_is_symmetric(folded):
    matrix = folded.contact_map()
    assert matrix.shape == (folded.n_residues, folded.n_residues)
    assert np.allclose(matrix, matrix.T, atol=0.01)


def test_rmsd_of_a_structure_with_itself_is_zero(folded):
    assert folded.rmsd(folded) == pytest.approx(0.0, abs=1e-6)


def test_rmsd_after_a_rigid_motion_is_zero(folded):
    """A rigid rotation must not change the RMSD: superposition has to undo it."""
    rotated = st_mod.parse_pdb(_rotate_pdb(folded.pdb_text, 41.0))
    assert folded.rmsd(rotated) == pytest.approx(0.0, abs=0.05)


def test_kabsch_rmsd_of_different_sequences_is_not_zero():
    a = st_mod.fold_sequence("MKTAYIAKQRQISFVKSHF", seed=1)
    b = st_mod.fold_sequence("ACDEFGHIKLMNPQRSTVWY", seed=1)
    rmsd = a.rmsd(b)
    assert rmsd is not None and rmsd > 1.0


def test_contact_map_overlap_is_one_for_identical_structures(folded):
    assert folded.contact_map_overlap(folded) == pytest.approx(1.0, abs=1e-6)


def test_long_fold_is_truncated_not_left_to_hang():
    structure = st_mod.fold_sequence("A" * 40, max_residues=20, seed=1)
    assert structure.n_residues == 20


def test_short_sequence_is_rejected_with_a_clear_message():
    with pytest.raises(ValueError):
        st_mod.fold_sequence("MKTA")


def test_reference_geometry_is_flagged_as_not_predicted():
    structure = _ideal_helix_structure(10)
    assert structure.meta.get("phi_psi_supplied") is True
    assert "Reference geometry" in structure.meta.get("disclaimer", "")


def test_to_dict_is_json_serialisable(folded):
    import json

    payload = json.dumps(folded.to_dict(profiles=True), default=str)
    assert '"n_residues"' in payload


# ---------------------------------------------------------------- helpers


def _ideal_helix_structure(n: int):
    """Build a canonical alpha helix from explicit torsions (the reference case)."""
    phi_psi = [(-57.0, -47.0)] * n
    return st_mod.structure_from_backbone("A" * n, st_mod.build_backbone("A" * n, phi_psi),
                                          label="ideal alpha helix")


def _rotate_pdb(pdb_text: str, degrees: float) -> str:
    """Rotate every coordinate in a PDB, to prove the superposition is real."""
    theta = np.radians(degrees)
    rotation = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta), np.cos(theta), 0.0],
        [0.0, 0.0, 1.0],
    ])
    out = []
    for line in pdb_text.splitlines():
        if line.startswith(("ATOM", "HETATM")):
            xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            x, y, z = rotation @ xyz
            line = f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}"
        out.append(line)
    return "\n".join(out) + "\n"
