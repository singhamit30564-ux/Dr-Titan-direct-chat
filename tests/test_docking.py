"""The docking engine: geometry, score behaviour and the NIM wire format."""

from __future__ import annotations

import numpy as np

from bionemo_web.nims import docking


def test_docking_returns_ranked_poses(folded, aspirin):
    result = docking.dock(folded, aspirin, num_poses=4, seed=11, refine_iterations=40)
    poses = result["poses"]
    assert len(poses) == 4
    assert [p.rank for p in poses] == [1, 2, 3, 4]
    scores = [p.score for p in poses]
    assert scores == sorted(scores, reverse=True), "poses must be ranked by score"
    for pose in poses:
        assert 0.0 <= pose.confidence <= 1.0
        assert pose.coords.shape[1] == 3
        assert pose.contacts >= 0 and pose.clashes >= 0 and pose.hbonds >= 0


def test_docking_is_deterministic_for_a_seed(folded, aspirin):
    a = docking.dock(folded, aspirin, num_poses=3, seed=99, refine_iterations=30)
    b = docking.dock(folded, aspirin, num_poses=3, seed=99, refine_iterations=30)
    assert np.allclose(a["poses"][0].coords, b["poses"][0].coords)


def test_poses_land_inside_the_protein_not_far_away(folded, aspirin):
    result = docking.dock(folded, aspirin, num_poses=3, seed=5, refine_iterations=40)
    centre = folded.ca_coords().mean(0)
    radius = np.linalg.norm(folded.ca_coords() - centre, axis=1).max()
    for pose in result["poses"]:
        distance = np.linalg.norm(pose.coords.mean(0) - centre)
        assert distance < radius * 1.5, "a pose escaped the protein entirely"


def test_confidence_penalises_clashes():
    clean = docking._confidence_from_score(30.0, {"clashes": 0})
    clashing = docking._confidence_from_score(30.0, {"clashes": 6})
    assert clashing < clean
    assert 0.0 <= clashing <= 1.0


def test_score_rewards_contacts_and_punishes_overlap(folded):
    from bionemo_web.bio import chem as chem_mod

    mol = chem_mod.embed_3d("CCO", optimize="MMFF")
    coords, elements, donors, acceptors = docking._ligand_atom_arrays(mol)
    protein_coords, protein_elements, protein_residues, _ = docking._protein_arrays(folded)

    # A pose in the middle of an atom must score worse than one in a cavity.
    overlapping = np.tile(protein_coords[0], (len(coords), 1))
    stats_overlap = docking._score_pose(overlapping, elements, donors, acceptors,
                                        protein_coords, protein_elements, protein_residues)
    assert stats_overlap["clashes"] > 0

    cavity = np.array(folded.find_pockets(top=1)[0]["centre"])
    placed = coords + cavity
    stats_placed = docking._score_pose(placed, elements, donors, acceptors,
                                       protein_coords, protein_elements, protein_residues)
    assert stats_placed["score"] > stats_overlap["score"]


def test_nim_wire_format_round_trips(folded, aspirin):
    result = docking.dock(folded, aspirin, num_poses=2, seed=3, refine_iterations=20)
    payload = docking.poses_to_nim_format(result["poses"], result["ligand_atom_count"])
    assert set(payload) == {"ligand_positions", "position_confidence"}
    assert len(payload["ligand_positions"]) == 2
    # DiffDock returns a flat xyz stream; it has to reproduce the coordinates.
    flat = payload["ligand_positions"][0]
    assert len(flat) == 3 * result["ligand_atom_count"]
    restored = docking.unflatten_positions(flat, result["ligand_atom_count"])
    assert np.allclose(restored, result["poses"][0].coords, atol=1e-3)


def test_docking_handles_a_bad_ligand_gracefully(folded):
    result = docking.dock(folded, "not a smiles", num_poses=2, seed=1)
    assert result["poses"] == []
    assert "error" in result


def test_surface_fallback_when_no_pocket_is_given(folded, aspirin):
    result = docking.dock(folded, aspirin, num_poses=2, seed=1, pockets=[], refine_iterations=20)
    assert result["poses"], "an empty pocket list must still produce poses"
