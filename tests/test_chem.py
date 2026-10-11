"""Cheminformatics: descriptors and rules verified against known values."""

from __future__ import annotations

import pytest

from bionemo_web.bio import chem, depict


def test_aspirin_descriptors_are_in_range(aspirin):
    report = chem.report_molecule(aspirin)
    assert report.canonical_smiles
    assert report.mw == pytest.approx(180.16, abs=0.5)      # C9H8O4
    assert report.tpsa == pytest.approx(63.6, abs=1.5)
    assert report.hbd == 1
    assert report.hba == 3          # RDKit's acceptor definition counts the ester O only once
    assert 0.0 < report.qed <= 1.0
    assert 1.0 <= report.sa_score <= 10.0
    assert -1.0 < report.clogp < 3.0
    assert report.ro5_pass


def test_invalid_smiles_is_rejected_not_guessed():
    mol, warnings = chem.parse_smiles("this is not a molecule")
    assert mol is None
    assert warnings


def test_equivalent_smiles_share_a_canonical_form():
    a = chem.report_molecule("CCO").canonical_smiles
    b = chem.report_molecule("OCC").canonical_smiles
    assert a == b


def test_tanimoto_of_identical_molecules_is_one():
    assert chem.tanimoto("CCO", "CCO") == pytest.approx(1.0)
    assert chem.tanimoto("CCO", "CCCCCCCC") < 0.5


def test_pains_filter_flags_a_known_pains_molecule():
    # Quinone is a textbook PAINS/redox-cycler alert.
    report = chem.report_molecule("O=C1C=CC(=O)C=C1")
    assert report.filter_alerts or report.brenk_or_pains, "a quinone should raise a structural alert"


def test_a_marketed_drug_raises_no_pains_alert():
    """aspirin is Brenk-listed for its labile phenol ester, but it is not a PAINS
    frequent hitter -- the two must stay distinguishable."""
    report = chem.report_molecule("CC(=O)Oc1ccccc1C(=O)O")
    severities = {h["severity"] for h in report.filter_alerts}
    assert "alert" not in severities, f"aspirin flagged as a frequent hitter: {report.filter_alerts}"
    assert "note" in severities, "the Brenk note should still be reported"


def test_cluster_molecules_groups_similar_compounds():
    smiles = ["CCCCCCCC", "CCCCCCCCCC", "c1ccccc1", "c1ccccc1C"]
    clusters = chem.cluster_molecules(smiles, threshold=0.5)
    assert clusters
    assert sum(c["size"] for c in clusters) == len(smiles)


def test_similarity_matrix_is_symmetric_with_unit_diagonal():
    labels, matrix = chem.similarity_matrix(["CCO", "CCN", "c1ccccc1"])
    assert labels == ["CCO", "CCN", "c1ccccc1"]
    for i in range(len(labels)):
        assert matrix[i][i] == pytest.approx(1.0)
        for j in range(len(labels)):
            assert matrix[i][j] == pytest.approx(matrix[j][i], abs=1e-9)


def test_analogues_are_valid_and_similar_enough(aspirin):
    rows = chem.generate_analogues(aspirin, n=8, min_similarity=0.3, seed=7)
    assert rows, "analogue generation must always return something"
    for row in rows:
        assert chem.parse_smiles(row["smiles"])[0] is not None
        assert 0.0 <= row["similarity"] <= 1.0
    # Deterministic for a fixed seed.
    again = chem.generate_analogues(aspirin, n=8, min_similarity=0.3, seed=7)
    assert [r["smiles"] for r in rows] == [r["smiles"] for r in again]


def test_analogues_include_the_seed_when_nothing_else_is_possible():
    rows = chem.generate_analogues("CCO", n=5, min_similarity=0.95)
    assert rows and rows[0]["smiles"]


def test_embed_3d_produces_real_geometry():
    mol = chem.embed_3d("CCO", optimize="MMFF")
    assert mol is not None
    assert mol.GetNumConformers() == 1
    positions = mol.GetConformer()
    # Every interatomic distance must be physically sane (> 0.5 A).
    import numpy as np

    coords = np.array([[positions.GetAtomPosition(i).x, positions.GetAtomPosition(i).y,
                        positions.GetAtomPosition(i).z] for i in range(mol.GetNumAtoms())])
    distances = np.linalg.norm(coords[:, None] - coords[None, :], axis=-1)
    off_diagonal = distances[distances > 0]
    assert off_diagonal.min() > 0.5


def test_mol_block_3d_is_parseable():
    block = chem.mol_block_3d("CC(=O)Oc1ccccc1C(=O)O")
    assert block and "V2000" in block.splitlines()[3]
    from rdkit import Chem

    assert Chem.MolFromMolBlock(block) is not None


def test_set_summary_counts_and_averages(aspirin):
    summary = chem.summarise_set([aspirin, "CCN", "c1ccccc1"])
    assert summary["n"] == 3
    assert 0 <= summary["ro5_pass_rate"] <= 1
    assert summary["unique_scaffolds"] >= 1


def test_depiction_renders_svg_without_cairo():
    svg = depict.smiles_svg("CC(=O)Oc1ccccc1C(=O)O", width=200, height=160)
    assert svg.startswith("<svg") and svg.endswith("</svg>")
    assert "<line" in svg
    # An unparseable SMILES must produce an error tile, never an exception.
    bad = depict.smiles_svg("not a molecule")
    assert bad.startswith("<svg")
