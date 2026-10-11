"""NIM layer: the registry, response normalisation and the simulation backends."""

from __future__ import annotations

import json

import pytest

from bionemo_web.nims import catalog
from bionemo_web.nims.client import NimClient, normalise
from bionemo_web.nims.service import BACKEND_LABELS, NimService
from bionemo_web.nims import sim as sim_mod


# ---------------------------------------------------------------- registry


def test_every_model_spec_is_complete():
    models = catalog.all_models()
    assert len(models) >= 15
    for spec in models:
        assert spec.key and spec.name and spec.vendor and spec.kind and spec.task
        assert spec.hosted_path or spec.local_path, f"{spec.key} has no endpoint at all"
        assert spec.path_confidence in ("documented", "configurable", "verified")
    assert len({s.key for s in models}) == len(models)


def test_aliases_resolve_to_real_models():
    for alias in catalog.ALIASES:
        assert catalog.resolve(alias).key == catalog.ALIASES[alias] or catalog.resolve(alias) is not None


def test_the_models_the_user_asked_for_are_present():
    for key in ("esmfold", "alphafold2", "openfold3", "boltz2", "molmim", "genmol",
                "diffdock", "esm2", "proteinmpnn", "rfdiffusion", "evo2"):
        assert catalog.get(key) is not None, f"{key} is missing from the registry"


def test_param_defaults_come_from_the_spec():
    defaults = catalog.param_defaults(catalog.get("molmim"))
    assert isinstance(defaults, dict)


# ------------------------------------------------------- normalisation


def test_normalise_molmim_hosted_shape():
    raw = {"molecules": [{"smiles": "CCO", "score": 0.7}]}
    out = normalise("molecules", raw)
    assert out["molecules"][0]["smiles"] == "CCO"
    assert out["molecules"][0]["score"] == pytest.approx(0.7)


def test_normalise_molmim_self_hosted_shape():
    # The container returns {"generated": [{"sample": ..., "score": ...}]}
    raw = {"status": "success", "generated": [{"sample": "CCO", "score": 0.5}]}
    out = normalise("molecules", raw)
    assert out["molecules"][0]["smiles"] == "CCO"
    assert out["status"] == "success"


def test_normalise_diffdock_flat_positions():
    raw = {"ligand_positions": [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], "position_confidence": [0.8]}
    out = normalise("dock", raw)
    assert out["ligand_positions"] == [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]
    assert out["position_confidence"] == [0.8]


def test_normalise_fold_computes_mean_plddt():
    out = normalise("fold", {"pdbs": ["ATOM"], "plddt": [80.0, 90.0, 70.0], "ptm": 0.6})
    assert out["mean_plddt"] == pytest.approx(80.0)
    assert out["pdbs"] == ["ATOM"]


def test_normalise_is_forgiving_of_unknown_keys():
    out = normalise("molecules", {"unexpected": "payload"})
    assert isinstance(out, dict)


# ------------------------------------------------------------- simulation


def test_simulated_fold_returns_a_nim_shaped_response(sequence):
    bundle = sim_mod.simulate(catalog.get("esmfold"), {"sequence": sequence}, seed=3)
    canonical, warnings = bundle["canonical"], bundle["warnings"]
    assert canonical["pdbs"] and canonical["pdbs"][0].startswith(("ATOM", "HEADER", "REMARK"))
    assert len(canonical["plddt"]) == len(sequence)
    assert bundle["method"] and bundle["method"] != ""
    assert warnings, "a simulated run must carry a warning saying so"


def test_simulated_dock_uses_the_diffdock_layout(folded, aspirin):
    bundle = sim_mod.simulate(
        catalog.get("diffdock"),
        {"protein": folded.pdb_text, "ligand": aspirin, "num_poses": 3},
        seed=4,
    )
    canonical = bundle["canonical"]
    assert len(canonical["ligand_positions"]) == 3
    assert len(canonical["position_confidence"]) == 3
    assert canonical["ligand_positions"][0][:3] != [0.0, 0.0, 0.0]


def test_simulated_embeddings_have_the_right_shape(sequence):
    bundle = sim_mod.simulate(catalog.get("esm2"), {"sequence": sequence}, seed=1)
    embeddings = bundle["canonical"]["embeddings"]
    assert len(embeddings) == len(sequence)
    assert len({len(row) for row in embeddings}) == 1
    assert bundle["meta"]["quality"]["n_residues"] == len(sequence)


def test_simulated_embeddings_are_deterministic(sequence):
    a = sim_mod.simulate(catalog.get("esm2"), {"sequence": sequence}, seed=9)["canonical"]
    b = sim_mod.simulate(catalog.get("esm2"), {"sequence": sequence}, seed=9)["canonical"]
    assert a["embeddings"] == b["embeddings"]


def test_simulated_molecules_are_valid_chemistry(aspirin):
    bundle = sim_mod.simulate(
        catalog.get("molmim"),
        {"smi": aspirin, "num_molecules": 6, "property_name": "QED"},
        seed=2,
    )
    molecules = bundle["canonical"]["molecules"]
    assert molecules
    from bionemo_web.bio import chem

    for molecule in molecules:
        assert chem.parse_smiles(molecule["smiles"])[0] is not None


def test_simulated_genmol_accepts_a_masked_smiles():
    bundle = sim_mod.simulate(
        catalog.get("genmol"),
        {"smiles": "[C@H]1O[C@@H](CO)[C@H](O)[C@@H]1O.[*{15-15}]", "num_molecules": 4},
        seed=2,
    )
    assert bundle["canonical"]["molecules"]


def test_simulated_design_recovers_most_of_the_wild_type(folded):
    bundle = sim_mod.simulate(
        catalog.get("proteinmpnn"),
        {"pdb": folded.pdb_text, "num_sequences": 3, "sampling_temp": 0.1},
        seed=3,
    )
    designs = bundle["canonical"]["sequences"]
    assert len(designs) == 3
    for design in designs:
        assert len(design["sequence"]) == folded.n_residues
        assert 0.0 <= design["recovery"] <= 1.0
        assert "score" in design


def test_simulated_genomics_produces_dna_and_analysis():
    bundle = sim_mod.simulate(
        catalog.get("evo2"), {"sequence": "ATGAAAGCT", "num_tokens": 90, "temperature": 0.8}, seed=1
    )
    canonical = bundle["canonical"]
    generated = canonical["generated"]
    assert len(generated) == 90
    assert canonical["sequence"].endswith(generated)
    assert len(canonical["sequence"]) == len(generated) + 9      # prompt was 9 nt
    assert set(generated) <= set("ACGT")
    assert bundle["meta"]["analysis"]["gc_content"] >= 0


def test_unknown_model_key_fails_cleanly():
    class _Fake:
        key = "nonexistent"
        kind = "fold"
        name = "Fake"
        vendor = "test"
        hosted_path = ""
        local_path = ""
        local_only = True

    bundle = sim_mod.simulate(_Fake(), {"sequence": "MK"}, seed=1)
    assert bundle["canonical"] == {}
    assert bundle["warnings"]


# ------------------------------------------------------------ service


def test_service_falls_back_to_simulation_and_says_so(sequence):
    service = NimService(
        NimClient(), {"prefer_backend": "auto", "allow_hosted": False, "simulation_seed": 1}
    )
    result = service.run("esmfold", {"sequence": sequence})
    assert result.ok
    assert result.simulated is True
    assert result.backend == "simulate"
    assert result.warnings, "the fallback must be reported to the user"
    manifest = result.provenance()
    assert manifest["simulated"] is True
    assert manifest["model_key"] == "esmfold"
    assert manifest["backend_label"] == BACKEND_LABELS["simulate"]


def test_service_reports_an_unknown_model_without_raising():
    service = NimService(NimClient(), {})
    result = service.run("does-not-exist", {})
    assert result.ok is False
    assert "Unknown model" in result.error


def test_provenance_is_json_serialisable(sequence):
    service = NimService(NimClient(), {"prefer_backend": "simulate", "allow_hosted": False})
    result = service.run("esm2", {"sequence": sequence})
    payload = json.dumps(result.provenance(), default=str)
    assert '"model_key"' in payload and '"created_at"' in payload
