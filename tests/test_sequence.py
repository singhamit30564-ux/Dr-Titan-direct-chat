"""Sequence analysis: the numbers have to be right, not plausible."""

from __future__ import annotations

import pytest

from bionemo_web.bio import constants as C
from bionemo_web.bio import sequence as seq_mod


def test_parse_fasta_multi_record():
    text = ">a first\nMKTAYIAKQR\n>b second\nGGGGGG\n"
    records = seq_mod.parse_fasta(text)
    assert [r["id"] for r in records] == ["a", "b"]
    assert records[0]["sequence"] == "MKTAYIAKQR"


def test_clean_sequence_strips_junk():
    cleaned, warnings = seq_mod.clean_sequence("mkt ayia\n1234kqr*")
    assert cleaned == "MKTAYIAKQR"
    assert warnings  # the junk is reported, not silently dropped


def test_detect_sequence_type():
    assert seq_mod.detect_sequence_type("MKTAYIAKQRQISFVKSHFSRQ") == "protein"
    assert seq_mod.detect_sequence_type("ATGGCATTTAAACCCGGGTTT") == "dna"
    assert seq_mod.detect_sequence_type("AUGCAAUUUCGG") in ("rna", "dna")


def test_molecular_weight_matches_hand_calculation():
    """Average residue masses plus one water; monoisotopic must come out lower."""
    report = seq_mod.analyse_protein("GGGG")
    # Constants are free-amino-acid average masses, so a chain of n residues is
    # n residue masses minus (n-1) waters -- i.e. one water is retained.
    expected = 4 * C.RESIDUE_MASS_AVG["G"] - 3 * C.WATER_MASS
    assert report.molecular_weight == pytest.approx(expected, abs=0.5)
    assert report.monoisotopic_mass < report.molecular_weight  # no 13C in the monoisotopic mass


def test_isoelectric_point_brackets_true_value():
    # A chain with only acidic residues must be very acidic.
    acidic = seq_mod.analyse_protein("DDDDDDE")
    basic = seq_mod.analyse_protein("KKKKKKR")
    assert acidic.isoelectric_point < 4.5
    assert basic.isoelectric_point > 9.5
    assert acidic.charge_pH7 < 0 < basic.charge_pH7


def test_gravy_extremes():
    hydrophobe = seq_mod.analyse_protein("IIIIIVVVVVLLLLL").gravy
    polar = seq_mod.analyse_protein("DDDDDEEEKKKRR").gravy
    assert hydrophobe > 2.0
    assert polar < -2.0


def test_translation_uses_the_standard_code():
    # ATG AAA TGT TAA -> M K C * (the stop codon is shown, not dropped)
    assert seq_mod.translate("ATGAAATGTTAA") == "MKC*"
    assert seq_mod.translate("ATG") == "M"


def test_start_codons_are_the_real_set():
    # The standard genetic code has exactly three start codons.
    assert set(C.START_CODONS) == {"ATG", "TTG", "CTG"}


def test_dna_analysis_reports_orfs_and_gc():
    # Keep the ORF comfortably above the 30-aa default reporting threshold.
    dna = "ATG" + "AAA" * 36 + "TAA" + "GCG" * 5
    report = seq_mod.analyse_dna(dna)
    assert report["gc_content"] == pytest.approx(
        sum(1 for c in dna if c in "GC") / len(dna), abs=1e-3
    )
    assert report["orf_count"] >= 1
    assert report["longest_orf_aa"] >= 30
    assert report["length"] == len(dna)
    assert report["effective_codons"] > 0


def test_melting_temperature_increases_with_gc():
    at_rich = seq_mod.melting_temperature_nn("ATATATATATATATAT")
    gc_rich = seq_mod.melting_temperature_nn("GCGCGCGCGCGCGCGC")
    assert gc_rich > at_rich


def test_reverse_complement():
    assert seq_mod.reverse_complement("ATGC") == "GCAT"


def test_motif_library_finds_known_motifs():
    """RGD attachment motif and a PKA-style site must both be found."""
    report = seq_mod.analyse_protein("GRGDSMKTAYIAKRRGSFVK")
    names = [str(m.get("name") if isinstance(m, dict) else m).lower() for m in report.motifs]
    assert any("rgd" in n for n in names), f"RGD not found in {names}"
    assert any("kinase" in n or "pka" in n for n in names), f"kinase site not found in {names}"


def test_prosite_patterns_convert_to_regex():
    """The motif library uses PROSITE syntax; the converter has to handle it."""
    regex = seq_mod.prosite_to_regex("N-{P}-[ST]-{P}")
    assert regex is not None


def test_analysis_is_deterministic():
    a = seq_mod.analyse_protein("MKTAYIAKQRQISFVKSHFSRQ").to_dict()
    b = seq_mod.analyse_protein("MKTAYIAKQRQISFVKSHFSRQ").to_dict()
    assert a == b
