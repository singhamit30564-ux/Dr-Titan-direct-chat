"""Scientific reference tables used by the BioNeMo Web Console.

Every table in this file is transcribed from a published source; the origin is
named next to each block.  Nothing here is invented, and no value was smoothed or
hand-tuned: the console reports the same numbers a reference implementation does.

Provenance
----------
* Hydrophobicity / flexibility / instability scales and the dipeptide instability
  weight value (DIWV) table: Biopython's ``Bio.SeqUtils.ProtParamData``
  (Biopython License / BSD-3-Clause), which transcribes
    - Kyte & Doolittle, J. Mol. Biol. 157:105-132 (1982)
    - Eisenberg et al., J. Mol. Biol. 179:125-142 (1984)
    - Guruprasad, Reddy & Pandit, Protein Engineering 4:155-161 (1990)
    - Bhaskaran & Ponnuswamy, Int. J. Peptide Protein Res. 32:242-255 (1988)
* pK values for the Bjellqvist isoelectric-point method: Biopython
  ``Bio.SeqUtils.IsoelectricPoint``
    - Bjellqvist et al., Electrophoresis 14:1023-1031 (1993)
    - Bjellqvist et al., Electrophoresis 15:529-539 (1994)
* Amino-acid residue masses: Biopython ``Bio.Data.IUPACData`` (IUPAC average and
  monoisotopic residue masses).
* DNA nearest-neighbour thermodynamic parameters: Biopython
  ``Bio.SeqUtils.MeltingTemp.DNA_NN3``
    - Allawi & SantaLucia, Biochemistry 36:10581-10594 (1997)
* Standard genetic code (NCBI translation table 1): Biopython
  ``Bio.Data.CodonTable``.
"""

# --- Kyte & Doolittle hydrophobicity (1982) -------------------------------
KYTE_DOOLITTLE = {
    "A": 1.8,
    "R": -4.5,
    "N": -3.5,
    "D": -3.5,
    "C": 2.5,
    "Q": -3.5,
    "E": -3.5,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "L": 3.8,
    "K": -3.9,
    "M": 1.9,
    "F": 2.8,
    "P": -1.6,
    "S": -0.8,
    "T": -0.7,
    "W": -0.9,
    "Y": -1.3,
    "V": 4.2
}

# --- Eisenberg normalised consensus hydrophobicity (1984) ------------------
EISENBERG = {
    "A": 0.62,
    "R": -2.53,
    "N": -0.78,
    "D": -0.9,
    "C": 0.29,
    "Q": -0.85,
    "E": -0.74,
    "G": 0.48,
    "H": -0.4,
    "I": 1.38,
    "L": 1.06,
    "K": -1.5,
    "M": 0.64,
    "F": 1.19,
    "P": 0.12,
    "S": -0.18,
    "T": -0.05,
    "W": 0.81,
    "Y": 0.26,
    "V": 1.08
}

# --- Bhaskaran & Ponnuswamy flexibility scale ------------------------------
FLEXIBILITY = {
    "A": 0.984,
    "C": 0.906,
    "E": 1.094,
    "D": 1.068,
    "G": 1.031,
    "F": 0.915,
    "I": 0.927,
    "H": 0.95,
    "K": 1.102,
    "M": 0.952,
    "L": 0.935,
    "N": 1.048,
    "Q": 1.037,
    "P": 1.049,
    "S": 1.046,
    "R": 1.008,
    "T": 0.997,
    "W": 0.904,
    "V": 0.931,
    "Y": 0.929
}

# --- Volume (A^3) of amino-acid side chains (Biopython 'mi' scale) ---------
SIDE_CHAIN_VOLUME = {
    "A": 5.33,
    "R": 4.18,
    "N": 3.71,
    "D": 3.59,
    "C": 7.93,
    "Q": 3.87,
    "E": 3.65,
    "G": 4.48,
    "H": 5.1,
    "I": 8.83,
    "L": 8.47,
    "K": 2.95,
    "M": 8.95,
    "F": 9.03,
    "P": 3.87,
    "S": 4.09,
    "T": 4.49,
    "W": 7.66,
    "Y": 5.89,
    "V": 7.63
}

# --- Guruprasad dipeptide instability weight values (1990) --------------
# instability_index = (10 / N) * sum(DIWV[a][b] for adjacent pairs)
DIWV = {
    "A": {
        "A": 1.0,
        "C": 44.94,
        "E": 1.0,
        "D": -7.49,
        "G": 1.0,
        "F": 1.0,
        "I": 1.0,
        "H": -7.49,
        "K": 1.0,
        "M": 1.0,
        "L": 1.0,
        "N": 1.0,
        "Q": 1.0,
        "P": 20.26,
        "S": 1.0,
        "R": 1.0,
        "T": 1.0,
        "W": 1.0,
        "V": 1.0,
        "Y": 1.0
    },
    "C": {
        "A": 1.0,
        "C": 1.0,
        "E": 1.0,
        "D": 20.26,
        "G": 1.0,
        "F": 1.0,
        "I": 1.0,
        "H": 33.6,
        "K": 1.0,
        "M": 33.6,
        "L": 20.26,
        "N": 1.0,
        "Q": -6.54,
        "P": 20.26,
        "S": 1.0,
        "R": 1.0,
        "T": 33.6,
        "W": 24.68,
        "V": -6.54,
        "Y": 1.0
    },
    "E": {
        "A": 1.0,
        "C": 44.94,
        "E": 33.6,
        "D": 20.26,
        "G": 1.0,
        "F": 1.0,
        "I": 20.26,
        "H": -6.54,
        "K": 1.0,
        "M": 1.0,
        "L": 1.0,
        "N": 1.0,
        "Q": 20.26,
        "P": 20.26,
        "S": 20.26,
        "R": 1.0,
        "T": 1.0,
        "W": -14.03,
        "V": 1.0,
        "Y": 1.0
    },
    "D": {
        "A": 1.0,
        "C": 1.0,
        "E": 1.0,
        "D": 1.0,
        "G": 1.0,
        "F": -6.54,
        "I": 1.0,
        "H": 1.0,
        "K": -7.49,
        "M": 1.0,
        "L": 1.0,
        "N": 1.0,
        "Q": 1.0,
        "P": 1.0,
        "S": 20.26,
        "R": -6.54,
        "T": -14.03,
        "W": 1.0,
        "V": 1.0,
        "Y": 1.0
    },
    "G": {
        "A": -7.49,
        "C": 1.0,
        "E": -6.54,
        "D": 1.0,
        "G": 13.34,
        "F": 1.0,
        "I": -7.49,
        "H": 1.0,
        "K": -7.49,
        "M": 1.0,
        "L": 1.0,
        "N": -7.49,
        "Q": 1.0,
        "P": 1.0,
        "S": 1.0,
        "R": 1.0,
        "T": -7.49,
        "W": 13.34,
        "V": 1.0,
        "Y": -7.49
    },
    "F": {
        "A": 1.0,
        "C": 1.0,
        "E": 1.0,
        "D": 13.34,
        "G": 1.0,
        "F": 1.0,
        "I": 1.0,
        "H": 1.0,
        "K": -14.03,
        "M": 1.0,
        "L": 1.0,
        "N": 1.0,
        "Q": 1.0,
        "P": 20.26,
        "S": 1.0,
        "R": 1.0,
        "T": 1.0,
        "W": 1.0,
        "V": 1.0,
        "Y": 33.601
    },
    "I": {
        "A": 1.0,
        "C": 1.0,
        "E": 44.94,
        "D": 1.0,
        "G": 1.0,
        "F": 1.0,
        "I": 1.0,
        "H": 13.34,
        "K": -7.49,
        "M": 1.0,
        "L": 20.26,
        "N": 1.0,
        "Q": 1.0,
        "P": -1.88,
        "S": 1.0,
        "R": 1.0,
        "T": 1.0,
        "W": 1.0,
        "V": -7.49,
        "Y": 1.0
    },
    "H": {
        "A": 1.0,
        "C": 1.0,
        "E": 1.0,
        "D": 1.0,
        "G": -9.37,
        "F": -9.37,
        "I": 44.94,
        "H": 1.0,
        "K": 24.68,
        "M": 1.0,
        "L": 1.0,
        "N": 24.68,
        "Q": 1.0,
        "P": -1.88,
        "S": 1.0,
        "R": 1.0,
        "T": -6.54,
        "W": -1.88,
        "V": 1.0,
        "Y": 44.94
    },
    "K": {
        "A": 1.0,
        "C": 1.0,
        "E": 1.0,
        "D": 1.0,
        "G": -7.49,
        "F": 1.0,
        "I": -7.49,
        "H": 1.0,
        "K": 1.0,
        "M": 33.6,
        "L": -7.49,
        "N": 1.0,
        "Q": 24.64,
        "P": -6.54,
        "S": 1.0,
        "R": 33.6,
        "T": 1.0,
        "W": 1.0,
        "V": -7.49,
        "Y": 1.0
    },
    "M": {
        "A": 13.34,
        "C": 1.0,
        "E": 1.0,
        "D": 1.0,
        "G": 1.0,
        "F": 1.0,
        "I": 1.0,
        "H": 58.28,
        "K": 1.0,
        "M": -1.88,
        "L": 1.0,
        "N": 1.0,
        "Q": -6.54,
        "P": 44.94,
        "S": 44.94,
        "R": -6.54,
        "T": -1.88,
        "W": 1.0,
        "V": 1.0,
        "Y": 24.68
    },
    "L": {
        "A": 1.0,
        "C": 1.0,
        "E": 1.0,
        "D": 1.0,
        "G": 1.0,
        "F": 1.0,
        "I": 1.0,
        "H": 1.0,
        "K": -7.49,
        "M": 1.0,
        "L": 1.0,
        "N": 1.0,
        "Q": 33.6,
        "P": 20.26,
        "S": 1.0,
        "R": 20.26,
        "T": 1.0,
        "W": 24.68,
        "V": 1.0,
        "Y": 1.0
    },
    "N": {
        "A": 1.0,
        "C": -1.88,
        "E": 1.0,
        "D": 1.0,
        "G": -14.03,
        "F": -14.03,
        "I": 44.94,
        "H": 1.0,
        "K": 24.68,
        "M": 1.0,
        "L": 1.0,
        "N": 1.0,
        "Q": -6.54,
        "P": -1.88,
        "S": 1.0,
        "R": 1.0,
        "T": -7.49,
        "W": -9.37,
        "V": 1.0,
        "Y": 1.0
    },
    "Q": {
        "A": 1.0,
        "C": -6.54,
        "E": 20.26,
        "D": 20.26,
        "G": 1.0,
        "F": -6.54,
        "I": 1.0,
        "H": 1.0,
        "K": 1.0,
        "M": 1.0,
        "L": 1.0,
        "N": 1.0,
        "Q": 20.26,
        "P": 20.26,
        "S": 44.94,
        "R": 1.0,
        "T": 1.0,
        "W": 1.0,
        "V": -6.54,
        "Y": -6.54
    },
    "P": {
        "A": 20.26,
        "C": -6.54,
        "E": 18.38,
        "D": -6.54,
        "G": 1.0,
        "F": 20.26,
        "I": 1.0,
        "H": 1.0,
        "K": 1.0,
        "M": -6.54,
        "L": 1.0,
        "N": 1.0,
        "Q": 20.26,
        "P": 20.26,
        "S": 20.26,
        "R": -6.54,
        "T": 1.0,
        "W": -1.88,
        "V": 20.26,
        "Y": 1.0
    },
    "S": {
        "A": 1.0,
        "C": 33.6,
        "E": 20.26,
        "D": 1.0,
        "G": 1.0,
        "F": 1.0,
        "I": 1.0,
        "H": 1.0,
        "K": 1.0,
        "M": 1.0,
        "L": 1.0,
        "N": 1.0,
        "Q": 20.26,
        "P": 44.94,
        "S": 20.26,
        "R": 20.26,
        "T": 1.0,
        "W": 1.0,
        "V": 1.0,
        "Y": 1.0
    },
    "R": {
        "A": 1.0,
        "C": 1.0,
        "E": 1.0,
        "D": 1.0,
        "G": -7.49,
        "F": 1.0,
        "I": 1.0,
        "H": 20.26,
        "K": 1.0,
        "M": 1.0,
        "L": 1.0,
        "N": 13.34,
        "Q": 20.26,
        "P": 20.26,
        "S": 44.94,
        "R": 58.28,
        "T": 1.0,
        "W": 58.28,
        "V": 1.0,
        "Y": -6.54
    },
    "T": {
        "A": 1.0,
        "C": 1.0,
        "E": 20.26,
        "D": 1.0,
        "G": -7.49,
        "F": 13.34,
        "I": 1.0,
        "H": 1.0,
        "K": 1.0,
        "M": 1.0,
        "L": 1.0,
        "N": -14.03,
        "Q": -6.54,
        "P": 1.0,
        "S": 1.0,
        "R": 1.0,
        "T": 1.0,
        "W": -14.03,
        "V": 1.0,
        "Y": 1.0
    },
    "W": {
        "A": -14.03,
        "C": 1.0,
        "E": 1.0,
        "D": 1.0,
        "G": -9.37,
        "F": 1.0,
        "I": 1.0,
        "H": 24.68,
        "K": 1.0,
        "M": 24.68,
        "L": 13.34,
        "N": 13.34,
        "Q": 1.0,
        "P": 1.0,
        "S": 1.0,
        "R": 1.0,
        "T": -14.03,
        "W": 1.0,
        "V": -7.49,
        "Y": 1.0
    },
    "V": {
        "A": 1.0,
        "C": 1.0,
        "E": 1.0,
        "D": -14.03,
        "G": -7.49,
        "F": 1.0,
        "I": 1.0,
        "H": 1.0,
        "K": -1.88,
        "M": 1.0,
        "L": 1.0,
        "N": 1.0,
        "Q": 1.0,
        "P": 20.26,
        "S": 1.0,
        "R": 1.0,
        "T": -7.49,
        "W": 1.0,
        "V": 1.0,
        "Y": -6.54
    },
    "Y": {
        "A": 24.68,
        "C": 1.0,
        "E": -6.54,
        "D": 24.68,
        "G": -7.49,
        "F": 1.0,
        "I": 1.0,
        "H": 13.34,
        "K": 1.0,
        "M": 44.94,
        "L": 1.0,
        "N": 1.0,
        "Q": 1.0,
        "P": 13.34,
        "S": 1.0,
        "R": -15.91,
        "T": -7.49,
        "W": -9.37,
        "V": 1.0,
        "Y": 13.34
    }
}

# --- Bjellqvist pK values (1993/1994) -----------------------------------
PKA_POSITIVE = {
    "Nterm": 7.5,
    "K": 10.0,
    "R": 12.0,
    "H": 5.98
}
PKA_NEGATIVE = {
    "Cterm": 3.55,
    "D": 4.05,
    "E": 4.45,
    "C": 9.0,
    "Y": 10.0
}
PK_CTERMINAL = {
    "D": 4.55,
    "E": 4.75
}
PK_NTERMINAL = {
    "A": 7.59,
    "M": 7.0,
    "S": 6.93,
    "P": 8.36,
    "T": 6.82,
    "V": 7.44,
    "E": 7.7
}

# --- IUPAC residue masses (Da) -----------------------------------------
RESIDUE_MASS_AVG = {
    "A": 89.0932,
    "C": 121.1582,
    "D": 133.1027,
    "E": 147.1293,
    "F": 165.1891,
    "G": 75.0666,
    "H": 155.1546,
    "I": 131.1729,
    "K": 146.1876,
    "L": 131.1729,
    "M": 149.2113,
    "N": 132.1179,
    "O": 255.3134,
    "P": 115.1305,
    "Q": 146.1445,
    "R": 174.201,
    "S": 105.0926,
    "T": 119.1192,
    "U": 168.0532,
    "V": 117.1463,
    "W": 204.2252,
    "Y": 181.1885
}

RESIDUE_MASS_MONO = {
    "A": 89.047678,
    "C": 121.019749,
    "D": 133.037508,
    "E": 147.053158,
    "F": 165.078979,
    "G": 75.032028,
    "H": 155.069477,
    "I": 131.094629,
    "K": 146.105528,
    "L": 131.094629,
    "M": 149.051049,
    "N": 132.053492,
    "O": 255.158292,
    "P": 115.063329,
    "Q": 146.069142,
    "R": 174.111676,
    "S": 105.042593,
    "T": 119.058243,
    "U": 168.964203,
    "V": 117.078979,
    "W": 204.089878,
    "Y": 181.073893
}

# --- Allawi & SantaLucia DNA nearest-neighbour parameters (1997) ----------
DNA_NN_SANTALUCIA = {
    "init": [
        0,
        0
    ],
    "init_A/T": [
        2.3,
        4.1
    ],
    "init_G/C": [
        0.1,
        -2.8
    ],
    "init_oneG/C": [
        0,
        0
    ],
    "init_allA/T": [
        0,
        0
    ],
    "init_5T/A": [
        0,
        0
    ],
    "sym": [
        0,
        -1.4
    ],
    "AA/TT": [
        -7.9,
        -22.2
    ],
    "AT/TA": [
        -7.2,
        -20.4
    ],
    "TA/AT": [
        -7.2,
        -21.3
    ],
    "CA/GT": [
        -8.5,
        -22.7
    ],
    "GT/CA": [
        -8.4,
        -22.4
    ],
    "CT/GA": [
        -7.8,
        -21.0
    ],
    "GA/CT": [
        -8.2,
        -22.2
    ],
    "CG/GC": [
        -10.6,
        -27.2
    ],
    "GC/CG": [
        -9.8,
        -24.4
    ],
    "GG/CC": [
        -8.0,
        -19.9
    ]
}

# --- Standard genetic code, NCBI translation table 1 --------------------
STANDARD_CODE = {
    "TTT": "F",
    "TTC": "F",
    "TTA": "L",
    "TTG": "L",
    "TCT": "S",
    "TCC": "S",
    "TCA": "S",
    "TCG": "S",
    "TAT": "Y",
    "TAC": "Y",
    "TGT": "C",
    "TGC": "C",
    "TGG": "W",
    "CTT": "L",
    "CTC": "L",
    "CTA": "L",
    "CTG": "L",
    "CCT": "P",
    "CCC": "P",
    "CCA": "P",
    "CCG": "P",
    "CAT": "H",
    "CAC": "H",
    "CAA": "Q",
    "CAG": "Q",
    "CGT": "R",
    "CGC": "R",
    "CGA": "R",
    "CGG": "R",
    "ATT": "I",
    "ATC": "I",
    "ATA": "I",
    "ATG": "M",
    "ACT": "T",
    "ACC": "T",
    "ACA": "T",
    "ACG": "T",
    "AAT": "N",
    "AAC": "N",
    "AAA": "K",
    "AAG": "K",
    "AGT": "S",
    "AGC": "S",
    "AGA": "R",
    "AGG": "R",
    "GTT": "V",
    "GTC": "V",
    "GTA": "V",
    "GTG": "V",
    "GCT": "A",
    "GCC": "A",
    "GCA": "A",
    "GCG": "A",
    "GAT": "D",
    "GAC": "D",
    "GAA": "E",
    "GAG": "E",
    "GGT": "G",
    "GGC": "G",
    "GGA": "G",
    "GGG": "G"
}
START_CODONS = ["TTG", "CTG", "ATG"]
STOP_CODONS = ["TAA", "TAG", "TGA"]


# ---------------------------------------------------------------------------
# Derived convenience tables (computed from the published data above).
# ---------------------------------------------------------------------------

AA_LETTERS = "ACDEFGHIKLMNPQRSTVWY"

AA_NAMES = {
    "A": "Alanine", "C": "Cysteine", "D": "Aspartate", "E": "Glutamate",
    "F": "Phenylalanine", "G": "Glycine", "H": "Histidine", "I": "Isoleucine",
    "K": "Lysine", "L": "Leucine", "M": "Methionine", "N": "Asparagine",
    "P": "Proline", "Q": "Glutamine", "R": "Arginine", "S": "Serine",
    "T": "Threonine", "V": "Valine", "W": "Tryptophan", "Y": "Tyrosine",
}

AA_3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "MSE": "M", "SEC": "U", "PYL": "O",
    "HSD": "H", "HSE": "H", "HSP": "H", "CYX": "C", "ASH": "D", "GLH": "E",
}

AA_CLASS = {
    "A": "hydrophobic", "V": "hydrophobic", "I": "hydrophobic", "L": "hydrophobic",
    "M": "hydrophobic", "F": "aromatic", "W": "aromatic", "Y": "aromatic",
    "G": "special", "P": "special", "C": "special",
    "S": "polar", "T": "polar", "N": "polar", "Q": "polar",
    "K": "positive", "R": "positive", "H": "positive",
    "D": "negative", "E": "negative",
}

# Residues reported as disordered / flexible by the Haimov & Srebnik (2016),
# Hutchinson & Thornton (1994) and Kim & Berg (1993) analyses summarised in
# Biopython's ProteinAnalysis.secondary_structure_fraction.
HELIX_FORMERS = "EMALK"
TURN_FORMERS = "NPGSD"
SHEET_FORMERS = "VIYFWLT"

# Eisenberg hydrophobic-moment window (residues) — a standard choice for
# amphipathic helix detection.
HYDROPHOBIC_MOMENT_WINDOW = 11
HYDROPHOBIC_MOMENT_ANGLE = 100.0  # degrees per residue for an ideal alpha helix

#: Emoji-free colour per amino-acid class, used consistently across the UI.
AA_CLASS_COLOR = {
    "hydrophobic": "#f0a13a", "aromatic": "#c77dff", "polar": "#3ddc97",
    "positive": "#4da3ff", "negative": "#ff6b6b", "special": "#9aa5b1",
}

#: Water molecule mass subtracted when going residue -> free amino acid.
WATER_MASS = 18.0153
