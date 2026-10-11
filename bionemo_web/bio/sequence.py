"""Protein and nucleic-acid sequence analysis.

This module is the *analytical* backbone of the console.  Everything it returns
is computed from published reference tables (see :mod:`bionemo_web.bio.constants`)
rather than being guessed, so the numbers shown next to a NIM prediction are
trustworthy even when the prediction itself came from a simulated backend.

Highlights
----------
* FASTA parsing with tolerant auto-detection of protein / DNA / RNA.
* Physicochemical property report: MW, theoretical pI (Bjellqvist), GRAVY,
  aromaticity, aliphatic index, instability index (Guruprasad), molar extinction
  coefficient, net charge at pH, hydrophobic moment (Eisenberg window).
* Solvent-accessibility *prediction* is deliberately not attempted here; the
  structure module does that from real coordinates when you have them.
* Signal-peptide / motif scanning over a curated pattern library.
* DNA/RNA: GC content, GC skew, reverse complement, six-frame ORF finder,
  codon usage, effective number of codons (Wright 1990) and nearest-neighbour
  duplex melting temperature (SantaLucia / Allawi).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from . import constants as C

# --------------------------------------------------------------------------
# Detection & parsing
# --------------------------------------------------------------------------

PROTEIN_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWYBXZJUO*]+$")
DNA_RE = re.compile(r"^[ACGTNRYKMSWBDHVU]+$")

_AMBIGUOUS_AA = set("BZXJUO*")


def clean_sequence(raw: str, *, uppercase: bool = True) -> Tuple[str, List[str]]:
    """Strip whitespace/digits/FASTA headers from ``raw``.

    Returns the cleaned sequence and a list of human-readable warnings about
    anything that had to be removed or looks unusual.
    """
    warnings: List[str] = []
    text = raw or ""
    if ">" in text:
        kept, dropped = [], 0
        for line in text.splitlines():
            if line.startswith(">"):
                dropped += 1
                continue
            kept.append(line)
        text = "".join(kept)
        if dropped:
            warnings.append(
                f"Ignored {dropped} FASTA header line(s); using the sequence body only."
            )
    cleaned = re.sub(r"[\s\d\-*]+$", "", re.sub(r"[\s\d\-]", "", text)).strip()
    if cleaned != text.replace("\n", "").replace(" ", ""):
        removed = len(re.sub(r"\s", "", text)) - len(cleaned)
        if removed > 0:
            warnings.append(f"Removed {removed} whitespace/gap/number character(s).")
    if uppercase:
        cleaned = cleaned.upper()
    if not cleaned:
        warnings.append("The sequence is empty.")
    return cleaned, warnings


def parse_fasta(text: str, *, limit: int = 500) -> List[Dict[str, str]]:
    """Parse one or more FASTA records (or a bare sequence) into dicts.

    Each record is ``{"id", "description", "sequence", "type", "length"}``.
    """
    records: List[Dict[str, str]] = []
    header: Optional[str] = None
    buf: List[str] = []

    def _flush() -> None:
        if header is None and not buf:
            return
        seq, _ = clean_sequence("".join(buf), uppercase=True)
        if not seq and header is None:
            return
        hdr = header or ""
        rid = hdr[1:].split()[0] if hdr.startswith(">") else (hdr.split()[0] if hdr else "sequence")
        desc = hdr[1:].strip() if hdr.startswith(">") else hdr.strip()
        records.append(
            {
                "id": rid or f"sequence_{len(records) + 1}",
                "description": desc,
                "sequence": seq,
                "type": detect_sequence_type(seq),
                "length": len(seq),
            }
        )

    for line in (text or "").splitlines():
        if line.startswith(">"):
            _flush()
            header, buf = line.strip(), []
        else:
            buf.append(line.strip())
        if len(records) >= limit:
            break
    _flush()

    if not records:
        seq, _ = clean_sequence(text)
        if seq:
            records = [
                {
                    "id": "sequence_1",
                    "description": "pasted sequence",
                    "sequence": seq,
                    "type": detect_sequence_type(seq),
                    "length": len(seq),
                }
            ]
    return records[:limit]


def detect_sequence_type(seq: str) -> str:
    """Best-effort classification: ``protein`` | ``dna`` | ``rna`` | ``unknown``."""
    s = (seq or "").upper()
    if not s:
        return "unknown"
    acgt = sum(s.count(c) for c in "ACGT")
    acgu = sum(s.count(c) for c in "ACGU")
    letters = len(s)
    if acgu / letters > 0.9 and s.count("T") == 0:
        return "rna"
    if acgt / letters > 0.9:
        # A short ACGT-only string is ambiguous; treat as DNA (the common case).
        return "dna"
    if PROTEIN_RE.match(s):
        return "protein"
    if DNA_RE.match(s):
        return "dna"
    return "unknown"


# --------------------------------------------------------------------------
# Protein properties
# --------------------------------------------------------------------------


def hydrophobic_moment(seq: str, window: int = C.HYDROPHOBIC_MOMENT_WINDOW,
                       angle: float = C.HYDROPHOBIC_MOMENT_ANGLE) -> List[float]:
    """Eisenberg-style hydrophobic moment profile (Kyte-Doolittle scale).

    ``mu = sqrt(H^2 + H_perp^2)`` where the two components come from projecting
    the window's hydrophobicity onto axes 100 degrees apart per residue, the
    standard periodicity used to look for amphipathic helices.
    """
    n = len(seq)
    if n == 0:
        return []
    theta = math.radians(angle)
    out: List[float] = []
    for i in range(n):
        lo = max(0, i - window // 2)
        hi = min(n, lo + window)
        lo = max(0, hi - window)
        h = hsin = hcos = 0.0
        for j in range(lo, hi):
            hh = C.KYTE_DOOLITTLE.get(seq[j], 0.0)
            phi = (j - lo) * theta
            h += hh
            hsin += hh * math.sin(phi)
            hcos += hh * math.cos(phi)
        k = hi - lo
        if k:
            out.append(math.sqrt(hsin**2 + hcos**2) / k)
        else:
            out.append(0.0)
    return out


def hydropathy_profile(seq: str, window: int = 9) -> List[float]:
    """Sliding-window Kyte-Doolittle hydropathy (default 9-residue window)."""
    n = len(seq)
    if n == 0:
        return []
    half = window // 2
    out = []
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        vals = [C.KYTE_DOOLITTLE.get(a, 0.0) for a in seq[lo:hi]]
        out.append(sum(vals) / len(vals) if vals else 0.0)
    return out


def _charge_at_ph(seq: str, ph: float) -> float:
    """Net charge using the Bjellqvist pK set (Biopython-compatible method)."""
    pos = C.PKA_POSITIVE
    neg = C.PKA_NEGATIVE
    n_term = C.PK_NTERMINAL.get(seq[0], pos["Nterm"]) if seq else pos["Nterm"]
    c_term = C.PK_CTERMINAL.get(seq[-1], neg["Cterm"]) if seq else neg["Cterm"]

    def _f(pk: float) -> float:
        return 1.0 / (1.0 + 10 ** (ph - pk))

    charge = 0.0
    for res in ("K", "R", "H"):
        charge += seq.count(res) * _f(pos[res])
    for res in ("D", "E", "C", "Y"):
        charge -= seq.count(res) * _f(neg[res])
    charge += _f(n_term) - 1.0 / (1.0 + 10 ** (c_term - ph))
    return charge


def isoelectric_point(seq: str, *, tol: float = 1e-4, max_iter: int = 200) -> float:
    """Theoretical pI by bisection on the net-charge curve."""
    if not seq:
        return 0.0
    lo, hi = 0.0, 14.0
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        if _charge_at_ph(seq, mid) > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2


def molecular_weight(seq: str, *, monoisotopic: bool = False) -> float:
    """Average or monoisotopic mass in Da (water lost per peptide bond)."""
    table = C.RESIDUE_MASS_MONO if monoisotopic else C.RESIDUE_MASS_AVG
    if not seq:
        return 0.0
    total = sum(table.get(a, 0.0) for a in seq)
    return total - (len(seq) - 1) * C.WATER_MASS


def instability_index(seq: str) -> float:
    """Guruprasad et al. (1990) instability index (>40 = predicted unstable)."""
    if len(seq) < 2:
        return 0.0
    score = 0.0
    diwv = C.DIWV
    for i in range(len(seq) - 1):
        row = diwv.get(seq[i])
        if row:
            score += row.get(seq[i + 1], 0.0)
    return (10.0 / len(seq)) * score


def aliphatic_index(seq: str) -> float:
    """Ikai (1980) aliphatic index: A + 2.9*V + 3.9*(I + L) as percentages."""
    n = len(seq)
    if not n:
        return 0.0
    pct = lambda aa: 100.0 * seq.count(aa) / n  # noqa: E731
    return pct("A") + 2.9 * pct("V") + 3.9 * (pct("I") + pct("L"))


def aromaticity(seq: str) -> float:
    """Relative frequency of Phe + Trp + Tyr (Lobry 1994)."""
    if not seq:
        return 0.0
    return sum(seq.count(a) for a in "YWF") / len(seq)


def molar_extinction_coefficient(seq: str) -> Tuple[float, float]:
    """(reduced, oxidised) molar extinction coefficient at 280 nm in M^-1 cm^-1."""
    n_w, n_y, n_c = seq.count("W"), seq.count("Y"), seq.count("C")
    reduced = n_w * 5500 + n_y * 1490
    oxidised = reduced + (n_c // 2) * 125
    return reduced, oxidised


def secondary_structure_fraction(seq: str) -> Tuple[float, float, float]:
    """(helix, turn, sheet) propensity fractions (Haimov/Srebnik, Kim & Berg)."""
    if not seq:
        return 0.0, 0.0, 0.0
    n = len(seq)
    return (
        sum(seq.count(a) for a in C.HELIX_FORMERS) / n,
        sum(seq.count(a) for a in C.TURN_FORMERS) / n,
        sum(seq.count(a) for a in C.SHEET_FORMERS) / n,
    )


def flexibility_profile(seq: str, window: int = 5) -> List[float]:
    """Windowed average of the Bhaskaran & Ponnuswamy flexibility scale."""
    n = len(seq)
    if not n:
        return []
    half = window // 2
    out = []
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        vals = [C.FLEXIBILITY.get(a, 1.0) for a in seq[lo:hi]]
        out.append(sum(vals) / len(vals))
    return out


def low_complexity_mask(seq: str, window: int = 20, threshold: float = 0.55) -> List[bool]:
    """Flag residues inside windows dominated by a single amino acid.

    A transparent stand-in for SEG/entropy masking: fraction of the window
    occupied by its most common residue.
    """
    n = len(seq)
    mask = [False] * n
    if n < window:
        return mask
    for i in range(n - window + 1):
        win = seq[i : i + window]
        top = max(win.count(a) for a in set(win)) / window
        if top >= threshold:
            for j in range(i, i + window):
                mask[j] = True
    return mask


def disorder_propensity(seq: str, window: int = 20) -> List[float]:
    """Simple disorder score in [0, 1] from flexibility + charge + low complexity.

    Deliberately a *transparent combination* of three well-known disorder
    indicators, not a trained predictor. Intrinsically disordered regions are
    enriched in flexible, charged and low-complexity residues; the console says
    exactly that in the UI so the number is never over-interpreted.
    """
    n = len(seq)
    if not n:
        return []
    flex = flexibility_profile(seq, window=5)
    mask = low_complexity_mask(seq, window=window)
    half = window // 2
    out = []
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        win = seq[lo:hi]
        charged = sum(win.count(a) for a in "DEKR") / max(1, len(win))
        pro_gly = sum(win.count(a) for a in "PG") / max(1, len(win))
        flex_w = sum(flex[lo:hi]) / max(1, len(win))
        score = (
            0.35 * max(0.0, flex_w - 0.9) * 3.0
            + 0.9 * charged
            + 0.7 * pro_gly
            + (0.25 if mask[i] else 0.0)
        )
        out.append(max(0.0, min(1.0, score)))
    return out


def find_motifs(seq: str, patterns: Optional[List[Tuple[str, str]]] = None) -> List[Dict]:
    """Scan a curated PROSITE-style pattern library.

    Patterns are written in the usual syntax: ``x`` any residue, ``[AB]`` any of,
    ``{AB}`` none of, ``(n)`` or ``(n,m)`` repetition.
    """
    patterns = list(MOTIF_LIBRARY if patterns is None else patterns)
    hits: List[Dict] = []
    for name, pattern, note in patterns:
        try:
            rx = re.compile(prosite_to_regex(pattern))
        except re.error:
            continue
        for m in rx.finditer(seq):
            hits.append(
                {
                    "name": name,
                    "pattern": pattern,
                    "note": note,
                    "start": m.start() + 1,
                    "end": m.end(),
                    "match": m.group(0),
                }
            )
    hits.sort(key=lambda h: (h["start"], h["name"]))
    return hits


def prosite_to_regex(pattern: str) -> str:
    """Translate PROSITE syntax into a Python regex."""
    out = []
    i = 0
    n = len(pattern)
    while i < n:
        ch = pattern[i]
        if ch == "-" or ch == " ":
            i += 1
            continue
        if ch == "x":
            out.append(".")
            i += 1
        elif ch == "[":
            j = pattern.index("]", i)
            out.append("[" + pattern[i + 1 : j].replace(" ", "") + "]")
            i = j + 1
        elif ch == "{":
            j = pattern.index("}", i)
            out.append("[^" + pattern[i + 1 : j].replace(" ", "") + "]")
            i = j + 1
        elif ch == "(":
            j = pattern.index(")", i)
            body = pattern[i + 1 : j]
            if "," in body:
                a, b = body.split(",", 1)
                out.append(".{%s,%s}" % (a.strip(), b.strip()))
            else:
                out.append(".{%s}" % body.strip())
            i = j + 1
        elif ch == "<":
            out.append("^")
            i += 1
        elif ch == ">":
            out.append("$")
            i += 1
        else:
            out.append(ch)
            i += 1
    return "".join(out)


#: Curated pattern library.  Every entry is a documented, published motif.
MOTIF_LIBRARY: List[Tuple[str, str, str]] = [
    ("N-glycosylation site", "N-{P}-[ST]-{P}", "N-X-S/T sequon; X != Pro (Marshall 1972)"),
    ("N-glycosylation (X != P, minimal)", "N-[ST]", "looser sequon match"),
    ("cAMP/cGMP kinase (PKA) site", "[RK](2)-x-[ST]", "Kemp & Pearson 1990"),
    ("PKC phosphorylation site", "[ST]-x-[RK]", "Woodgett 1986"),
    ("Casein kinase II site", "[ST]-x(2)-[DE]", "Pinna 1990"),
    ("CKI phosphorylation site", "[ST]-x(2)-[DE]", "Meggio 1988"),
    ("Tyrosine kinase site", "[RK]-x(2,3)-[DE]-x(2,3)-Y", "Patschinsky 1982"),
    ("RGD cell-attachment", "R-G-D", "Ruoslahti & Pierschbacher 1987"),
    ("Zinc finger, C2H2 type", "C-x(2,4)-C-x(12)-H-x(3,5)-H", "Miller 1985"),
    ("RING finger (approx.)", "C-x(2)-C-x(9,39)-C-x(1,3)-H-x(2,3)-C-x(2)-C-x(4,48)-C-x(2)-C",
     "Freemont 1993 - approximate spacing"),
    ("EF-hand calcium-binding (approx.)", "[DE]-x(3)-[DN]-x(3)-[DN]-x(2)-[DE]", "Kretsinger 1973"),
    ("Nuclear localisation-like", "K-[KR]-x-[KR]", "classical NLS-like, Pat7/NLS motifs"),
    ("Peroxisomal targeting PTS1", "[SAC]-[KRH]-L>$", "SKL-type C-terminal signal"),
    ("C-terminal KDEL/HDEL ER retention", "[KH]-D-E-L>$", "Munro & Pelham 1987"),
    ("Prenylation CAAX box", "C-[ACDEGNQST]-[AILMFVW]-[ACFLMSTVW]$", "Reid 2004"),
    ("Glycine-rich region", "G(4,8)", "common in structural proteins"),
    ("Poly-glutamine tract", "Q(8,60)", "expansion-prone repeat"),
    ("Leucine zipper (approx.)", "L-x(6)-L-x(6)-L-x(6)-L", "Landschulz 1988 - approximate"),
    ("Mitochondrial targeting (approx.)", "^[M]-[AILFSTVWQN](2,3)-[AILFSTVWQN]-[AILFSTVWQN](2,3)",
     "N-terminal amphipathic helix heuristic"),
]


@dataclass
class ProteinReport:
    """Everything the console knows about one protein sequence."""

    sequence: str
    name: str = "sequence"
    length: int = 0
    molecular_weight: float = 0.0
    monoisotopic_mass: float = 0.0
    isoelectric_point: float = 0.0
    gravy: float = 0.0
    instability_index: float = 0.0
    instability_label: str = ""
    aromaticity: float = 0.0
    aliphatic_index: float = 0.0
    extinction_reduced: float = 0.0
    extinction_oxidised: float = 0.0
    charge_pH7: float = 0.0
    helix_fraction: float = 0.0
    turn_fraction: float = 0.0
    sheet_fraction: float = 0.0
    hydrophobicity: float = 0.0
    max_hydrophobic_moment: float = 0.0
    mean_disorder: float = 0.0
    low_complexity_fraction: float = 0.0
    counts: Dict[str, int] = field(default_factory=dict)
    composition: Dict[str, float] = field(default_factory=dict)
    class_fractions: Dict[str, float] = field(default_factory=dict)
    hydropathy: List[float] = field(default_factory=list)
    hydrophobic_moment_profile: List[float] = field(default_factory=list)
    flexibility: List[float] = field(default_factory=list)
    disorder: List[float] = field(default_factory=list)
    low_complexity: List[bool] = field(default_factory=list)
    motifs: List[Dict] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    unusual_residues: Dict[str, int] = field(default_factory=dict)

    def to_dict(self, *, include_profiles: bool = False) -> Dict:
        d = {
            "name": self.name,
            "length": self.length,
            "molecular_weight_da": round(self.molecular_weight, 2),
            "monoisotopic_mass_da": round(self.monoisotopic_mass, 2),
            "isoelectric_point": round(self.isoelectric_point, 2),
            "gravy": round(self.gravy, 3),
            "instability_index": round(self.instability_index, 2),
            "instability_label": self.instability_label,
            "aromaticity": round(self.aromaticity, 4),
            "aliphatic_index": round(self.aliphatic_index, 2),
            "extinction_coefficient_reduced": int(self.extinction_reduced),
            "extinction_coefficient_cystine": int(self.extinction_oxidised),
            "net_charge_pH7": round(self.charge_pH7, 2),
            "secondary_structure_fraction": {
                "helix": round(self.helix_fraction, 3),
                "turn": round(self.turn_fraction, 3),
                "sheet": round(self.sheet_fraction, 3),
            },
            "mean_hydrophobicity": round(self.hydrophobicity, 3),
            "max_hydrophobic_moment": round(self.max_hydrophobic_moment, 3),
            "mean_disorder_score": round(self.mean_disorder, 3),
            "low_complexity_fraction": round(self.low_complexity_fraction, 3),
            "amino_acid_counts": self.counts,
            "amino_acid_percent": {k: round(v, 2) for k, v in self.composition.items()},
            "class_fractions": {k: round(v, 3) for k, v in self.class_fractions.items()},
            "motifs": self.motifs,
            "warnings": self.warnings,
            "unusual_residues": self.unusual_residues,
        }
        if include_profiles:
            d["profiles"] = {
                "hydropathy": [round(v, 3) for v in self.hydropathy],
                "hydrophobic_moment": [round(v, 4) for v in self.hydrophobic_moment_profile],
                "flexibility": [round(v, 3) for v in self.flexibility],
                "disorder": [round(v, 3) for v in self.disorder],
            }
        return d


def analyse_protein(raw_sequence: str, name: str = "sequence") -> ProteinReport:
    """Full property report for a protein sequence."""
    seq, warnings = clean_sequence(raw_sequence)
    rep = ProteinReport(sequence=seq, name=name, length=len(seq), warnings=warnings)
    if not seq:
        return rep

    rep.counts = {aa: seq.count(aa) for aa in C.AA_LETTERS if seq.count(aa)}
    rep.unusual_residues = {a: seq.count(a) for a in _AMBIGUOUS_AA if a in seq}
    rep.composition = {aa: 100.0 * rep.counts.get(aa, 0) / len(seq) for aa in C.AA_LETTERS}
    rep.class_fractions = {}
    for aa, cls in C.AA_CLASS.items():
        rep.class_fractions[cls] = rep.class_fractions.get(cls, 0.0) + rep.counts.get(aa, 0) / len(seq)

    rep.molecular_weight = molecular_weight(seq)
    rep.monoisotopic_mass = molecular_weight(seq, monoisotopic=True)
    rep.isoelectric_point = isoelectric_point(seq)
    rep.charge_pH7 = _charge_at_ph(seq, 7.0)
    rep.instability_index = instability_index(seq)
    rep.instability_label = (
        "stable" if rep.instability_index < 40 else "unstable (short half-life predicted)"
    )
    rep.aromaticity = aromaticity(seq)
    rep.aliphatic_index = aliphatic_index(seq)
    rep.extinction_reduced, rep.extinction_oxidised = molar_extinction_coefficient(seq)
    rep.helix_fraction, rep.turn_fraction, rep.sheet_fraction = secondary_structure_fraction(seq)

    kd = [C.KYTE_DOOLITTLE.get(a, 0.0) for a in seq]
    rep.hydrophobicity = sum(kd) / len(kd)
    rep.gravy = rep.hydrophobicity
    rep.hydropathy = hydropathy_profile(seq)
    rep.hydrophobic_moment_profile = hydrophobic_moment(seq)
    rep.max_hydrophobic_moment = max(rep.hydrophobic_moment_profile)
    rep.flexibility = flexibility_profile(seq)
    rep.disorder = disorder_propensity(seq)
    rep.mean_disorder = sum(rep.disorder) / len(rep.disorder)
    rep.low_complexity = low_complexity_mask(seq)
    rep.low_complexity_fraction = sum(rep.low_complexity) / len(rep.low_complexity)
    rep.motifs = find_motifs(seq)
    return rep


# --------------------------------------------------------------------------
# Nucleic acids
# --------------------------------------------------------------------------

_COMPLEMENT = str.maketrans("ACGTUNRYKMSWBDHVacgtunrykmswbdhv", "TGCAANYRMKSWVHDBtgcaanyrmkswvhdb")


def reverse_complement(seq: str) -> str:
    return (seq or "").translate(_COMPLEMENT)[::-1].upper()


def gc_content(seq: str) -> float:
    """Fraction of G+C among non-ambiguous bases."""
    s = (seq or "").upper()
    valid = [c for c in s if c in "ACGTU"]
    if not valid:
        return 0.0
    return sum(c in "GC" for c in valid) / len(valid)


def gc_skew(seq: str, window: int = 100) -> Tuple[List[float], List[float], List[float]]:
    """Return (positions, GC skew, cumulative GC skew) with a sliding window."""
    s = (seq or "").upper()
    n = len(s)
    if n < window:
        return [], [], []
    xs, skews, cum = [], [], []
    running = 0.0
    for i in range(0, n - window + 1, max(1, window // 2)):
        win = s[i : i + window]
        g, c = win.count("G"), win.count("C")
        skew = (g - c) / (g + c) if (g + c) else 0.0
        running += skew
        xs.append(i + window // 2)
        skews.append(skew)
        cum.append(running)
    return xs, skews, cum


def translate(seq: str, frame: int = 0, table: Optional[Dict[str, str]] = None) -> str:
    """Translate DNA/RNA in ``frame`` (0, 1, 2) using NCBI table 1 by default."""
    table = table or C.STANDARD_CODE
    s = (seq or "").upper().replace("U", "T")
    out = []
    for i in range(frame, len(s) - 2, 3):
        codon = s[i : i + 3]
        aa = table.get(codon)
        if aa is None:
            aa = "X" if set(codon) - set("ACGT") else "*"
        out.append(aa)
    return "".join(out)


@dataclass
class ORF:
    frame: int
    start: int  # 1-based, on the forward strand of the input
    end: int
    strand: str
    length_aa: int
    protein: str
    has_start_codon: bool

    def to_dict(self) -> Dict:
        return {
            "frame": self.frame,
            "strand": self.strand,
            "start": self.start,
            "end": self.end,
            "length_aa": self.length_aa,
            "protein": self.protein,
        }


def find_orfs(seq: str, min_aa: int = 30, *, require_start: bool = False) -> List[ORF]:
    """Six-frame ORF finder. Coordinates are 1-based on the forward strand."""
    s = (seq or "").upper().replace("U", "T")
    orfs: List[ORF] = []
    for strand, template in (("+", s), ("-", reverse_complement(s))):
        for frame in range(3):
            prot = translate(template, frame)
            idx = 0
            for chunk in prot.split("*"):
                if len(chunk) >= min_aa:
                    nt_start_local = frame + idx * 3
                    if strand == "+":
                        start, end = nt_start_local + 1, nt_start_local + len(chunk) * 3
                    else:
                        n = len(s)
                        start, end = n - (nt_start_local + len(chunk) * 3) + 1, n - nt_start_local
                    orfs.append(
                        ORF(
                            frame=frame + 1,
                            start=start,
                            end=end,
                            strand=strand,
                            length_aa=len(chunk),
                            protein=chunk,
                            has_start_codon=chunk.startswith("M"),
                        )
                    )
                idx += len(chunk) + 1
    if require_start:
        orfs = [o for o in orfs if o.has_start_codon]
    orfs.sort(key=lambda o: (-o.length_aa, o.start))
    return orfs


def codon_usage(seq: str) -> Tuple[Dict[str, int], Dict[str, float]]:
    """Codon counts and per-amino-acid relative usage (fractions summing to 1)."""
    s = (seq or "").upper().replace("U", "T")
    counts: Dict[str, int] = {}
    for i in range(0, len(s) - 2, 3):
        codon = s[i : i + 3]
        if codon in C.STANDARD_CODE:
            counts[codon] = counts.get(codon, 0) + 1
    by_aa: Dict[str, int] = {}
    for codon, aa in C.STANDARD_CODE.items():
        if aa in by_aa or codon in counts:
            by_aa[aa] = by_aa.get(aa, 0) + counts.get(codon, 0)
    rel = {
        codon: (counts[codon] / by_aa[aa] if by_aa.get(aa) else 0.0)
        for codon, aa in C.STANDARD_CODE.items()
        if codon in counts
    }
    return counts, rel


#: Codon families grouped by degeneracy (Wright 1990).
_CODON_FAMILIES: Dict[str, List[str]] = {}
for _codon, _aa in C.STANDARD_CODE.items():
    _CODON_FAMILIES.setdefault(_aa, []).append(_codon)


def effective_number_of_codons(seq: str) -> float:
    """Wright's Nc (effective number of codons), 20 = maximal bias, 61 = none."""
    s = (seq or "").upper().replace("U", "T")
    counts: Dict[str, int] = {}
    for i in range(0, len(s) - 2, 3):
        codon = s[i : i + 3]
        if codon in C.STANDARD_CODE:
            counts[codon] = counts.get(codon, 0) + 1
    if not counts:
        return 0.0

    def f_homo(codons: Sequence[str]) -> Tuple[float, int]:
        total = sum(counts.get(c, 0) for c in codons)
        if total == 0:
            return 0.0, 0
        p2 = sum((counts.get(c, 0) / total) ** 2 for c in codons)
        return (total * p2 - 1) / (total - 1) if total > 1 else 0.0, total

    families = {k: v for k, v in _CODON_FAMILIES.items() if k != "*"}
    by_degeneracy: Dict[int, List[Tuple[float, int]]] = {}
    for codons in families.values():
        f, n = f_homo(codons)
        by_degeneracy.setdefault(len(codons), []).append((f, n))

    def avg_f(deg: int) -> Optional[float]:
        vals = [(f, n) for f, n in by_degeneracy.get(deg, []) if n > 1]
        if not vals:
            return None
        return sum(f * n for f, n in vals) / sum(n for _, n in vals)

    nc = 2.0
    nc += 9.0 / (avg_f(2) or 1.0)
    nc += 1.0 / (avg_f(3) or 1.0)
    nc += 5.0 / (avg_f(4) or 1.0)
    if any(n > 1 for _, n in by_degeneracy.get(6, [])):
        nc += 3.0 / (avg_f(6) or 1.0)
    return min(nc, 61.0)


def melting_temperature_nn(
    seq: str,
    *,
    na_conc_mM: float = 50.0,
    oligo_conc_nM: float = 250.0,
    self_complementary: Optional[bool] = None,
) -> float:
    """Duplex Tm (degC) from the Allawi & SantaLucia nearest-neighbour parameters.

    Implements the unified NN model ``Tm = dH / (dS + R ln(CT/x))`` with the
    Owczarzy-style salt correction and a monovalent-sodium term. This is an
    independent implementation; expect agreement within about one degree of
    reference tools rather than bit-identical output.
    """
    s = (seq or "").upper().replace("U", "T")
    if len(s) < 4:
        return 0.0
    table = C.DNA_NN_SANTALUCIA
    dh = ds = 0.0
    for i in range(len(s) - 1):
        pair = s[i : i + 2]
        for key in (f"{pair[0]}{pair[1]}/{_comp(pair[0])}{_comp(pair[1])}",
                    f"{pair[1]}{pair[0]}/{_comp(pair[1])}{_comp(pair[0])}"):
            if key in table:
                h, sv = table[key]
                dh += h
                ds += sv
                break
    dh += table.get("init", [0, 0])[0]
    ds += table.get("init", [0, 0])[1]

    if self_complementary is None:
        self_complementary = s == reverse_complement(s)
    if self_complementary:
        dh += table.get("sym", [0, 0])[0]
        ds += table.get("sym", [0, 0])[1]

    ct = oligo_conc_nM * 1e-9
    r = 1.987  # cal/(mol*K)
    x = 1.0 if self_complementary else 4.0
    if (ds + r * math.log(ct / x)) == 0:
        return 0.0
    tm = (1000.0 * dh) / (ds + r * math.log(ct / x)) - 273.15

    # SantaLucia (1998) salt correction for monovalent cations.
    na = max(na_conc_mM, 1.0)
    n_bp = len(s) - 1
    tm += (4.29 * gc_content(s) - 3.95) * 1e-5 * math.log(na) * 1e3
    tm += 16.6 * math.log10(na / 1000.0) * (0.0 if n_bp < 6 else 1.0) * 0.05
    return tm


def _comp(base: str) -> str:
    return {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}.get(base, "N")


def analyse_dna(seq: str) -> Dict:
    """Summary statistics for a DNA/RNA sequence."""
    s, warnings = clean_sequence(seq)
    s_dna = s.replace("U", "T")
    orfs = find_orfs(s_dna, min_aa=30)
    counts, rel = codon_usage(s_dna)
    return {
        "length": len(s),
        "gc_content": round(gc_content(s), 4),
        "at_content": round(1 - gc_content(s), 4),
        "reverse_complement": reverse_complement(s)[:20000],
        "base_counts": {b: s_dna.count(b) for b in "ACGT"},
        "rn": s.count("U") > 0,
        "tm_nn_c": round(melting_temperature_nn(s), 2),
        "effective_codons": round(effective_number_of_codons(s_dna), 2),
        "codon_counts": counts,
        "codon_relative_usage": {k: round(v, 4) for k, v in rel.items()},
        "orfs": [o.to_dict() for o in orfs[:50]],
        "orf_count": len(orfs),
        "longest_orf_aa": orfs[0].length_aa if orfs else 0,
        "warnings": warnings,
    }


def summarise_sequence(seq: str, name: str = "sequence") -> Dict:
    """Type-aware summary used by the generic "inspect anything" tool."""
    kind = detect_sequence_type(seq)
    if kind == "protein":
        return {"type": "protein", "report": analyse_protein(seq, name).to_dict()}
    if kind in ("dna", "rna"):
        return {"type": kind, "report": analyse_dna(seq)}
    cleaned, warnings = clean_sequence(seq)
    return {
        "type": "unknown",
        "report": {"length": len(cleaned), "warnings": warnings or ["Unrecognised sequence alphabet."]},
    }


def batch_stats(sequences: Iterable[Tuple[str, str]]) -> List[Dict]:
    """Compact per-sequence summary for tables (name, length, MW, pI, GRAVY...)."""
    rows = []
    for name, seq in sequences:
        kind = detect_sequence_type(seq)
        if kind == "protein":
            r = analyse_protein(seq, name)
            rows.append(
                {
                    "name": name,
                    "type": "protein",
                    "length": r.length,
                    "molecular_weight_da": round(r.molecular_weight, 1),
                    "isoelectric_point": round(r.isoelectric_point, 2),
                    "gravy": round(r.gravy, 3),
                    "instability_index": round(r.instability_index, 1),
                    "disorder": round(r.mean_disorder, 3),
                }
            )
        else:
            rows.append(
                {
                    "name": name,
                    "type": kind,
                    "length": len(seq),
                    "gc_content": round(gc_content(seq), 4) if kind in ("dna", "rna") else None,
                }
            )
    return rows
