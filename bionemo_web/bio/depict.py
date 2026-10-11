"""Dependency-free 2D molecule depictions.

RDKit's own drawing code renders through Cairo, which drags in system libraries
(``libXrender``, ``libcairo``) that are frequently missing on headless servers --
including, as it happens, this development sandbox. Rather than make the console
fail to draw a molecule because a system library is absent, this module uses
RDKit for what it is genuinely good at (perception, layout via
``Compute2DCoords``) and emits the SVG itself.

What it draws: bond skeleton with double/triple bonds as parallel lines, aromatic
rings, element labels for every heteroatom and terminal carbon, implicit hydrogen
counts where chemistry demands them, and a title/subtitle line. What it does not
draw: wedge/dash stereochemistry, atom colouring by partial charge, or
publication-grade typography. It is a depiction, not a journal figure -- and it
renders identically in every browser with no server-side graphics stack.
"""

from __future__ import annotations

import html
import math
from typing import Dict, List, Optional, Sequence, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem

#: Covalent radii (Angstrom) used only to decide whether a bond is worth drawing.
_ELEMENT_COLOURS = {
    "O": "#e04b4b", "N": "#3b6fd4", "S": "#c9a227", "F": "#3fa34d",
    "Cl": "#3fa34d", "Br": "#9c4a1a", "I": "#7d3fbf", "P": "#d97706",
    "C": "#1f2937",
}

_STYLE = """
text { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }
.atom-label { font-weight: 600; }
.smiles-caption { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
"""


def _element_colour(symbol: str) -> str:
    return _ELEMENT_COLOURS.get(symbol, "#6b7280")


def molecule_svg(
    molecule: "Chem.Mol",
    *,
    width: int = 320,
    height: int = 240,
    title: str = "",
    subtitle: str = "",
    show_carbon_labels: bool = False,
    background: str = "#ffffff",
) -> str:
    """Render ``molecule`` as a standalone SVG string."""
    try:
        mol = Chem.Mol(molecule)
        if mol.GetNumConformers() == 0 or mol.GetConformer().Is3D():
            AllChem.Compute2DCoords(mol)
        Chem.rdDepictor.StraightenDepiction(mol)
    except Exception:
        try:
            AllChem.Compute2DCoords(mol)
        except Exception:
            return _error_svg(width, height, "could not lay out this molecule")

    conf = mol.GetConformer()
    positions = {
        atom.GetIdx(): (conf.GetAtomPosition(atom.GetIdx()).x, conf.GetAtomPosition(atom.GetIdx()).y)
        for atom in mol.GetAtoms()
    }
    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    span_x = (max(xs) - min(xs)) or 1.0
    span_y = (max(ys) - min(ys)) or 1.0

    pad_top = 34 if title else 18
    pad_bottom = 34 if subtitle else 16
    usable_w = width - 40
    usable_h = height - pad_top - pad_bottom
    scale = min(usable_w / span_x, usable_h / span_y) * 0.92
    # Keep line weights and labels legible at any size.
    stroke = max(1.1, min(2.4, 9.0 / (scale / 20.0 + 1.0)))
    font = max(9.0, min(14.0, 7.0 + scale * 0.30))
    half = font * 0.34

    def tx(x: float, y: float) -> Tuple[float, float]:
        return (
            20 + (x - (min(xs) + max(xs)) / 2) * scale + usable_w / 2,
            pad_top + (-y + (min(ys) + max(ys)) / 2) * scale + usable_h / 2,
        )

    parts: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="2D structure {html.escape(title or "molecule")}">',
        f"<style>{_STYLE}</style>",
        f'<rect width="{width}" height="{height}" rx="8" fill="{background}"/>',
    ]

    # Aromatic rings: draw the inner circle rather than alternating double bonds,
    # which is both simpler and how chemists read a benzene.
    aromatic_rings: List[Sequence[int]] = []
    for ring in mol.GetRingInfo().AtomRings():
        if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring) and len(ring) <= 7:
            aromatic_rings.append(ring)
    ring_atoms = {i for ring in aromatic_rings for i in ring}
    ring_centres = {}
    for ring in aromatic_rings:
        cx = sum(positions[i][0] for i in ring) / len(ring)
        cy = sum(positions[i][1] for i in ring) / len(ring)
        radius = min(math.dist(positions[i], (cx, cy)) for i in ring) * 0.62
        ring_centres[tuple(sorted(ring))] = (cx, cy, radius)

    # Bonds
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        x1, y1 = tx(*positions[i])
        x2, y2 = tx(*positions[j])
        order = bond.GetBondTypeAsDouble()
        in_aromatic_ring = i in ring_atoms and j in ring_atoms
        if in_aromatic_ring and order == 1.5:
            order = 1.0
        if order >= 1.5:
            # Parallel offset line for a double bond, shortened so it does not
            # overshoot into neighbouring bonds.
            dx, dy = x2 - x1, y2 - y1
            length = math.hypot(dx, dy) or 1.0
            ox, oy = -dy / length * (stroke * 1.9), dx / length * (stroke * 1.9)
            shrink = 0.16
            ax, ay = x1 + dx * shrink, y1 + dy * shrink
            bx, by = x2 - dx * shrink, y2 - dy * shrink
            parts.append(
                f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                f'stroke="#111827" stroke-width="{stroke:.2f}" stroke-linecap="round"/>'
            )
            parts.append(
                f'<line x1="{ax + ox:.1f}" y1="{ay + oy:.1f}" x2="{bx + ox:.1f}" y2="{by + oy:.1f}" '
                f'stroke="#111827" stroke-width="{stroke * 0.85:.2f}" stroke-linecap="round"/>'
            )
        else:
            parts.append(
                f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                f'stroke="#111827" stroke-width="{stroke:.2f}" stroke-linecap="round"/>'
            )

    # Aromatic ring circles
    for ring in aromatic_rings:
        cx, cy, radius = ring_centres[tuple(sorted(ring))]
        px, py = tx(cx, cy)
        parts.append(
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{radius * scale * 0.85:.1f}" fill="none" '
            f'stroke="#111827" stroke-width="{stroke * 0.7:.2f}"/>'
        )

    # Atom labels
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        symbol = atom.GetSymbol()
        charge = atom.GetFormalCharge()
        label = symbol
        hs = atom.GetTotalNumHs()
        draw_hetero = symbol != "C"
        draw_carbon = show_carbon_labels or (symbol == "C" and atom.GetDegree() == 0)
        draw_carbon = draw_carbon or (symbol == "C" and atom.GetIsAromatic() is False and atom.GetDegree() == 1
                                      and atom.GetTotalNumHs() == 3)
        if not (draw_hetero or draw_carbon):
            continue
        if hs and symbol in ("O", "N", "S"):
            label += "H" if hs == 1 else f"H{hs}"
        if charge:
            label += ("+" if charge > 0 else "−") + (str(abs(charge)) if abs(charge) > 1 else "")
        px, py = tx(*positions[idx])
        parts.append(
            f'<rect x="{px - len(label) * half - 1:.1f}" y="{py - font * 0.62:.1f}" '
            f'width="{len(label) * half * 2 + 2:.1f}" height="{font * 1.28:.1f}" rx="3" '
            f'fill="{background}" opacity="0.88"/>'
        )
        parts.append(
            f'<text class="atom-label" x="{px:.1f}" y="{py + font * 0.34:.1f}" '
            f'font-size="{font:.1f}" text-anchor="middle" fill="{_element_colour(symbol)}">{html.escape(label)}</text>'
        )

    if title:
        parts.append(
            f'<text class="smiles-caption" x="12" y="20" font-size="12.5" fill="#111827">'
            f"{html.escape(title[:44])}</text>"
        )
    if subtitle:
        parts.append(
            f'<text x="12" y="{height - 10}" font-size="11.5" fill="#6b7280">'
            f"{html.escape(subtitle[:70])}</text>"
        )
    parts.append("</svg>")
    return "".join(parts)


def _error_svg(width: int, height: int, message: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        f'<rect width="{width}" height="{height}" rx="8" fill="#f9fafb" stroke="#e5e7eb"/>'
        f'<text x="{width / 2:.0f}" y="{height / 2:.0f}" font-size="12" fill="#9ca3af" '
        f'text-anchor="middle">{html.escape(message)}</text></svg>'
    )


def smiles_svg(smiles: str, **kwargs) -> str:
    """Convenience wrapper: SMILES in, SVG out (never raises)."""
    mol, warnings = None, []
    try:
        from . import chem as chem_mod

        mol, warnings = chem_mod.parse_smiles(smiles)
    except Exception:
        mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return _error_svg(kwargs.get("width", 320), kwargs.get("height", 240),
                          f"invalid SMILES: {str(warnings[0] if warnings else smiles)[:40]}")
    return molecule_svg(mol, **kwargs)


def reaction_svg(reactant_smiles: str, product_smiles: str, **kwargs) -> str:
    """Two depictions side by side with an arrow -- used to show a seed and a hit."""
    left = smiles_svg(reactant_smiles, **{**kwargs, "title": kwargs.get("title", "") or "seed"})
    right = smiles_svg(product_smiles, **{**kwargs, "title": "analogue"})
    width = kwargs.get("width", 320)
    height = kwargs.get("height", 240)
    arrow = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="56" height="{height}">'
        f'<line x1="6" y1="{height / 2}" x2="44" y2="{height / 2}" stroke="#9ca3af" stroke-width="2"/>'
        f'<polygon points="44,{height / 2 - 6} 56,{height / 2} 44,{height / 2 + 6}" fill="#9ca3af"/>'
        f"</svg>"
    )
    return (
        f'<div style="display:flex;align-items:center;gap:6px">'
        f'<div style="flex:0 0 {width}px">{left}</div>'
        f'<div style="flex:0 0 56px">{arrow}</div>'
        f'<div style="flex:0 0 {width}px">{right}</div></div>'
    )
