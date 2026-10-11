"""Interactive 3D molecular viewer, built on a vendored copy of 3Dmol.js.

Why vendored: the app must work on a machine with no internet access, and the
component iframe that Streamlit renders has its own document, so a CDN ``<script
src>`` is both a privacy leak and a hard dependency. ``static/3Dmol-min.js``
(537 KB, 3Dmol.js 2.5.5, BSD-3) is inlined into the component HTML instead.

The component is fully self-contained: its toolbar (representation, colouring,
spin, reset, screenshot, fullscreen) is plain JavaScript inside the iframe, so
switching a representation costs no Streamlit rerun at all.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import streamlit as st

from .theme import PALETTE, SS_COLOURS

STATIC_DIR = Path(__file__).resolve().parents[1] / "static"


@lru_cache(maxsize=1)
def _viewer_js() -> str:
    """Read the vendored 3Dmol bundle once per process."""
    path = STATIC_DIR / "3Dmol-min.js"
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def viewer_available() -> bool:
    return bool(_viewer_js())


def _component_html(
    models: Sequence[Dict[str, str]],
    *,
    height: int,
    background: str,
    style: str,
    spin: bool,
    caption: str,
    legend: Sequence[Tuple[str, str]],
    autozoom: bool,
) -> str:
    js = _viewer_js()
    payload = json.dumps({"models": list(models), "style": style, "spin": spin,
                          "legend": [list(x) for x in legend], "autozoom": autozoom},
                         ensure_ascii=False)
    legend_html = "".join(
        f'<span class="lg"><i style="background:{colour}"></i>{label}</span>' for label, colour in legend
    )
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>
  html, body {{ margin:0; padding:0; background:{background}; color:{PALETTE['text']};
    font-family:-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }}
  #wrap {{ position:relative; width:100%; height:{height}px; overflow:hidden; border-radius:12px; }}
  #canvas {{ width:100%; height:100%; }}
  #bar {{ position:absolute; top:8px; left:8px; display:flex; gap:6px; flex-wrap:wrap;
          background:rgba(11,15,20,0.78); border:1px solid {PALETTE['border']}; border-radius:10px;
          padding:6px 8px; backdrop-filter: blur(6px); max-width: calc(100% - 16px); }}
  #bar select, #bar button {{ background:{PALETTE['surface_2']}; color:{PALETTE['text']};
          border:1px solid {PALETTE['border']}; border-radius:7px; padding:4px 8px; font-size:11.5px; cursor:pointer; }}
  #bar button:hover, #bar select:hover {{ border-color:{PALETTE['accent']}; color:{PALETTE['accent_soft']}; }}
  #legend {{ position:absolute; bottom:8px; left:8px; display:flex; gap:10px; flex-wrap:wrap;
             background:rgba(11,15,20,0.78); border:1px solid {PALETTE['border']}; border-radius:10px;
             padding:5px 9px; font-size:11px; }}
  .lg {{ display:flex; align-items:center; gap:5px; color:{PALETTE['text_dim']}; }}
  .lg i {{ width:9px; height:9px; border-radius:3px; display:inline-block; }}
  #cap {{ position:absolute; top:9px; right:10px; font-size:11px; color:{PALETTE['text_dim']};
          background:rgba(11,15,20,0.7); border-radius:8px; padding:4px 8px; max-width:45%;
          overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
  #err {{ padding:24px; color:{PALETTE['bad']}; font-size:13px; }}
</style></head>
<body>
<div id="wrap">
  <div id="canvas"></div>
  <div id="bar">
    <select id="style" title="Representation">
      <option value="cartoon">Cartoon</option>
      <option value="cartoon+sticks">Cartoon + binding-site sticks</option>
      <option value="sticks">Sticks</option>
      <option value="sphere">Space-filling</option>
      <option value="line">Lines</option>
    </select>
    <select id="colour" title="Colouring">
      <option value="ss">By secondary structure</option>
      <option value="chain">By chain</option>
      <option value="spectrum">Rainbow (N→C)</option>
      <option value="b">By confidence / B-factor</option>
      <option value="hydro">By hydrophobicity</option>
    </select>
    <button id="spin">Spin</button>
    <button id="pockets">Cavities</button>
    <button id="reset">Reset</button>
    <button id="png">PNG</button>
    <button id="full">Fullscreen</button>
  </div>
  <div id="cap">{caption}</div>
  <div id="legend">{legend_html}</div>
</div>
<script>{js}</script>
<script>
const DATA = {payload};
const wrap = document.getElementById('wrap');
let viewer = null, highlightPockets = true, spinning = false;

function hexToInt(h) {{ return parseInt(h.replace('#',''), 16); }}

function styleFor(mode) {{
  const ss = {{ 'H': hexToInt('{SS_COLOURS["H"]}'), 'G': hexToInt('{SS_COLOURS["G"]}'),
                'E': hexToInt('{SS_COLOURS["E"]}'), 'B': hexToInt('{SS_COLOURS["B"]}'),
                'T': hexToInt('{SS_COLOURS["T"]}'), 'C': hexToInt('{SS_COLOURS["C"]}') }};
  const colour = document.getElementById('colour').value;
  const cartoon = {{ cartoon: {{ color: 'spectrum' }} }};
  if (colour === 'ss') cartoon.cartoon.color = 'ss';
  else if (colour === 'chain') cartoon.cartoon.colorscheme = 'chainHetatm';
  else if (colour === 'spectrum') cartoon.cartoon.color = 'spectrum';
  else if (colour === 'b') cartoon.cartoon.color = 'roygb';
  else if (colour === 'hydro') cartoon.cartoon.color = 'whiteCarbon';
  if (mode === 'sticks') return {{ stick: {{ colorscheme: 'Jmol', radius: 0.14 }} }};
  if (mode === 'sphere') return {{ sphere: {{ scale: 0.9 }} }};
  if (mode === 'line') return {{ line: {{ linewidth: 2 }} }};
  if (mode === 'cartoon+sticks') return null;   // handled explicitly
  return {{ cartoon: cartoon.cartoon }};
}}

function applyStyles() {{
  if (!viewer) return;
  const mode = document.getElementById('style').value;
  viewer.setStyle({{}}, {{}});
  DATA.models.forEach(function (m, idx) {{
    const sel = m.kind === 'ligand' ? {{ model: idx }} : {{ model: idx }};
    if (m.kind === 'ligand') {{
      viewer.setStyle(sel, {{ stick: {{ colorscheme: 'greenCarbon', radius: 0.18 }},
                              sphere: {{ scale: 0.32 }} }});
    }} else if (mode === 'cartoon+sticks') {{
      viewer.setStyle(sel, {{ cartoon: {{ color: document.getElementById('colour').value === 'ss' ? 'ss' : 'spectrum' }} }});
      viewer.addStyle({{ model: idx, not: {{ resn: '__none__' }} }}, {{ stick: {{ hidden: true }} }});
      // Binding-site residues are added as their own model by the caller, so a
      // simple stick overlay for everything within the site is enough here.
      viewer.addStyle({{ model: idx, atom: m.site_atoms || [] }},
                      {{ stick: {{ colorscheme: 'cyanCarbon', radius: 0.16 }} }});
    }} else {{
      const s = styleFor(mode);
      if (s) viewer.setStyle(sel, s);
    }}
    if (m.kind === 'pocket' && highlightPockets) {{
      viewer.addStyle(sel, {{ sphere: {{ radius: m.radius || 1.4, opacity: 0.30,
                                         color: hexToInt('{PALETTE["warn"]}') }} }});
    }} else if (m.kind === 'pocket') {{
      // Hide pocket probes when toggled off.
      viewer.setStyle(sel, {{ sphere: {{ hidden: true }} }});
    }}
  }});
  viewer.render();
}}

function boot() {{
  if (typeof $3Dmol === 'undefined') {{
    document.getElementById('canvas').outerHTML = '<div id="err">3Dmol.js could not be loaded: ' +
      'bionemo_web/static/3Dmol-min.js is missing from this installation.</div>';
    return;
  }}
  try {{
    viewer = $3Dmol.createViewer('canvas', {{ backgroundColor: '{background}', antialias: true }});
    DATA.models.forEach(function (m) {{
      const fmt = m.format || 'pdb';
      const model = viewer.addModel(m.data, fmt);
      if (m.kind === 'ligand') {{ model.setStyle({{}}, {{ stick: {{ colorscheme: 'greenCarbon', radius: 0.18 }} }}); }}
      if (m.kind === 'pocket') {{ model.setStyle({{}}, {{ sphere: {{ radius: m.radius || 1.4, opacity: 0.28 }} }}); }}
      if (m.kind === 'site') {{ model.setStyle({{}}, {{ stick: {{ colorscheme: 'cyanCarbon', radius: 0.16 }} }}); }}
    }});
    applyStyles();
    viewer.zoomTo();
    if (DATA.autozoom) viewer.zoom(1.08);
    viewer.render();
    if (DATA.spin) {{ viewer.spin('y'); spinning = true; }}
  }} catch (e) {{
    document.getElementById('canvas').outerHTML = '<div id="err">Viewer error: ' + e + '</div>';
  }}
}}

document.getElementById('style').addEventListener('change', applyStyles);
document.getElementById('colour').addEventListener('change', applyStyles);
document.getElementById('spin').addEventListener('click', function () {{
  if (!viewer) return;
  spinning = !spinning;
  if (spinning) viewer.spin('y'); else viewer.spin(false);
}});
document.getElementById('pockets').addEventListener('click', function () {{
  highlightPockets = !highlightPockets; applyStyles();
}});
document.getElementById('reset').addEventListener('click', function () {{
  if (viewer) {{ viewer.zoomTo(); viewer.render(); }}
}});
document.getElementById('png').addEventListener('click', function () {{
  if (!viewer) return;
  const canvas = document.querySelector('#canvas canvas');
  if (!canvas) return;
  const a = document.createElement('a');
  a.href = canvas.toDataURL('image/png');
  a.download = 'bionemo-view.png';
  a.click();
}});
document.getElementById('full').addEventListener('click', function () {{
  const el = document.getElementById('wrap');
  if (!document.fullscreenElement) {{ el.requestFullscreen && el.requestFullscreen(); }}
  else {{ document.exitFullscreen(); }}
  setTimeout(function () {{ if (viewer) {{ viewer.resize(); viewer.render(); }} }}, 250);
}});
window.addEventListener('resize', function () {{ if (viewer) {{ viewer.resize(); viewer.render(); }} }});
boot();
</script></body></html>"""


def show_viewer(
    *,
    pdb: str = "",
    ligand_molblocks: Optional[Sequence[Tuple[str, str]]] = None,
    pocket_probes: Optional[Sequence[Tuple[List[float], float, str]]] = None,
    site_pdb: str = "",
    height: int = 460,
    caption: str = "",
    style: str = "cartoon",
    spin: bool = False,
    legend: Sequence[Tuple[str, str]] = (),
) -> None:
    """Render a structure, optionally with ligands and cavity probes.

    ``ligand_molblocks`` is a sequence of ``(name, mol block)``;
    ``pocket_probes`` is ``(centre xyz, radius, label)``; ``site_pdb`` is an
    optional PDB of just the binding-site residues, drawn as sticks.
    """
    if not viewer_available():
        st.warning(
            "The 3D viewer needs `bionemo_web/static/3Dmol-min.js`. "
            "Re-run the setup step (`bash scripts/setup.sh`) to vendor it."
        )
        st.code(pdb[:4000] or "(no structure)", language="text")
        return

    models: List[Dict[str, str]] = []
    if pdb:
        models.append({"data": pdb, "format": "pdb", "kind": "structure"})
    for name, block in ligand_molblocks or []:
        if block:
            models.append({"data": block, "format": "sdf", "kind": "ligand"})
    if site_pdb:
        models.append({"data": site_pdb, "format": "pdb", "kind": "site"})
    for centre, radius, label in pocket_probes or []:
        # A single pseudo-atom PDB record is the simplest way to place a probe
        # sphere in 3Dmol without a separate geometric primitive API.
        x, y, z = centre
        probe = (
            f"HETATM    1  X   PRB A   1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          X\n"
            "END\n"
        )
        models.append({"data": probe, "format": "pdb", "kind": "pocket",
                       "radius": max(0.6, float(radius)), "label": label})

    html = _component_html(
        models, height=height, background=PALETTE["bg_soft"], style=style, spin=spin,
        caption=caption, legend=legend, autozoom=True,
    )
    st.components.v1.html(html, height=height + 8, scrolling=False)


def show_pose_viewer(
    protein_pdb: str,
    ligand_smiles: str,
    poses: Sequence,
    *,
    selected: int = 0,
    pockets: Optional[Sequence[Dict]] = None,
    height: int = 460,
) -> None:
    """Show one selected docking pose of a ligand inside the receptor."""
    from ..bio import chem as chem_mod

    models_ligands: List[Tuple[str, str]] = []
    pose = None
    for index, candidate in enumerate(poses):
        if getattr(candidate, "rank", index + 1) == selected:
            pose = candidate
            break
    if pose is None and poses:
        pose = poses[0]
    if pose is not None:
        block = chem_mod.mol_block_from_coords(ligand_smiles, pose.coords) if hasattr(
            chem_mod, "mol_block_from_coords") else ""
        if not block:
            from ..engine import export as export_mod

            block = export_mod._mol_block_from_coords(ligand_smiles, pose.coords)
        models_ligands.append((f"pose {pose.rank}", block))

    probes = []
    if pockets:
        for pocket in pockets[:4]:
            radius = max(1.2, (float(pocket.get("volume_a3") or 300.0) / 6.0) ** (1 / 3))
            probes.append((pocket.get("centre", [0, 0, 0]), radius, pocket.get("id", "")))

    # Binding-site residues of the selected pose, extracted as their own model so
    # they can be drawn as sticks without touching the cartoon.
    site_pdb = ""
    if pose is not None and pose.interaction_residues:
        wanted = {r[1:] for r in pose.interaction_residues if len(r) > 1}
        lines = []
        for line in protein_pdb.splitlines():
            if line.startswith(("ATOM", "HETATM")) and line[22:27].strip() in wanted:
                lines.append(line)
        if lines:
            site_pdb = "\n".join(lines) + "\nEND\n"

    show_viewer(
        pdb=protein_pdb,
        ligand_molblocks=models_ligands,
        pocket_probes=probes if pockets else None,
        site_pdb=site_pdb,
        height=height,
        caption=(
            f"pose {pose.rank}: confidence {pose.confidence:.2f}, {pose.hbonds} H-bond(s), "
            f"{pose.clashes} clash(es)" if pose is not None else ""
        ),
        legend=[("receptor", PALETTE["text_dim"]), ("ligand", PALETTE["accent"]),
                ("binding site", PALETTE["info"]), ("cavity probe", PALETTE["warn"])],
        style="cartoon+sticks",
    )
