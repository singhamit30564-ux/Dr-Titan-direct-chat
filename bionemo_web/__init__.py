"""Dr-Titan BioNeMo Web Console.

Package layout:

* ``bio``      -- pure science (sequence, chemistry, structure, depiction)
* ``nims``     -- model layer: catalogue, HTTP client, backend policy, simulators
* ``engine``   -- orchestration: router, tools, agent, jobs, store, export, LLM
* ``ui``       -- Streamlit rendering (theme, viewer, plots, cards, chat)
* ``console``  -- the non-chat pages
* ``static``   -- vendored 3Dmol.js viewer

Importing the package configures two things globally that would otherwise leak
into the user's console output on every keystroke:

* RDKit's C++ logger is quietened. The router legitimately asks RDKit whether a
  token looks like a molecule, which means English words reach the SMILES parser
  and would otherwise print parse errors for "is", "the", "backbone", and so on.
* Nothing else: no environment variables are read at import time, and no network
  or filesystem work happens until a component is actually used.
"""

from __future__ import annotations

__version__ = "1.0.0"

try:  # pragma: no cover - depends on the RDKit build
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.info")
    RDLogger.DisableLog("rdApp.warning")
    RDLogger.DisableLog("rdApp.debug")
except Exception:
    # RDKit is a hard requirement for the app, but the package must stay
    # importable (e.g. for packaging metadata) if it is somehow missing.
    pass
