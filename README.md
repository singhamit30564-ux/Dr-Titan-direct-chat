# Dr-Titan · BioNeMo Web Console

A local, chat-first web console for **NVIDIA BioNeMo** — protein folding, molecule
generation, docking, sequence design, embeddings and genomics — running on a
Streamlit server on your own machine.

![status](https://img.shields.io/badge/tests-116%20passing-76b900) ![python](https://img.shields.io/badge/python-3.10%2B-76b900) ![gpu](https://img.shields.io/badge/GPU-not%20required-76b900)

> **Two sentences that matter.** BioNeMo's inference models ship as *NIM
> microservices* (GPU containers that serve HTTP); this console drives them. When
> no NIM is reachable it answers from its own documented models instead — and
> every single result says which one answered, in the UI, in the exports, and in
> the database. Nothing is ever silently simulated.

---

## Quick start

```bash
git clone <your-fork> dr-titan-bionemo-console
cd dr-titan-bionemo-console
bash scripts/setup.sh          # venv + requirements + vendors the 3D viewer
bash scripts/run_console.sh    # http://localhost:8501
```

That is the whole setup. **No GPU and no API key are needed to start** — the
console opens with its own simulation backends and tells you so on every card.

To run the real models, either start local NIM containers
(`scripts/start_nims.sh`, needs a GPU) or set an NGC key for NVIDIA's hosted
endpoints:

```bash
export NGC_API_KEY=nvapi-xxxxxxxx      # free key: https://build.nvidia.com
bash scripts/run_console.sh
```

Then open **Administration → Backends**, set one endpoint per model, press
*Probe every configured endpoint*, and the badges on your results flip from
`SIMULATED` to `LOCAL NIM` or `NVIDIA HOSTED`.

---

## What it looks like

| Page | What it is for |
| --- | --- |
| **Chat** | The main interface. Ask in plain language: *"fold MKTAYIAK…"*, *"generate 12 analogues of aspirin optimising QED"*, *"dock it into the last structure"*, *"run a virtual screen"*. Attach PDB/FASTA/SDF files. Long jobs run in the background with live progress. |
| **Structure Lab** | Fold with ESMFold / AlphaFold2 / OpenFold3 / Boltz-2, then measure: cavities, SASA, secondary structure, per-residue confidence, contact maps. Paste or upload a PDB to analyse one you already have. |
| **Molecule Workbench** | Generate analogues with MolMIM/GenMol, then judge them locally: 30 descriptors, QED, Ertl SA, Lipinski/Veber, PAINS/Brenk/NIH alerts, Murcko scaffolds, similarity matrix, clustering. |
| **Screening Pipeline** | The generative virtual-screening chain end to end: fold target → detect cavities → generate analogues → dock each one → rank by a documented composite of pose quality and drug-likeness. Exports ranked CSV, an SDF of the shortlist and a Markdown report. |
| **Administration** | An honest probe of this machine (GPU, driver, container toolkit, NGC key, PyTorch), endpoint registry with health checks, copy-paste `docker run` commands for every NIM, language-model settings, latency/call log, job monitor. |
| **History** | Every run with its full provenance manifest, every artifact, every conversation (Markdown/JSON export), and the SQLite database stats. |

Everything is interactive: a vendored 3Dmol.js viewer with representation and
colouring controls, Plotly charts, sortable tables and per-result downloads
(PDB, SDF, CSV, JSON, FASTA, Markdown).

---

## The models

| Task | Models | NIM container |
| --- | --- | --- |
| Structure prediction | ESMFold, AlphaFold2, OpenFold3, Boltz-2 | `nvcr.io/nim/meta/esmfold`, `.../deepmind/alphafold2` |
| Protein–ligand complexes | Boltz-2, OpenFold3 | `nvcr.io/nim/nvidia/boltz2` |
| Sequence design | ProteinMPNN, RFdiffusion, ESM-2 generate | `nvcr.io/nim/ipd/proteinmpnn` |
| Embeddings | ESM-2, AMPLIFY, Geneformer | `nvcr.io/nim/meta/esm2-650m` |
| Molecule generation | MolMIM, GenMol | `nvcr.io/nim/nvidia/molmim` |
| Docking | DiffDock | `nvcr.io/nim/mit/diffdock` |
| Genomics | Evo 2, CodonFM | `nvcr.io/nim/arc/evo2-40b` |
| Protein search | MSA Search | `nvcr.io/nim/nvidia/msa-search` |

Each entry carries its endpoint path, parameter schema and a
`path_confidence` marker (`documented` / `configurable` / `verified`), because
NVIDIA renames paths between NIM releases — the client also discovers the real
path from `/docs` and `/v1/models` at runtime.

**Looking for the *training* code?** That is the BioNeMo Framework, a separate
codebase; `scripts/fetch_bionemo.sh` clones it into `third_party/` for reference
and `docs/ARCHITECTURE.md` explains the split.

---

## Honesty rules this app enforces

1. **Every result is badged.** `LOCAL NIM`, `NVIDIA HOSTED`, `LOCAL RUNTIME`,
   `SIMULATED` or `LOCAL ANALYSIS`, plus the model key and latency.
2. **Provenance is recorded**, not decorative: model, backend, URL, parameters,
   warnings and timing go into SQLite and into the JSON manifest attached to each
   result. A `curl` command that reproduces the exact call is shown for every
   service call.
3. **A simulated number is never presented as a prediction.** Simulated runs
   carry a warning in the payload, a badge in the UI, an `_simulated` marker in
   the response, a REMARK line in exported PDBs, and a sentence in the summary.
4. **Descriptors are exact chemistry.** QED, SA, cLogP, TPSA, PAINS and friends
   are computed with RDKit locally — they do not depend on which backend
   generated the molecule, and they are not simulated.
5. **The router is deterministic.** Tool calls come from rules and entity
   extraction, not from a language model guessing. A language model may only plan
   as a documented last resort, and its proposal is validated against the tool
   schemas before anything runs.

See `docs/SIMULATION.md` for exactly what each fallback model computes.

---

## Documentation

| Document | Contents |
| --- | --- |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Module map, data flow, why the layers are split this way |
| [`docs/NIM_SETUP.md`](docs/NIM_SETUP.md) | GPU host setup, NGC keys, per-model containers, ports, troubleshooting |
| [`docs/SIMULATION.md`](docs/SIMULATION.md) | What the fallback models are and what they are not |
| [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) | Docker, compose, remote/behind-a-proxy, backups, security notes |

---

## Requirements

* Python 3.10+ (3.11 recommended)
* Any OS; the test suite runs on Linux, macOS and Windows
* ~300 MB of disk for the environment, plus ~0.5 MB of vendored viewer
* **Optional, for real models:** an NVIDIA GPU (Ampere or newer for most NIMs),
  a recent driver, Docker with `nvidia-container-toolkit`, and an NGC API key

The vendored 3D viewer (`bionemo_web/static/3Dmol-min.js`, 3Dmol.js 2.5.5, BSD-3)
means the 3D panes work with no internet access. Depictions of 2D structures are
drawn by the console's own SVG renderer, so no Cairo/PIL system libraries are
required.

---

## Development

```bash
make test        # pytest: 116 tests, offline, ~70 s
make lint        # syntax-check every module
make demo        # run in pure simulation mode
make nims        # start local NIM containers (needs GPU + NGC key)
```

The test suite runs the real Streamlit app through Streamlit's own test harness,
so it covers UI wiring end to end, not just libraries.

---

## Licence

Apache-2.0 (see `LICENSE`). Vendored third-party code keeps its own licence:
3Dmol.js is BSD-3 (`bionemo_web/static/3Dmol-LICENSE`). NVIDIA, BioNeMo and NIM
are trademarks of NVIDIA Corporation; this project is not affiliated with NVIDIA.
