# Architecture

The console is a Streamlit front end over three layers that can each be tested
and replaced on their own:

```
app.py                     thin: page config, sidebar, routing
└── bionemo_web/
    ├── ui/                rendering only -- no science, no HTTP
    │   ├── theme.py       palette, CSS, badges, cards, metric tiles
    │   ├── viewer3d.py    3Dmol.js component (vendored JS, inlined)
    │   ├── plots.py       Plotly figures
    │   ├── cards.py       one renderer per result type
    │   └── chat.py        chat surface, composer, job progress
    ├── console/pages.py   the five non-chat pages
    ├── engine/            orchestration -- no UI imports
    │   ├── router.py      sentence -> tool plan (deterministic)
    │   ├── tools.py       tool registry + execution + local analysis
    │   ├── agent.py       one chat turn, end to end
    │   ├── jobs.py        background queue with cooperative cancel
    │   ├── store.py       SQLite: conversations, runs, artifacts, jobs
    │   ├── export.py      CSV/SDF/PDB/JSON/Markdown writers
    │   └── llm.py         providers + the offline analyst
    ├── nims/              everything that talks to models
    │   ├── catalog.py     17 model specs: paths, params, containers, provenance level
    │   ├── client.py      HTTP client, health probe, OpenAPI path discovery
    │   ├── service.py     backend policy + RunResult provenance
    │   ├── docking.py     geometric docking engine (simulated DiffDock)
    │   ├── sim.py         NIM-shaped simulators for every task
    │   └── local_runtime.py  host probe + torch/transformers path + docker commands
    └── bio/               pure science, no network, no Streamlit
        ├── constants.py   tables extracted from BioPython (with provenance)
        ├── sequence.py    protein/DNA analysis
        ├── chem.py        RDKit descriptors, rules, fingerprints, analogues
        ├── structure.py   PDB I/O, SASA, secondary structure, cavities, folding
        ├── depict.py      2D structure SVG renderer (no Cairo)
        └── embed.py       embedding maths (PCA, k-means, neighbours)
```

## Why the layers are split this way

**`ui/` never computes and `bio/` never renders.** A renderer bug cannot corrupt a
number, and a physics change cannot break a layout. It also means the whole
science layer is testable without Streamlit, and the whole UI is testable without
a GPU — which is exactly how `tests/` is organised.

**`engine/router.py` is deterministic on purpose.** The same sentence always
produces the same plan. Model calls chosen by an LLM are attractive in a demo and
miserable in practice: they are unreproducible, hard to test and impossible to
explain when they go wrong. Here the planner is a keyword-and-entity scorer you
can read, and a language model is consulted only when nothing matches — its
proposal is then validated against the tool schemas before execution.

**`nims/service.py` is the single place that decides a backend.** Tools call
`service.run(model, payload)` and receive a `RunResult` carrying the backend, URL,
status, latency, attempt chain and warnings. No tool knows how to fall back, so
no tool can fall back inconsistently.

**Result objects, not strings.** A tool returns a `ToolOutcome` with a renderer
name, a payload and provenance. The chat, the workbench pages and the exporters
all consume the same object, which is why a chat result and a workbench result can
produce identical files.

## One chat turn

```
user message
  └─ store.add_message()                                   (persist)
  └─ agent.session_context()                               (last sequence / structure / SMILES)
  └─ router.plan()                                         (deterministic; LLM only as last resort)
        │  needs missing input? -> clarification turn, no model call
        └─ heavy tool? -> jobs.submit() and reply immediately (fragment polls progress)
  └─ tools.execute() per step
        └─ nims/service.run() -> local NIM | hosted NIM | framework | simulate
              └─ RunResult (backend, url, params, warnings, latency)
        └─ local analysis enrichment (descriptors, pockets, contacts)
  └─ store.record_run(provenance)                          (audit trail)
  └─ narration: LLM if connected, else OfflineAnalyst from the result objects
  └─ store.add_message(payload={plan, outcomes, run_ids, provenance})
```

Reloading the page re-renders from that stored payload — including the backend
badge, which is why provenance is snapshotted into `ToolOutcome.to_dict()` rather
than looked up again.

## Background jobs

Folding 300 residues, docking 40 poses or screening 20 molecules takes seconds to
minutes; Streamlit reruns the whole script on every interaction, so that work
cannot live in a callback. `engine/jobs.py` is a process-wide thread pool with
jobs persisted through `queued → running → done|failed|cancelled`. Cancellation is
cooperative: long routines are written as small steps that call
`ctx.check_cancelled()` between them, which is also what makes progress reporting
possible. The chat's job panel is a self-polling Streamlit *fragment*, so progress
moves without re-running the app or losing the conversation.

## Storage

`engine/store.py` is SQLite in `.runtime/console.db` (git-ignored):

| table | holds |
| --- | --- |
| `conversations` | titles, pin state, tags |
| `messages` | role, content, and the full result payload as JSON |
| `runs` | every model invocation with its provenance manifest |
| `call_log` | raw HTTP outcomes, for the latency table |
| `artifacts` | exported files, so History can serve them again |
| `jobs` | background job lifecycle |
| `settings` | endpoints, backend preference, LLM config |

Heavy objects (structures, poses) live in memory for the session; what is
persisted is their summary plus provenance. That keeps the database small while
keeping the audit trail complete.

## The science, briefly

* **Secondary structure** is Kabsch–Sander: hydrogen-bond energy from a
  DSSP-style electrostatic model, α/3-10/β-bridge/turn passes, island smoothing.
  Verified against ideal geometry (an ideal α-helix assigns 30/30 `H`; an
  isolated extended strand assigns no `E`, because a β-bridge needs a partner).
* **SASA** is Shrake–Rupley with 92 points and a 1.4 Å probe.
* **Cavities** are found on a lattice: voxelise, mark solvent-accessible space by
  geodesic depth from the boundary, take enclosed regions, then cluster and rank
  by volume, enclosure, burial depth and lining residues. Depth is capped (25 Å)
  because the lattice is finite, and that cap is printed in the UI.
* **Folding** (`fold_sequence`) is a torsional Monte Carlo anneal on an
  HP-style potential with a Flory radius-of-gyration restraint and ideal Engh &
  Huber backbone geometry. It is a physical model, not a trained predictor.
* **Docking** samples rigid-body transforms in detected cavities and refines them
  on a fully documented score: steric overlap, size-normalised contact density,
  hydrophobic contact fraction, hydrogen bonds, burial — scaled by an
  interaction-breadth factor so a fragment cannot outrank a real hit. Confidence is
  a logistic on that score, attenuated by remaining clashes.

Every constant in these routines is a module-level name with a comment, so a
disagreement about a number is a code review, not an argument with a black box.
