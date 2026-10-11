# What the simulation backends are (and are not)

The console is designed so that a laptop with no GPU is still useful. When no NIM
answers, the relevant tool runs a **built-in model** instead. This document states
exactly what each one computes, so its output can be judged on its merits.

**The rule that governs all of this:** a simulated result is labelled everywhere —
in the payload (`warnings`, `meta.simulated`), in the UI badge (`SIMULATED`), in
the run's provenance manifest, in exported PDB files (`REMARK`), in the
screening report, and in the summary sentence. It is never presented as a
prediction.

| Tool | Fallback | What it actually computes |
| --- | --- | --- |
| Fold (`esmfold`, `alphafold2`, `openfold3`, `boltz2`) | Torsional Monte Carlo anneal | Ideal Engh & Huber backbone geometry placed along φ/ψ sampled by simulated annealing. Energy = steric clashes + Kyte–Doolittle burial + Flory radius-of-gyration restraint (2.2·N^0.38) + chain continuity + a torsion prior. Confidence is a per-residue proxy from burial and local packing, mapped to the pLDDT scale — a proxy, not a calibrated pLDDT. |
| Embeddings (`esm2`, `amplify`, `geneformer`) | Random projection of residue features | Ten physicochemical feature axes per residue, projected into 128 dimensions with a fixed, seeded matrix plus a positional component. Useful for plumbing, clustering demos and UI development; carries **no** evolutionary information. |
| Generation (`molmim`, `genmol`) | RDKit analogue enumeration | BRICS recombination, single-edit R-group enumeration and bioisostere swaps around the seed, filtered by a Tanimoto window and ranked by the requested objective (QED or plogP). Every molecule is real, valid chemistry — the *generator* is what is simulated, not the molecules. |
| Docking (`diffdock`) | Geometric docking engine | RDKit ETKDGv3 + MMFF94 ligand embedding; rigid-body poses in detected cavities; hill-climb refinement on a documented score (steric overlap, size-normalised contacts, hydrophobic fraction, H-bonds, burial, interaction breadth). Pose confidence is a logistic on that score. Geometry is real; the scoring function is not learned. |
| Design (`proteinmpnn`) | SS- and burial-conditioned sampler | Per-position probabilities from relative SASA, residue contacts and secondary structure, with a wild-type prior, sampled at temperature. Reports per-design score, recovery and every substitution. |
| Genomics (`evo2`, `codonfm`) | 6-mer Markov model | A k-mer model fitted on the prompt; generation with temperature and top-k; the product is analysed with the same DNA analyser used elsewhere (GC, ORFs, codon bias). |
| Complexes (`boltz2`) | Fold + rigid-body ligand placement | The receptor fold above, then the ligand placed in the best cavity, with an affinity-like score derived from the same interaction terms as docking. |

## What is *not* simulated

These are exact computations, run locally with RDKit and the console's own
geometry code, and they do not change with the backend:

* all molecular descriptors (MW, exact mass, cLogP, TPSA, HBD/HBA, Fsp3, QED, Ertl
  SA score, molar refractivity, solubility estimate)
* drug-likeness rules and structural alerts (Lipinski, Veber, PAINS, Brenk, NIH,
  reactive groups — with severity levels, so a Brenk note on aspirin's phenol
  ester is not reported as a PAINS frequent-hitter alert)
* Murcko scaffolds, BRICS fragments, fingerprints, Tanimoto similarities, MCS,
  clustering
* protein/DNA analysis (MW, pI, GRAVY, instability index, aliphatic index,
  extinction coefficients, hydropathy and flexibility profiles, disorder, SEG
  low-complexity masking, motif library, ORFs, codon usage, Tm)
* structure measurements: SASA (Shrake–Rupley), secondary structure
  (Kabsch–Sander), contact maps, Kabsch RMSD, contact-map overlap, cavities
* all geometry of exported files: an SDF or PDB you download contains real 3D
  coordinates

## How to stop using the fallback

1. Start one NIM (needs a GPU): `bash scripts/start_nims.sh esmfold`
2. In the console: **Administration → Backends**, set that model's endpoint to
   `http://localhost:8000`, press *Probe every configured endpoint*.
3. The badge becomes `LOCAL NIM`; `Production` results now come from the service.
   To forbid fallbacks entirely, choose **Local NIM only** as the default backend
   — the run then fails with a clear error rather than answering from a model.

## Honest limits worth knowing

* The folding model produces plausible topology and clean geometry. It does not
  know your protein. Treat a fold as a hypothesis to be tested against a NIM.
* The docking score is a shape-and-chemistry heuristic. It is monotone in pose
  quality, but its confidence values are not comparable to DiffDock's.
* Secondary-structure *assignment* is validated against reference geometry;
  secondary-structure *prediction* (the `predicted_ss` band shown next to it) is a
  propensity heuristic and is displayed separately for exactly that reason.
* The cavity druggability hint is a transparent blend of enclosure, depth, burial
  and lining-residue count. It is a triage aid, not a trained score.
