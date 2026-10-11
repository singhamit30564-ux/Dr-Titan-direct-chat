"""The model registry: every BioNeMo NIM the console knows how to talk to.

Each entry records the documented HTTP route, the request parameters the
microservice accepts, and how the console should interpret the response.  Routes
follow NVIDIA's published NIM API documentation; where a route can differ
between container versions the entry is flagged as ``configurable`` and the
console's endpoint auto-discovery (``/openapi.json``) or the Settings page can
override it at runtime.

Nothing here is guessed silently: :attr:`NimSpec.path_confidence` says how much
trust to place in a default path, and the UI surfaces that flag next to the
field.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

HOSTED_BASE = "https://health.api.nvidia.com/v1"

#: Task families the UI knows how to render.
TASK_KINDS = [
    "structure_prediction",
    "sequence_design",
    "embeddings",
    "molecule_generation",
    "docking",
    "genomics",
    "segmentation",
]


@dataclass
class ParamSpec:
    """One request parameter, used to build the form and the payload."""

    name: str
    label: str
    type: str = "str"  # str|int|float|bool|select|sequence|smiles|pdb|fasta|text
    default: object = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    options: List[str] = field(default_factory=list)
    help: str = ""
    advanced: bool = False
    required: bool = False


@dataclass
class NimSpec:
    """A single BioNeMo NIM model / microservice."""

    key: str
    name: str
    vendor: str
    task: str
    kind: str  # how results are parsed: fold|embeddings|molecules|dock|design|genomics
    description: str
    hosted_path: str = ""
    local_path: str = ""
    local_port: int = 8000
    docs_url: str = "https://docs.nvidia.com/nim/bionemo/latest/index.html"
    container: str = ""
    params: List[ParamSpec] = field(default_factory=list)
    inputs: List[str] = field(default_factory=list)
    output_keys: List[str] = field(default_factory=list)
    path_confidence: str = "documented"  # documented | configurable | verified
    local_only: bool = False
    experimental: bool = False
    notes: str = ""
    tags: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

_MODELS: List[NimSpec] = [
    # ---------------------------------------------------------------- folding
    NimSpec(
        key="esmfold",
        name="ESMFold",
        vendor="Meta / NVIDIA",
        task="structure_prediction",
        kind="fold",
        description=(
            "Single-sequence protein structure prediction. Fast, no MSA required, "
            "per-residue pLDDT confidence, suitable for screening many designs."
        ),
        hosted_path="/biology/mit/esmfold",
        local_path="/biology/mit/esmfold/predict-structure-from-sequence",
        container="nvcr.io/nim/meta/esmfold",
        inputs=["sequence"],
        output_keys=["pdbs", "plddt", "ptm"],
        params=[
            ParamSpec("sequence", "Amino-acid sequence", "sequence", required=True,
                      help="Single-letter code. Up to ~1024 residues for the standard NIM profile."),
            ParamSpec("residue_indices", "Residue indices to return", "str", "",
                      advanced=True, help="Optional comma-separated subset."),
        ],
        tags=["fast", "no-MSA", "screening"],
    ),
    NimSpec(
        key="alphafold2",
        name="AlphaFold2",
        vendor="DeepMind / NVIDIA",
        task="structure_prediction",
        kind="fold",
        description=(
            "MSA-based structure prediction with PAE and pLDDT. The reference for "
            "single-chain accuracy when a sequence has good homolog coverage."
        ),
        hosted_path="/biology/mit/alphafold2/predict-structure-from-sequence",
        local_path="/biology/mit/alphafold2/predict-structure-from-sequence",
        container="nvcr.io/nim/deepmind/alphafold2",
        inputs=["sequence"],
        output_keys=["pdbs", "plddt", "ptm", "pae"],
        params=[
            ParamSpec("sequence", "Amino-acid sequence", "sequence", required=True),
            ParamSpec("databases", "MSA databases", "str", "uniref90,small_bfd",
                      advanced=True, help="Comma-separated: uniref90, mgnify, small_bfd, ..."),
            ParamSpec("relax_prediction", "AMBER relax", "bool", True, advanced=True,
                      help="Energy-minimise side chains (slower, better stereochemistry)."),
            ParamSpec("algorithm", "Algorithm", "select", "mmseqs2",
                      options=["mmseqs2", "jackhmmer"], advanced=True),
        ],
        tags=["MSA", "accurate", "pAE"],
    ),
    NimSpec(
        key="openfold3",
        name="OpenFold3",
        vendor="NVIDIA / AlQuB",
        task="structure_prediction",
        kind="fold",
        description=(
            "Open-source AlphaFold3-style model: joint structure, complex and "
            "ligand-aware prediction with confidence heads."
        ),
        hosted_path="/biology/nvidia/openfold3/predict-structure-from-sequence",
        local_path="/biology/nvidia/openfold3/predict-structure-from-sequence",
        container="nvcr.io/nim/nvidia/openfold3",
        inputs=["sequence"],
        output_keys=["pdbs", "plddt", "ptm"],
        path_confidence="configurable",
        params=[
            ParamSpec("sequence", "Amino-acid sequence", "sequence", required=True),
            ParamSpec("ligands", "Ligand SMILES (complex prediction)", "str", "",
                      advanced=True, help="Optional; enables protein-ligand complex prediction."),
            ParamSpec("num_loops", "Recycling loops", "int", 3, minimum=1, maximum=10, advanced=True),
        ],
        tags=["AF3", "complexes"],
    ),
    NimSpec(
        key="boltz2",
        name="Boltz-2",
        vendor="MIT / NVIDIA",
        task="structure_prediction",
        kind="fold",
        description=(
            "Structure + binding-affinity prediction in one pass. Useful for ranking "
            "candidate ligands against a target without running docking first."
        ),
        hosted_path="/biology/mit/boltz2/predict-structure-and-affinity",
        local_path="/biology/mit/boltz2/predict-structure-and-affinity",
        container="nvcr.io/nim/mit/boltz2",
        inputs=["sequence"],
        output_keys=["structure", "affinity_pred_value", "affinity_probability_binary"],
        path_confidence="configurable",
        params=[
            ParamSpec("sequence", "Protein sequence", "sequence", required=True),
            ParamSpec("ligand_smiles", "Ligand SMILES", "smiles", "",
                      help="Leave empty for apo structure prediction."),
            ParamSpec("diffusion_samples", "Diffusion samples", "int", 1, minimum=1, maximum=10, advanced=True),
        ],
        tags=["affinity", "co-folding"],
    ),
    # ------------------------------------------------------------- embeddings
    NimSpec(
        key="esm2",
        name="ESM-2 (650M)",
        vendor="Meta / NVIDIA",
        task="embeddings",
        kind="embeddings",
        description=(
            "Protein language model embeddings. Per-residue and mean-pooled vectors "
            "for clustering, novelty screening, variant effect proxies and ML featurisation."
        ),
        hosted_path="/biology/meta/esm2-650m/embeddings",
        local_path="/embeddings",
        container="nvcr.io/nim/meta/esm2-650m",
        inputs=["sequence"],
        output_keys=["embeddings", "mean_embeddings"],
        path_confidence="configurable",
        params=[
            ParamSpec("sequence", "Amino-acid sequence", "sequence", required=True),
            ParamSpec("model", "Model size", "select", "esm2-650m",
                      options=["esm2-8m", "esm2-650m", "esm2-3b", "esm2-15b"]),
        ],
        tags=["PLM", "featurisation"],
    ),
    NimSpec(
        key="amplify",
        name="AMPLIFY (350M)",
        vendor="Amgen / NVIDIA",
        task="embeddings",
        kind="embeddings",
        description="Protein language model trained on a curated corpus; strong for fitness-style downstream tasks.",
        local_only=True,
        local_path="/embeddings",
        container="BioNeMo Framework / bionemo-recipes",
        inputs=["sequence"],
        output_keys=["embeddings"],
        path_confidence="configurable",
        params=[ParamSpec("sequence", "Amino-acid sequence", "sequence", required=True)],
        tags=["PLM", "BioNeMo Framework"],
    ),
    NimSpec(
        key="geneformer",
        name="Geneformer",
        vendor="Broad / NVIDIA",
        task="embeddings",
        kind="embeddings",
        description="Single-cell transcriptomics foundation model (gene-rank embeddings).",
        local_only=True,
        local_path="/embeddings",
        container="BioNeMo Framework / bionemo-recipes",
        inputs=["sequence"],
        output_keys=["embeddings"],
        path_confidence="configurable",
        params=[ParamSpec("sequence", "Gene token sequence / ranked gene list", "str", "")],
        tags=["single-cell", "BioNeMo Framework"],
        notes="Geneformer consumes ranked gene token lists, not amino-acid sequences.",
    ),
    # -------------------------------------------------------------- design
    NimSpec(
        key="proteinmpnn",
        name="ProteinMPNN",
        vendor="Baker Lab / NVIDIA",
        task="sequence_design",
        kind="design",
        description=(
            "Inverse folding: given a protein backbone, propose sequences that fold "
            "into it. The workhorse for stabilising or solubilising a design."
        ),
        hosted_path="/biology/ipd/proteinmpnn/predict",
        local_path="/biology/ipd/proteinmpnn/generate",
        container="nvcr.io/nim/ipd/proteinmpnn",
        inputs=["pdb"],
        output_keys=["sequences", "scores"],
        path_confidence="configurable",
        params=[
            ParamSpec("pdb", "Backbone structure (PDB)", "pdb", required=True,
                      help="Fixed residues can be specified with /chain/residue syntax."),
            ParamSpec("num_sequences", "Number of sequences", "int", 8, minimum=1, maximum=1000),
            ParamSpec("sampling_temp", "Sampling temperature", "float", 0.1, minimum=0.0001, maximum=1.0),
            ParamSpec("fixed_positions", "Fixed positions", "str", "", advanced=True,
                      help="Comma-separated, e.g. 12,14,30."),
            ParamSpec("model_name", "Model", "select", "v_48_020",
                      options=["v_48_002", "v_48_010", "v_48_020", "v_48_030"], advanced=True),
        ],
        tags=["inverse folding", "stability"],
    ),
    NimSpec(
        key="rfdiffusion",
        name="RFDiffusion",
        vendor="Baker Lab / NVIDIA",
        task="sequence_design",
        kind="fold",
        description=(
            "Backbone diffusion for de-novo design: binder scaffolds, motif scaffolding "
            "and symmetric oligomers. Output is a backbone to be re-designed with ProteinMPNN."
        ),
        hosted_path="/biology/ipd/rfdiffusion/generate",
        local_path="/biology/ipd/rfdiffusion/generate",
        container="nvcr.io/nim/ipd/rfdiffusion",
        inputs=["sequence", "pdb"],
        output_keys=["pdbs"],
        path_confidence="configurable",
        params=[
            ParamSpec("task", "Design task", "select", "unconditional",
                      options=["unconditional", "binder_design", "motif_scaffolding"]),
            ParamSpec("sequence", "Target sequence (for binder design)", "sequence", ""),
            ParamSpec("pdb", "Motif / target PDB", "pdb", ""),
            ParamSpec("num_designs", "Number of designs", "int", 4, minimum=1, maximum=50),
            ParamSpec("noise_scale", "Noise scale", "float", 1.0, minimum=0.1, maximum=2.0, advanced=True),
            ParamSpec("contigmap", "Contig map", "str", "", advanced=True,
                      help="e.g. 50-80 for a new chain of 50-80 residues."),
        ],
        tags=["de novo", "binder"],
        notes="RFDiffusion NIMs typically expose a queue-based API; the console polls the job status route.",
    ),
    NimSpec(
        key="esm2_generate",
        name="ESM-2 generation",
        vendor="Meta / NVIDIA",
        task="sequence_design",
        kind="design",
        description="Autoregressive sequence generation from the ESM-2 family, optionally constrained by a prompt.",
        hosted_path="/biology/meta/esm2-650m/generate",
        local_path="/generate",
        container="nvcr.io/nim/meta/esm2-650m",
        inputs=["sequence"],
        output_keys=["sequence", "sequences"],
        path_confidence="configurable",
        params=[
            ParamSpec("sequence", "Prompt sequence", "sequence", ""),
            ParamSpec("num_tokens", "Tokens to generate", "int", 100, minimum=1, maximum=1000),
            ParamSpec("temperature", "Temperature", "float", 1.0, minimum=0.01, maximum=5.0),
            ParamSpec("top_p", "Top-p", "float", 0.95, minimum=0.0, maximum=1.0, advanced=True),
        ],
        tags=["generation"],
    ),
    # ------------------------------------------------------------- molecules
    NimSpec(
        key="molmim",
        name="MolMIM",
        vendor="NVIDIA",
        task="molecule_generation",
        kind="molecules",
        description=(
            "Controlled molecule generation around a seed: latent-space sampling with "
            "CMA-ES optimisation of a chosen property (QED, pLogP) under a similarity floor."
        ),
        hosted_path="/biology/nvidia/molmim/generate",
        local_path="/generate",
        container="nvcr.io/nim/nvidia/molmim",
        inputs=["smiles"],
        output_keys=["molecules", "generated"],
        params=[
            ParamSpec("smi", "Seed SMILES", "smiles", required=True),
            ParamSpec("algorithm", "Algorithm", "select", "CMA-ES", options=["CMA-ES", "none"]),
            ParamSpec("num_molecules", "Molecules", "int", 12, minimum=1, maximum=512),
            ParamSpec("property_name", "Objective", "select", "QED", options=["QED", "plogP"]),
            ParamSpec("minimize", "Minimise objective", "bool", False),
            ParamSpec("min_similarity", "Minimum similarity", "float", 0.3, minimum=0.0, maximum=1.0),
            ParamSpec("particles", "Particles", "int", 30, minimum=2, maximum=200, advanced=True),
            ParamSpec("iterations", "Iterations", "int", 10, minimum=1, maximum=100, advanced=True),
            ParamSpec("scaled_radius", "Scaled radius (algorithm=none)", "float", 1.0,
                      minimum=0.0, maximum=5.0, advanced=True),
        ],
        tags=["lead optimisation", "latent space"],
        notes="The MolMIM NIM also exposes /embedding, /hidden, /decode and /sampling for latent-space work.",
    ),
    NimSpec(
        key="genmol",
        name="GenMol",
        vendor="NVIDIA",
        task="molecule_generation",
        kind="molecules",
        description=(
            "Fragment-based discrete diffusion: fill in masked positions of a SMILES "
            "string (use [*{15-15}] style masks) and score the products in one pass."
        ),
        hosted_path="/biology/nvidia/genmol/generate",
        local_path="/generate",
        container="nvcr.io/nim/nvidia/genmol",
        inputs=["smiles"],
        output_keys=["molecules"],
        params=[
            ParamSpec("smiles", "Masked SMILES", "smiles", "[C@H]1O[C@@H](CO)[C@H](O)[C@@H]1O.[*{15-15}]",
                      help="Masked positions are written as [*{n-m}]."),
            ParamSpec("num_molecules", "Molecules", "int", 10, minimum=1, maximum=500),
            ParamSpec("temperature", "Temperature", "float", 2.0, minimum=0.1, maximum=5.0),
            ParamSpec("noise", "Noise", "float", 1.0, minimum=0.0, maximum=3.0),
            ParamSpec("step_size", "Step size", "int", 1, minimum=1, maximum=20, advanced=True),
            ParamSpec("scoring", "Scoring function", "select", "QED", options=["QED", "SA", "logP"]),
        ],
        tags=["fragment", "diffusion"],
    ),
    # -------------------------------------------------------------- docking
    NimSpec(
        key="diffdock",
        name="DiffDock",
        vendor="MIT / NVIDIA",
        task="docking",
        kind="dock",
        description=(
            "Diffusion-based blind docking: predicts binding poses of a ligand to a "
            "protein without a specified pocket, with a confidence per pose."
        ),
        hosted_path="/biology/mit/diffdock",
        local_path="/molecular-docking/diffdock/generate",
        container="nvcr.io/nim/mit/diffdock",
        inputs=["pdb", "smiles"],
        output_keys=["ligand_positions", "position_confidence"],
        params=[
            ParamSpec("protein", "Protein (PDB text or FASTA)", "pdb", required=True),
            ParamSpec("ligand", "Ligand (SMILES or SDF)", "smiles", required=True),
            ParamSpec("ligand_file_type", "Ligand format", "select", "smiles", options=["smiles", "sdf"],
                      advanced=True),
            ParamSpec("num_poses", "Number of poses", "int", 10, minimum=1, maximum=40),
            ParamSpec("time_divisions", "Time divisions", "int", 20, minimum=1, maximum=100, advanced=True),
            ParamSpec("steps", "Steps", "int", 18, minimum=1, maximum=100, advanced=True),
            ParamSpec("save_trajectory", "Save trajectory", "bool", False, advanced=True),
        ],
        tags=["blind docking", "poses"],
        notes="Poses are returned as flat xyz coordinates for every ligand atom, in input atom order.",
    ),
    # ------------------------------------------------------------- genomics
    NimSpec(
        key="evo2",
        name="Evo 2",
        vendor="Arc Institute / NVIDIA",
        task="genomics",
        kind="genomics",
        description=(
            "Genomic foundation model over nucleotide sequences: generation with "
            "per-token probabilities, variant effect scoring and embedding extraction "
            "across very long contexts."
        ),
        hosted_path="/biology/arc/evo2-40b/generate",
        local_path="/generate",
        container="nvcr.io/nim/arc/evo2-40b",
        inputs=["sequence"],
        output_keys=["sequence", "probs"],
        path_confidence="configurable",
        params=[
            ParamSpec("sequence", "DNA prompt sequence", "fasta", ""),
            ParamSpec("num_tokens", "Tokens to generate", "int", 100, minimum=1, maximum=5000),
            ParamSpec("temperature", "Temperature", "float", 0.7, minimum=0.0, maximum=2.0),
            ParamSpec("top_k", "Top-k", "int", 4, minimum=0, maximum=100, advanced=True),
            ParamSpec("enable_sampled_probs", "Return sampled probabilities", "bool", True, advanced=True),
        ],
        tags=["DNA", "variant effects"],
        notes="Evo 2 forward endpoints return per-position log-probabilities used for variant-effect and exon scoring.",
    ),
    NimSpec(
        key="codonfm",
        name="CodonFM",
        vendor="NVIDIA Research",
        task="genomics",
        kind="genomics",
        description="Codon-level language model for mRNA design: codon usage, stability and expression-oriented scoring.",
        local_only=True,
        local_path="/generate",
        container="BioNeMo Framework / bionemo-recipes",
        inputs=["sequence"],
        output_keys=["sequence"],
        path_confidence="configurable",
        params=[ParamSpec("sequence", "Coding DNA sequence", "fasta", "")],
        tags=["mRNA", "codon optimisation", "BioNeMo Framework"],
    ),
    # --------------------------------------------------------- segmentation
    NimSpec(
        key="vista3d",
        name="VISTA-3D",
        vendor="NVIDIA",
        task="segmentation",
        kind="segmentation",
        description="Interactive 3D medical-image segmentation (CT). Ships in the BioNeMo NIM family for imaging pipelines.",
        hosted_path="/vista3d/inference",
        local_path="/inference",
        container="nvcr.io/nim/nvidia/vista-3d",
        inputs=[],
        output_keys=["mask"],
        path_confidence="configurable",
        params=[
            ParamSpec("image", "Volume reference (NIfTI path on the NIM host)", "str", ""),
            ParamSpec("points", "Point prompts", "str", "", help="[[x,y,z,label], ...]"),
        ],
        tags=["imaging", "CT"],
        notes="Needs a volume on the NIM host; the console links out rather than shuttling large NIfTI files.",
    ),
    NimSpec(
        key="msa_search",
        name="MSA Search",
        vendor="NVIDIA",
        task="structure_prediction",
        kind="fold",
        description="ColabFold-style MSA search service used to feed folding pipelines.",
        hosted_path="/biology/nvidia/msa-search",
        local_path="/msa",
        container="nvcr.io/nim/nvidia/msa-search",
        inputs=["sequence"],
        output_keys=["msa", "a3m"],
        path_confidence="configurable",
        params=[ParamSpec("sequence", "Amino-acid sequence", "sequence", required=True)],
        tags=["MSA", "infrastructure"],
        experimental=True,
    ),
]

BY_KEY: Dict[str, NimSpec] = {m.key: m for m in _MODELS}

#: Alias -> key, so free text in the chat ("fold this", "dock that") resolves.
ALIASES: Dict[str, str] = {
    "esm-fold": "esmfold",
    "esm_fold": "esmfold",
    "af2": "alphafold2",
    "alphafold": "alphafold2",
    "alpha-fold": "alphafold2",
    "openfold": "openfold3",
    "af3": "openfold3",
    "boltz": "boltz2",
    "mol-mim": "molmim",
    "molmim generate": "molmim",
    "gen-mol": "genmol",
    "diff-dock": "diffdock",
    "mpnn": "proteinmpnn",
    "rf-diffusion": "rfdiffusion",
    "rf diffusion": "rfdiffusion",
    "evo-2": "evo2",
    "evo": "evo2",
    "codon": "codonfm",
    "vista": "vista3d",
    "vista-3d": "vista3d",
    "msa": "msa_search",
}

VENDOR_COLORS = {
    "NVIDIA": "#76b900",
    "Meta / NVIDIA": "#4da3ff",
    "DeepMind / NVIDIA": "#7c5cff",
    "NVIDIA / AlQuB": "#00c2a8",
    "MIT / NVIDIA": "#ff8a3d",
    "Baker Lab / NVIDIA": "#ff5c8a",
    "Amgen / NVIDIA": "#c77dff",
    "Broad / NVIDIA": "#3ddc97",
    "Arc Institute / NVIDIA": "#ffd166",
    "NVIDIA Research": "#76b900",
}


def all_models() -> List[NimSpec]:
    return list(_MODELS)


def get(key: str) -> Optional[NimSpec]:
    if not key:
        return None
    if key in BY_KEY:
        return BY_KEY[key]
    return BY_KEY.get(ALIASES.get(key.lower().strip(), ""))


def by_task(task: str) -> List[NimSpec]:
    return [m for m in _MODELS if m.task == task]


def resolve(query: str) -> Optional[NimSpec]:
    """Fuzzy-resolve a free-text model mention to a spec."""
    q = (query or "").lower().strip()
    if not q:
        return None
    if q in BY_KEY:
        return BY_KEY[q]
    if q in ALIASES:
        return BY_KEY[ALIASES[q]]
    for m in _MODELS:
        if m.name.lower() == q or m.key == q:
            return m
    for m in _MODELS:
        if q in m.key or q in m.name.lower():
            return m
    for alias, key in ALIASES.items():
        if alias in q:
            return BY_KEY[key]
    return None


def param_defaults(spec: NimSpec) -> Dict[str, object]:
    return {p.name: p.default for p in spec.params if p.default is not None}


def summary_table() -> List[Dict]:
    """Compact registry view for the UI."""
    return [
        {
            "key": m.key,
            "name": m.name,
            "vendor": m.vendor,
            "task": m.task.replace("_", " "),
            "hosted": bool(m.hosted_path) and not m.local_only,
            "local_path": m.local_path,
            "container": m.container,
            "confidence": m.path_confidence,
            "tags": m.tags,
            "description": m.description,
        }
        for m in _MODELS
    ]
