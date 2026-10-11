"""What is actually installed on this machine, and what to type to fix it.

The BioNeMo Framework and the BioNeMo NIM containers both need an NVIDIA GPU.
This module inspects the host honestly (GPU, driver, CUDA, PyTorch, container
toolkit, fetched framework checkout, NGC access) and turns the answer into the
exact commands a user should run, so the console can point at the gap instead of
pretending it does not exist.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

FRAMEWORK_DIR = Path(__file__).resolve().parents[2] / "third_party" / "bionemo-framework"
FRAMEWORK_REPO = "https://github.com/NVIDIA/bionemo-framework"

#: The NIM containers used by the console, in the order the docs introduce them.
NIM_CONTAINERS: List[Dict[str, str]] = [
    {"key": "esmfold", "image": "nvcr.io/nim/meta/esmfold:latest", "port": "8000",
     "note": "Fast single-sequence folding. Smallest container to start with."},
    {"key": "alphafold2", "image": "nvcr.io/nim/deepmind/alphafold2:latest", "port": "8000",
     "note": "Needs MSA databases mounted; ~1 TB of reference data for full accuracy."},
    {"key": "molmim", "image": "nvcr.io/nim/nvidia/molmim:latest", "port": "8000",
     "note": "Small molecule generation around a seed."},
    {"key": "diffdock", "image": "nvcr.io/nim/mit/diffdock:latest", "port": "8000",
     "note": "Blind docking with per-pose confidence."},
    {"key": "proteinmpnn", "image": "nvcr.io/nim/ipd/proteinmpnn:latest", "port": "8000",
     "note": "Inverse folding from a backbone."},
    {"key": "genmol", "image": "nvcr.io/nim/nvidia/genmol:latest", "port": "8000",
     "note": "Fragment-based molecule generation."},
    {"key": "evo2", "image": "nvcr.io/nim/arc/evo2-40b:latest", "port": "8000",
     "note": "Genomic foundation model (large download)."},
]


def _which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


def _run(cmd: List[str], timeout: int = 6) -> str:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (proc.stdout or "") + (proc.stderr or "")
    except Exception:
        return ""


def detect_gpus() -> Dict[str, Any]:
    """NVIDIA GPUs visible to this host, with driver and CUDA runtime info."""
    info: Dict[str, Any] = {"count": 0, "names": [], "driver": "", "cuda": "", "source": ""}
    if _which("nvidia-smi"):
        out = _run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap",
             "--format=csv,noheader"]
        )
        rows = [r.strip() for r in out.splitlines() if r.strip() and "," in r]
        if rows:
            info["count"] = len(rows)
            for row in rows:
                parts = [p.strip() for p in row.split(",")]
                info["names"].append(parts[0] if parts else "GPU")
                if len(parts) > 1 and not info["driver"]:
                    info["driver"] = parts[1]
                    info["cuda"] = _run(["nvidia-smi"]).split("CUDA Version:")[-1].split()[0] if "CUDA Version:" in _run(["nvidia-smi"]) else ""
            info["source"] = "nvidia-smi"
            return info
    # CUDA toolkit without the driver query tool
    nvcc = _run(["nvcc", "--version"])
    if "release" in nvcc:
        info["cuda"] = nvcc.split("release")[-1].strip().split(",")[0]
        info["source"] = "nvcc"
    return info


def _module_version(name: str) -> Optional[str]:
    try:
        spec = importlib.util.find_spec(name)
    except Exception:
        return None
    if spec is None:
        return None
    try:
        module = __import__(name)
        return str(getattr(module, "__version__", "installed"))
    except Exception:
        return "installed"


def detect_container_toolkit() -> Dict[str, Any]:
    return {
        "docker": _which("docker"),
        "podman": _which("podman"),
        "nvidia_runtime": bool(_which("nvidia-container-runtime") or _which("nvidia-container-toolkit")),
        "ngc_cli": _which("ngc"),
    }


def detect_framework_checkout(path: Optional[Path] = None) -> Dict[str, Any]:
    root = Path(path or FRAMEWORK_DIR)
    info: Dict[str, Any] = {"path": str(root), "present": root.is_dir(), "version": "", "models": [], "recipes": []}
    if not info["present"]:
        return info
    version_file = root / "VERSION"
    if version_file.exists():
        info["version"] = version_file.read_text().strip()
    models_dir = root / "models"
    if models_dir.is_dir():
        info["models"] = sorted(p.name for p in models_dir.iterdir() if p.is_dir())
    recipes_dir = root / "recipes"
    if recipes_dir.is_dir():
        info["recipes"] = sorted(p.name for p in recipes_dir.iterdir() if p.is_dir())[:40]
    return info


def detect_environment() -> Dict[str, Any]:
    """One dict describing everything the console can use to run real models."""
    gpus = detect_gpus()
    containers = detect_container_toolkit()
    framework = detect_framework_checkout()
    torch_version = _module_version("torch")
    transformers_version = _module_version("transformers")
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")

    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "gpus": gpus,
        "gpu_available": gpus["count"] > 0,
        "containers": containers,
        "framework_checkout": framework["present"],
        "framework": framework,
        "torch": torch_version,
        "transformers": transformers_version,
        "cuda_visible_devices": cuda_visible,
        "ngc_key_present": bool(os.environ.get("NGC_API_KEY") or os.environ.get("NVIDIA_API_KEY")),
        "nvidia_api_key_present": bool(
            os.environ.get("NVIDIA_API_KEY") or os.environ.get("NGC_API_KEY")
        ),
    }
    env["can_run_nim_containers"] = bool(
        gpus["count"] > 0 and (containers["docker"] or containers["podman"]) and containers["nvidia_runtime"]
    )
    env["can_run_framework"] = bool(torch_version and gpus["count"] > 0)
    env["advice"] = _advice(env)
    return env


def _advice(env: Dict[str, Any]) -> List[str]:
    """Short, ordered, actionable statements about what is missing."""
    tips: List[str] = []
    if not env["gpu_available"]:
        tips.append(
            "No NVIDIA GPU detected. BioNeMo NIM containers and BioNeMo Framework recipes "
            "both require one; on this machine the console will use its built-in simulation "
            "backends, which is exactly what they are for."
        )
    if env["gpu_available"] and not (env["containers"]["docker"] or env["containers"]["podman"]):
        tips.append(
            "A GPU is present but no container runtime was found. Install Docker and "
            "nvidia-container-toolkit to run NIM containers."
        )
    if env["gpu_available"] and not env["containers"]["nvidia_runtime"]:
        tips.append(
            "nvidia-container-toolkit was not found, so containers will not see the GPU. "
            "Install it and restart the Docker daemon."
        )
    if not env["ngc_key_present"]:
        tips.append(
            "NGC_API_KEY is not set. Pulling NIM containers from nvcr.io and using the "
            "hosted NIM endpoints both need one (free at build.nvidia.com)."
        )
    if not env["framework_checkout"]:
        tips.append(
            "The BioNeMo Framework source is not present. Run scripts/fetch_bionemo.sh to "
            "clone it into third_party/bionemo-framework (recipes, models and docs)."
        )
    if not env["torch"]:
        tips.append(
            "PyTorch is not installed, so the in-process runtime backend is unavailable. "
            "pip install torch transformers enables local ESM-2 embeddings."
        )
    if not tips:
        tips.append("Environment looks ready: GPU, container runtime, NGC key and framework all present.")
    return tips


def nim_run_commands() -> List[Dict[str, str]]:
    """Copy-paste commands to start each NIM locally and register it here."""
    commands = []
    for entry in NIM_CONTAINERS:
        commands.append(
            {
                "key": entry["key"],
                "title": f"{entry['key']} - port {entry['port']}",
                "note": entry["note"],
                "command": (
                    f"docker run --rm -it --gpus all \\\n"
                    f"  -e NGC_API_KEY=$NGC_API_KEY \\\n"
                    f"  -p {entry['port']}:8000 \\\n"
                    f"  {entry['image']}"
                ),
                "verify": f"curl -s http://localhost:{entry['port']}/v1/health/ready",
                "register": f"http://localhost:{entry['port']}",
            }
        )
    return commands


def registry_commands() -> List[Dict[str, str]]:
    return [
        {
            "title": "Authenticate to NGC",
            "command": "docker login nvcr.io -u '$oauthtoken' -p $NGC_API_KEY",
            "note": "Username must be exactly $oauthtoken for NGC.",
        },
        {
            "title": "Fetch the BioNeMo source (this repo's helper)",
            "command": "bash scripts/fetch_bionemo.sh",
            "note": "Clones NVIDIA/bionemo-framework into third_party/bionemo-framework.",
        },
        {
            "title": "List NIM model profiles before pulling",
            "command": "ngc registry model list nvidia/nim/*biology*",
            "note": "Some images are multi-GB; check the profile and VRAM needs first.",
        },
    ]


def esm2_embeddings_torch(sequence: str, model_id: str = "facebook/esm2_t12_35M_UR50D") -> Optional[Dict[str, Any]]:
    """Real ESM-2 embeddings via transformers, if the model is available locally.

    Returns ``None`` (never raises) when transformers or the weights are missing,
    so the caller can fall back to the simulation backend with a clear message.
    """
    import time

    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except Exception:
        return None
    sequence = (sequence or "").strip().upper()
    if not sequence:
        return None
    started = time.time()
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        model = AutoModel.from_pretrained(model_id, local_files_only=True)
    except Exception:
        return None
    device = "cuda" if (hasattr(torch, "cuda") and torch.cuda.is_available()) else "cpu"
    model = model.to(device).eval()
    with torch.no_grad():
        batch = tokenizer(sequence, return_tensors="pt").to(device)
        output = model(**batch)
        hidden = output.last_hidden_state[0, 1 : len(sequence) + 1]
        embeddings = hidden.cpu().numpy().round(5).tolist()
    return {
        "embeddings": embeddings,
        "mean": [round(float(v), 5) for v in hidden.mean(0).cpu().numpy().tolist()],
        "model_id": model_id,
        "device": device,
        "latency_ms": round((time.time() - started) * 1000, 1),
    }
