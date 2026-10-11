"""Sequence embeddings and low-dimensional projections.

Used for the ESM-2 / AMPLIFY / Geneformer embedding views.  When a BioNeMo NIM
answers, its vectors are used directly.  When the console is in simulation mode
it builds **physicochemical embeddings** instead: each residue is described by a
small set of published properties (hydrophobicity, volume, charge, polarity,
flexibility, helix/sheet propensity) plus a windowed context average, then the
matrix is centred.  These are real feature vectors that cluster biophysically
sensible residues together -- they are emphatically *not* the output of a
language model, and the UI labels them as simulation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import constants as C

#: Feature axes used by the fallback (simulation) embedding.
FEATURE_NAMES = [
    "kyte_doolittle",
    "eisenberg",
    "side_chain_volume",
    "flexibility",
    "charge",
    "polarity",
    "helix_propensity",
    "sheet_propensity",
    "aromatic",
    "is_context_window",
]

_CHARGE = {"K": 1.0, "R": 1.0, "H": 0.3, "D": -1.0, "E": -1.0}
_POLARITY = {"S": 1.0, "T": 1.0, "N": 1.0, "Q": 1.0, "C": 0.5, "Y": 0.6,
             "K": 0.8, "R": 0.8, "H": 0.7, "D": 1.0, "E": 1.0}
_AROMATIC = {"F": 1.0, "W": 1.0, "Y": 1.0, "H": 0.5}


def residue_feature_matrix(seq: str, window: int = 5) -> np.ndarray:
    """Per-residue property matrix, shape ``(len(seq), 10)``."""
    n = len(seq)
    if n == 0:
        return np.zeros((0, len(FEATURE_NAMES)))
    raw = np.zeros((n, len(FEATURE_NAMES) - 1))
    for i, aa in enumerate(seq):
        raw[i] = [
            C.KYTE_DOOLITTLE.get(aa, 0.0),
            C.EISENBERG.get(aa, 0.0),
            C.SIDE_CHAIN_VOLUME.get(aa, 100.0) / 100.0,
            C.FLEXIBILITY.get(aa, 1.0) - 1.0,
            _CHARGE.get(aa, 0.0),
            _POLARITY.get(aa, 0.0),
            1.0 if aa in C.HELIX_FORMERS else 0.0,
            1.0 if aa in C.SHEET_FORMERS else 0.0,
            _AROMATIC.get(aa, 0.0),
        ]
    half = window // 2
    context = np.zeros(n)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        context[i] = float(raw[lo:hi, 0].mean())  # windowed hydrophobicity
    matrix = np.hstack([raw, context[:, None]])
    # Standardise columns so no single property dominates a distance metric.
    mean = matrix.mean(0, keepdims=True)
    std = matrix.std(0, keepdims=True)
    std[std < 1e-9] = 1.0
    return (matrix - mean) / std


def simulated_embeddings(seq: str, dim: int = 128, window: int = 5, seed: int = 7) -> List[List[float]]:
    """Expand the property matrix into ``dim``-dimensional pseudo-embeddings.

    A fixed random projection of the (real) physicochemical features.  The
    projection is deterministic for a given seed, so downstream clustering and
    similarity are reproducible run to run.
    """
    features = residue_feature_matrix(seq, window)
    if features.shape[0] == 0:
        return []
    rng = np.random.default_rng(seed)
    projection = rng.normal(0.0, 1.0 / np.sqrt(features.shape[1]), size=(features.shape[1], dim))
    projected = features @ projection
    # Add a smooth positional component so chain order is visible in 2D views.
    pos = np.linspace(-1.0, 1.0, features.shape[0])[:, None]
    projected[:, 0] += pos[:, 0] * 0.5
    projected[:, 1] += np.sin(np.linspace(0, 4 * np.pi, features.shape[0])) * 0.25
    norms = np.linalg.norm(projected, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    return (projected / norms).round(5).tolist()


# --------------------------------------------------------------------------
# Projections & similarity
# --------------------------------------------------------------------------


def pca(matrix: np.ndarray, n_components: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Principal-component projection (numpy SVD; no scikit-learn needed)."""
    if matrix.size == 0:
        return np.zeros((0, n_components)), np.zeros(0)
    centred = matrix - matrix.mean(0, keepdims=True)
    u, s, _vt = np.linalg.svd(centred, full_matrices=False)
    coords = u[:, :n_components] * s[:n_components]
    variance = (s**2) / max(1, (matrix.shape[0] - 1))
    explained = variance / variance.sum() if variance.sum() else np.zeros_like(variance)
    return coords, explained[:n_components]


def project_embeddings(embeddings: Sequence[Sequence[float]], method: str = "pca") -> Dict:
    """2D projection of an embedding matrix (``pca`` or ``pca3``)."""
    matrix = np.asarray(embeddings, dtype=float)
    if matrix.ndim != 2 or matrix.size == 0:
        return {"x": [], "y": [], "explained": []}
    n_components = 3 if method == "pca3" else 2
    coords, explained = pca(matrix, n_components)
    out = {
        "x": [round(float(v), 4) for v in coords[:, 0]] if coords.shape[1] > 0 else [],
        "y": [round(float(v), 4) for v in coords[:, 1]] if coords.shape[1] > 1 else [],
        "explained": [round(float(v), 4) for v in explained],
        "n": int(matrix.shape[0]),
        "dim": int(matrix.shape[1]),
    }
    if n_components == 3 and coords.shape[1] > 2:
        out["z"] = [round(float(v), 4) for v in coords[:, 2]]
    return out


def cosine_similarity_matrix(embeddings: Sequence[Sequence[float]]) -> List[List[float]]:
    m = np.asarray(embeddings, dtype=float)
    if m.ndim != 2 or m.size == 0:
        return []
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    unit = m / norms
    sim = unit @ unit.T
    return [[round(float(v), 4) for v in row] for row in sim]


def nearest_neighbours(embeddings: Sequence[Sequence[float]], index: int, k: int = 8) -> List[Tuple[int, float]]:
    """Cosine nearest neighbours of one residue embedding."""
    m = np.asarray(embeddings, dtype=float)
    if m.ndim != 2 or index < 0 or index >= len(m):
        return []
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    unit = m / norms
    sim = unit @ unit[index]
    order = np.argsort(-sim)
    return [(int(i), round(float(sim[i]), 4)) for i in order if i != index][:k]


def cluster_embeddings(embeddings: Sequence[Sequence[float]], n_clusters: int = 6, seed: int = 0) -> List[int]:
    """k-means over embedding rows (deterministic k-means++ initialisation)."""
    m = np.asarray(embeddings, dtype=float)
    n = len(m)
    if n == 0:
        return []
    k = int(max(1, min(n_clusters, n)))
    rng = np.random.default_rng(seed)
    centres = [m[rng.integers(0, n)]]
    for _ in range(k - 1):
        d2 = np.min(
            np.stack([((m - c) ** 2).sum(1) for c in centres], axis=1),
            axis=1,
        )
        total = d2.sum()
        probs = d2 / total if total > 0 else np.full(n, 1.0 / n)
        centres.append(m[rng.choice(n, p=probs)])
    centres_arr = np.stack(centres)
    labels = np.zeros(n, dtype=int)
    for _ in range(40):
        dist = np.stack([((m - c) ** 2).sum(1) for c in centres_arr], axis=1)
        new_labels = dist.argmin(1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(k):
            members = m[labels == c]
            if len(members):
                centres_arr[c] = members.mean(0)
    return labels.tolist()


def embedding_quality_report(seq: str, embeddings: Sequence[Sequence[float]]) -> Dict:
    """Summarise an embedding run in terms the UI can display."""
    m = np.asarray(embeddings, dtype=float)
    if m.ndim != 2 or m.size == 0:
        return {"n_residues": 0}
    per_residue_norm = np.linalg.norm(m, axis=1)
    return {
        "n_residues": int(m.shape[0]),
        "dim": int(m.shape[1]),
        "mean_norm": round(float(per_residue_norm.mean()), 4),
        "norm_std": round(float(per_residue_norm.std()), 4),
        "mean_pairwise_cosine": round(
            float(
                (
                    (m / np.maximum(per_residue_norm[:, None], 1e-9))
                    @ (m / np.maximum(per_residue_norm[:, None], 1e-9)).T
                ).mean()
            ),
            4,
        ),
        "unique_letters": len(set(seq)),
    }
