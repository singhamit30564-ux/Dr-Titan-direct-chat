"""Language-model providers, with a fully offline fallback.

The console is useful with **no** LLM at all: model calls are chosen by a
deterministic router and the science comes from the NIMs / analysis engines.
A language model only ever does two jobs here --

1. explain and summarise results that were already computed, and
2. answer free-form questions about the work in the session.

Providers, in the order they are tried: a local OpenAI-compatible server (vLLM,
a NIM LLM container, LM Studio, llama.cpp server), Ollama, NVIDIA's hosted
endpoint (``integrate.api.nvidia.com``), and OpenAI.  If none answers -- or the
user switches it off -- :class:`OfflineAnalyst` writes the summary from the
result objects themselves.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple

import requests

log = logging.getLogger("bionemo_web.llm")

NVIDIA_BASE = "https://integrate.api.nvidia.com/v1"
OPENAI_BASE = "https://api.openai.com/v1"
OLLAMA_BASE = "http://localhost:11434"

#: Models that are known-good for tool-free summarisation on build.nvidia.com.
NVIDIA_MODELS = [
    "meta/llama-3.1-70b-instruct",
    "meta/llama-3.1-8b-instruct",
    "nvidia/llama-3.1-nemotron-70b-instruct",
    "mistralai/mistral-nemotron",
    "qwen/qwen2.5-32b-instruct",
]

OPENAI_MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"]

DEFAULT_SYSTEM_PROMPT = (
    "You are the analysis assistant inside a local BioNeMo console used by protein "
    "engineers and medicinal chemists. You are given structured results that were "
    "computed by NVIDIA BioNeMo NIMs or by the console's own verified analysis "
    "engines, together with a provenance record stating which backend produced each "
    "number and whether it was simulated.\n\n"
    "Rules you must follow:\n"
    "1. Ground every quantitative claim in the supplied results. Never invent a "
    "number, a pLDDT, a docking score or a property value.\n"
    "2. If a result is marked SIMULATED, say so once, plainly, and describe it as a "
    "geometric/physical model rather than a prediction.\n"
    "3. Be concrete and terse: 3-8 sentences or a short list. No filler, no "
    "restating the question, no marketing language.\n"
    "4. Flag genuine caveats that matter scientifically (low confidence regions, "
    "clashes, strained geometry, PAINS alerts, absent MSA) and skip the generic ones.\n"
    "5. Never give dosing, clinical, or synthesis instructions for anything that "
    "looks like a controlled substance; note the flag and move on."
)


@dataclass
class LLMConfig:
    """Everything needed to talk to one provider."""

    provider: str = "auto"        # auto|local|ollama|nvidia|openai|off
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    temperature: float = 0.2
    max_tokens: int = 900
    timeout: int = 60
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    enabled: bool = True

    @classmethod
    def from_settings(cls, settings: Dict[str, Any]) -> "LLMConfig":
        return cls(
            provider=str(settings.get("llm_provider", "auto")),
            model=str(settings.get("llm_model", "")),
            base_url=str(settings.get("llm_base_url", "")),
            api_key=str(settings.get("llm_api_key", "")),
            temperature=float(settings.get("llm_temperature", 0.2)),
            max_tokens=int(settings.get("llm_max_tokens", 900)),
            timeout=int(settings.get("llm_timeout", 60)),
            enabled=bool(settings.get("llm_enabled", True)),
        )

    def resolved(self) -> Tuple[str, str, str]:
        """Return ``(base_url, api_key, model)`` for the configured provider."""
        provider = self.provider
        if provider == "auto":
            env_local = os.environ.get("LOCAL_LLM_BASE_URL") or os.environ.get("VLLM_BASE_URL")
            if env_local:
                return env_local.rstrip("/"), self.api_key or os.environ.get("LOCAL_LLM_API_KEY", ""), self.model or "local-model"
            if os.environ.get("NVIDIA_API_KEY") or os.environ.get("NGC_API_KEY"):
                return NVIDIA_BASE, os.environ.get("NVIDIA_API_KEY") or os.environ.get("NGC_API_KEY", ""), self.model or NVIDIA_MODELS[0]
            if os.environ.get("OPENAI_API_KEY"):
                return OPENAI_BASE, os.environ["OPENAI_API_KEY"], self.model or OPENAI_MODELS[0]
            return OLLAMA_BASE, "", self.model or "llama3.1:8b"
        if provider == "local":
            return (self.base_url or os.environ.get("LOCAL_LLM_BASE_URL", "http://localhost:8000/v1")).rstrip("/"), self.api_key, self.model or "local-model"
        if provider == "ollama":
            return OLLAMA_BASE, "", self.model or "llama3.1:8b"
        if provider == "nvidia":
            return NVIDIA_BASE, self.api_key or os.environ.get("NVIDIA_API_KEY", ""), self.model or NVIDIA_MODELS[0]
        if provider == "openai":
            return OPENAI_BASE, self.api_key or os.environ.get("OPENAI_API_KEY", ""), self.model or OPENAI_MODELS[0]
        return "", "", ""


@dataclass
class LLMStatus:
    available: bool
    provider: str
    model: str
    message: str
    models: List[str] = field(default_factory=list)


class LLMClient:
    """A thin, forgiving OpenAI-compatible client."""

    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        self.config = config or LLMConfig()
        self._session = requests.Session()
        self._status: Optional[LLMStatus] = None

    # -- discovery -----------------------------------------------------
    def status(self, *, refresh: bool = False) -> LLMStatus:
        if self._status is not None and not refresh:
            return self._status
        cfg = self.config
        if not cfg.enabled or cfg.provider == "off":
            self._status = LLMStatus(False, "off", "", "Language model disabled; using the offline analyst.")
            return self._status
        base, key, model = cfg.resolved()
        if not base:
            self._status = LLMStatus(False, cfg.provider, model, "No provider configured.")
            return self._status
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        try:
            resp = self._session.get(f"{base}/models", headers=headers, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                models = [
                    m.get("id") for m in (data.get("data") or [])
                    if isinstance(m, dict) and m.get("id")
                ]
                chosen = model if (not models or model in models) else (models[0] if models else model)
                self._status = LLMStatus(True, cfg.provider, chosen, f"Connected to {base}", models[:60])
                return self._status
            if resp.status_code in (401, 403):
                self._status = LLMStatus(False, cfg.provider, model, f"Authentication failed at {base} (HTTP {resp.status_code}).")
                return self._status
            self._status = LLMStatus(False, cfg.provider, model, f"{base} answered HTTP {resp.status_code}.")
            return self._status
        except Exception as exc:
            self._status = LLMStatus(False, cfg.provider, model, f"{base} unreachable: {str(exc)[:120]}")
            return self._status

    # -- completion ----------------------------------------------------
    def complete(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> str:
        base, key, model = self.config.resolved()
        if not base:
            return ""
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": self.config.temperature if temperature is None else temperature,
            "max_tokens": self.config.max_tokens if max_tokens is None else max_tokens,
            "stream": False,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        try:
            resp = self._session.post(
                f"{base}/chat/completions", json=payload, headers=headers, timeout=self.config.timeout
            )
            if resp.status_code >= 400:
                log.warning("LLM HTTP %s: %s", resp.status_code, resp.text[:200])
                return ""
            data = resp.json()
            return (data.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
        except Exception as exc:
            log.warning("LLM call failed: %s", exc)
            return ""

    def stream(
        self, messages: List[Dict[str, str]], **kwargs: Any
    ) -> Generator[str, None, None]:
        """Yield content deltas; falls back to a single chunk if streaming is unsupported."""
        base, key, model = self.config.resolved()
        if not base:
            return
        payload = {
            "model": model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "stream": True,
        }
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        try:
            with self._session.post(
                f"{base}/chat/completions", json=payload, headers=headers,
                timeout=self.config.timeout, stream=True,
            ) as resp:
                if resp.status_code >= 400:
                    text = self.complete(messages, **kwargs)
                    if text:
                        yield text
                    return
                for line in resp.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data:"):
                        continue
                    chunk = line[5:].strip()
                    if chunk == "[DONE]":
                        break
                    try:
                        data = json.loads(chunk)
                    except Exception:
                        continue
                    delta = (data.get("choices") or [{}])[0].get("delta", {})
                    if delta.get("content"):
                        yield delta["content"]
        except Exception as exc:
            log.warning("LLM stream failed: %s", exc)
            text = self.complete(messages, **kwargs)
            if text:
                yield text

    def summarise(self, instruction: str, context: str) -> str:
        """One-shot: explain pre-computed results. Empty string means 'no model'."""
        if not self.status().available:
            return ""
        messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": f"{instruction}\n\n--- RESULTS ---\n{context[:12000]}\n--- END RESULTS ---"},
        ]
        return self.complete(messages).strip()


# --------------------------------------------------------------------------
# Offline analyst
# --------------------------------------------------------------------------


class OfflineAnalyst:
    """Writes grounded summaries without any model at all.

    Each method takes the same structured results the LLM would receive and
    produces a short, factual paragraph.  This is the default narrator, so the
    console never shows an empty answer.
    """

    @staticmethod
    def summarise_run(run: Any) -> str:
        """Describe a :class:`~bionemo_web.nims.service.RunResult` in words."""
        if run is None:
            return ""
        if not getattr(run, "ok", False):
            return (
                f"The {getattr(run, 'spec_key', 'model')} run did not produce a result. "
                f"{getattr(run, 'error', '') or 'See the warnings below.'}"
            )
        spec_key = getattr(run, "spec_key", "")
        canonical = getattr(run, "canonical", {}) or {}
        meta = getattr(run, "meta", {}) or {}
        parts: List[str] = []

        if spec_key in ("esmfold", "alphafold2", "openfold3", "boltz2", "rfdiffusion"):
            structure = meta.get("structure")
            if structure is not None:
                plddt = structure.plddt()
                mean = float(plddt.mean()) if plddt is not None and len(plddt) else None
                ss = structure.secondary_structure() if structure.n_residues else ""
                from collections import Counter

                comp = Counter(ss)
                parts.append(
                    f"Model built with {structure.n_residues} residues; radius of gyration "
                    f"{structure.radius_of_gyration():.1f} A."
                )
                if mean is not None:
                    parts.append(f"Mean per-residue confidence {mean:.1f}.")
                if comp:
                    parts.append(
                        f"Secondary structure: {comp.get('H', 0)} helical, {comp.get('E', 0)} strand, "
                        f"{comp.get('C', 0) + comp.get('T', 0)} coil/turn residues."
                    )
                if mean is not None and mean < 70:
                    parts.append("Confidence is low in places - treat the fold as a hypothesis.")
        elif spec_key == "esm2":
            quality = meta.get("quality") or {}
            if quality:
                parts.append(
                    f"Produced {quality.get('n_residues')} per-residue vectors of dimension "
                    f"{quality.get('dim')}; mean pairwise cosine "
                    f"{quality.get('mean_pairwise_cosine')}."
                )
        elif spec_key in ("molmim", "genmol"):
            rows = meta.get("rows") or []
            summary = meta.get("set_summary") or {}
            if rows:
                top = rows[0]
                parts.append(f"Generated {len(rows)} molecules; best {top['property']} = {top['score']:.3f}.")
                parts.append(
                    f"Similarity to the seed ranges {min(r['similarity'] for r in rows):.2f}-"
                    f"{max(r['similarity'] for r in rows):.2f}."
                )
                if summary:
                    parts.append(
                        f"Set-average MW {summary.get('mean_mw')} Da, cLogP {summary.get('mean_clogp')}, "
                        f"QED {summary.get('mean_qed')}; {int(100 * summary.get('ro5_pass_rate', 0))}% pass "
                        f"Lipinski, {summary.get('unique_scaffolds')} distinct scaffolds."
                    )
                alerts = summary.get("alerts", 0)
                if alerts:
                    parts.append(f"{alerts} structural alert(s) flagged (PAINS/Brenk/reactive groups).")
        elif spec_key == "diffdock":
            poses = meta.get("poses") or []
            if poses:
                best = poses[0]
                parts.append(
                    f"{len(poses)} poses; best confidence {best.confidence:.2f} in "
                    f"{best.pocket_id or 'the top cavity'}."
                )
                parts.append(
                    f"That pose forms {best.hbonds} H-bond(s) and {best.contacts} heavy-atom contacts "
                    f"with {len(best.interaction_residues)} residues"
                    + (f" ({', '.join(best.interaction_residues[:6])}...)." if best.interaction_residues else ".")
                )
                if best.clashes:
                    parts.append(f"{best.clashes} steric overlap(s) remain - the pose is not physically clean.")
                pockets = meta.get("pockets") or []
                if pockets:
                    parts.append(
                        f"Docking considered {len(pockets)} detected cavity(ies); the top one has volume "
                        f"{pockets[0].get('volume_a3')} A^3 and druggability hint "
                        f"{pockets[0].get('druggability_hint')}."
                    )
        elif spec_key == "proteinmpnn":
            designs = meta.get("designs") or []
            if designs:
                top = designs[0]
                muts = top.get("mutations") or []
                parts.append(
                    f"{len(designs)} designs at temperature {meta.get('temperature')}; "
                    f"best sequence recovers {100 * top['recovery']:.0f}% of the wild type "
                    f"and carries {len(muts)} substitution(s)."
                )
                hydrophobic = sum(1 for m in muts if m["designed"] in "AVILMFW")
                parts.append(f"{hydrophobic} of those substitutions introduce hydrophobic residues.")
        elif spec_key == "evo2":
            analysis = meta.get("analysis") or {}
            if analysis:
                parts.append(
                    f"Generated {len(meta.get('generated', ''))} nt; GC content "
                    f"{100 * analysis.get('gc_content', 0):.1f}%, effective codon number "
                    f"{analysis.get('effective_codons')}."
                )
                parts.append(
                    f"Longest ORF {analysis.get('longest_orf_aa')} aa; {analysis.get('orf_count')} ORFs "
                    f"over 30 aa in the product."
                )
        else:
            keys = ", ".join(sorted(canonical)[:6])
            parts.append(f"Result contains: {keys}.")

        if getattr(run, "simulated", False):
            parts.append(
                "This run used the built-in simulation backend, so the numbers come from the "
                "documented model described above rather than from a NIM prediction."
            )
        return " ".join(parts)

    @staticmethod
    def answer_question(question: str, context: Dict[str, Any]) -> str:
        """Answer from the session context when no model is available."""
        q = (question or "").lower()
        stats = context.get("stats") or {}
        last = context.get("last_run") or {}
        if any(word in q for word in ("what can you", "help", "commands", "how do i")):
            return OfflineAnalyst.capabilities()
        if "pocket" in q:
            pockets = (last.get("meta") or {}).get("pockets") or []
            if pockets:
                return "Detected cavities (volume A^3, druggability hint): " + "; ".join(
                    f"{p['id']} {p.get('volume_a3')} / {p.get('druggability_hint')}" for p in pockets[:5]
                )
            return "No pocket analysis is in the session yet - run a fold or a docking job first."
        if "simulat" in q or "real" in q:
            return (
                f"This session has {stats.get('simulated_runs', 0)} simulated run(s) and "
                f"{stats.get('real_runs', 0)} run(s) served by a real backend. "
                "Every result card carries a backend badge and the run inspector stores the full provenance."
            )
        return (
            "No language model is connected, so I answer from the computed results only. "
            + OfflineAnalyst.capabilities()
        )

    @staticmethod
    def capabilities() -> str:
        return (
            "I can fold sequences (ESMFold/AlphaFold2/OpenFold3/Boltz-2), generate molecules "
            "(MolMIM/GenMol), dock ligands (DiffDock), design sequences against a backbone "
            "(ProteinMPNN/RFdiffusion), embed sequences (ESM-2/AMPLIFY/Geneformer) and "
            "generate or score DNA (Evo 2/CodonFM) - through local NIM containers, NVIDIA's "
            "hosted endpoints, or the built-in simulation backends. Ask in plain language, "
            "e.g. 'fold MKTAYIAK...', 'generate 10 analogues of aspirin optimising QED', "
            "'dock ibuprofen into this PDB', or 'run the virtual screening pipeline'. "
            "Connect a local LLM (Ollama or any OpenAI-compatible server) in Settings for "
            "richer explanations."
        )


def context_for_llm(run: Any, extra: Optional[Dict[str, Any]] = None) -> str:
    """Build a compact, number-dense context block for a model to summarise."""
    if run is None:
        return json.dumps(extra or {}, indent=2, default=str)[:8000]
    spec_key = getattr(run, "spec_key", "")
    meta = getattr(run, "meta", {}) or {}
    block: Dict[str, Any] = {
        "model": spec_key,
        "backend": getattr(run, "backend", ""),
        "simulated": bool(getattr(run, "simulated", False)),
        "method": getattr(run, "method", ""),
        "latency_ms": getattr(run, "latency_ms", 0),
        "warnings": list(getattr(run, "warnings", []) or []),
        "error": getattr(run, "error", ""),
    }
    structure = meta.get("structure")
    if structure is not None:
        block["structure"] = {
            "n_residues": structure.n_residues,
            "radius_of_gyration": round(structure.radius_of_gyration(), 2),
            "mean_plddt": round(float(structure.plddt().mean()), 2) if structure.plddt() is not None else None,
            "secondary_structure": structure.secondary_structure()[:400],
            "pockets": (meta.get("pockets") or [])[:5],
        }
    for key in ("pose_stats", "rows", "designs", "set_summary", "quality", "analysis", "fold_meta"):
        if key in meta:
            value = meta[key]
            if isinstance(value, list) and value and isinstance(value[0], dict):
                block[key] = value[:8]
            elif isinstance(value, dict):
                block[key] = {k: v for k, v in list(value.items())[:25]}
    if extra:
        block.update(extra)
    return json.dumps(block, indent=2, default=str)[:12000]


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Pull the first JSON object out of a model response."""
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text.strip())
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
    return None
