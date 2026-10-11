"""Backend selection: local NIM -> hosted NIM -> local BioNeMo runtime -> simulator.

The console never silently swaps one for another.  :class:`RunResult` records
which backend answered, over which URL, with which parameters and in how long,
and that provenance is attached to every result, exported file and report.
"""

from __future__ import annotations

import datetime as _dt
import logging
import platform
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from . import catalog
from . import sim as sim_mod
from .client import NimCallResult, NimClient, ProbeResult, normalise
from .local_runtime import detect_environment

log = logging.getLogger("bionemo_web.service")

APP_VERSION = "1.0.0"

BACKENDS = ["auto", "local", "hosted", "framework", "simulate"]

BACKEND_LABELS = {
    "auto": "Auto (prefer real services)",
    "local": "Local NIM container",
    "hosted": "NVIDIA-hosted NIM (API key)",
    "framework": "Local BioNeMo / PyTorch runtime",
    "simulate": "Built-in simulation",
}


@dataclass
class RunResult:
    """Everything that happened during one model invocation."""

    ok: bool
    spec_key: str
    backend: str
    method: str = ""
    canonical: Dict[str, Any] = field(default_factory=dict)
    raw: Any = None
    meta: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    error: str = ""
    latency_ms: float = 0.0
    url: str = ""
    status: Optional[int] = None
    request_payload: Dict[str, Any] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    attempts: int = 1
    simulated: bool = False
    created_at: str = field(default_factory=lambda: _dt.datetime.now().isoformat(timespec="seconds"))
    fallback_chain: List[str] = field(default_factory=list)

    def provenance(self, spec: Optional[catalog.NimSpec] = None) -> Dict[str, Any]:
        """Reproducibility manifest for this run."""
        spec = spec or catalog.get(self.spec_key)
        return {
            "app": "Dr-Titan BioNeMo Web Console",
            "app_version": APP_VERSION,
            "created_at": self.created_at,
            "model_key": self.spec_key,
            "model_name": spec.name if spec else self.spec_key,
            "vendor": spec.vendor if spec else "",
            "backend": self.backend,
            "backend_label": BACKEND_LABELS.get(self.backend, self.backend),
            "method": self.method,
            "simulated": self.simulated,
            "url": self.url,
            "http_status": self.status,
            "latency_ms": self.latency_ms,
            "attempts": self.attempts,
            "request": self.request_payload,
            "fallback_chain": self.fallback_chain,
            "environment": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
            },
        }

    def badge(self) -> Tuple[str, str]:
        """(label, tone) for the UI status chip."""
        if self.backend == "local":
            return ("LOCAL NIM", "good")
        if self.backend == "hosted":
            return ("NVIDIA HOSTED", "good")
        if self.backend == "framework":
            return ("LOCAL RUNTIME", "info")
        return ("SIMULATED", "warn")


class NimService:
    """Runs models, choosing a backend, and explains what it chose."""

    def __init__(self, client: NimClient, settings: Dict[str, Any]) -> None:
        self.client = client
        self.settings = settings
        self._env_cache: Optional[Dict[str, Any]] = None

    # -- environment ---------------------------------------------------
    def environment(self) -> Dict[str, Any]:
        if self._env_cache is None:
            self._env_cache = detect_environment()
        return self._env_cache

    def refresh_environment(self) -> Dict[str, Any]:
        self._env_cache = detect_environment()
        return self._env_cache

    # -- backend decision ---------------------------------------------
    def decide_backend(self, spec: catalog.NimSpec, requested: str = "auto") -> Tuple[str, List[str]]:
        """Return the backend to use plus the chain of attempts that was considered."""
        chain: List[str] = []
        prefer = requested if requested != "auto" else str(self.settings.get("prefer_backend", "auto"))
        allow_hosted = bool(self.settings.get("allow_hosted", True))
        has_key = bool(self.client.hosted_api_key)

        if requested == "auto" and prefer != "auto":
            requested = prefer  # respect the preference but keep fallbacks

        if requested == "local":
            return "local", ["local"]
        if requested == "hosted":
            return "hosted", ["hosted"]
        if requested == "simulate":
            return "simulate", ["simulate"]
        if requested == "framework":
            return "framework", ["framework"]

        # auto
        if self.client.base_url(spec, "local"):
            chain.append("local")
        if allow_hosted and has_key and spec.hosted_path and not spec.local_only:
            chain.append("hosted")
        if self.environment().get("framework_checkout") or self.environment().get("torch"):
            chain.append("framework")
        chain.append("simulate")
        # When a local endpoint is configured we still trust it first but keep the
        # simulator as an explicit fallback, which the provenance records.
        return chain[0], chain

    # -- running -------------------------------------------------------
    def run(
        self,
        model: Union[str, catalog.NimSpec],
        payload: Dict[str, Any],
        *,
        backend: str = "auto",
        timeout: Optional[int] = None,
        seed: Optional[int] = None,
        url_override: str = "",
        allow_fallback: bool = True,
    ) -> RunResult:
        spec = model if isinstance(model, catalog.NimSpec) else catalog.get(model)
        if spec is None:
            return RunResult(
                ok=False,
                spec_key=str(model),
                backend="none",
                error=f"Unknown model key: {model}",
                warnings=["Unknown model."],
            )
        chosen, chain = self.decide_backend(spec, backend)
        seed = int(seed if seed is not None else self.settings.get("simulation_seed", 20241011))
        attempts: List[str] = []

        order = [chosen] + [b for b in chain if b != chosen] if allow_fallback else [chosen]
        last_error = ""
        for attempt_backend in order:
            attempts.append(attempt_backend)
            if attempt_backend in ("local", "hosted"):
                if attempt_backend == "local" and not self.client.base_url(spec, "local"):
                    last_error = "No local endpoint configured."
                    continue
                result = self.client.call(
                    spec,
                    payload,
                    backend=attempt_backend,
                    timeout=timeout,
                    url_override=url_override if attempt_backend == chosen else "",
                )
                if result.ok and result.canonical:
                    return self._from_call(result, spec, attempts, simulated=False)
                last_error = result.error or "The service returned no usable result."
                log.warning("%s backend failed for %s: %s", attempt_backend, spec.key, last_error)
                continue

            if attempt_backend == "framework":
                result = self._run_framework(spec, payload)
                if result.ok:
                    return self._finish(result, spec, attempts, simulated=False)
                last_error = result.error
                continue

            if attempt_backend == "simulate":
                bundle = sim_mod.simulate(spec, payload, seed=seed)
                canonical = bundle.get("canonical") or {}
                ok = bool(canonical)
                result = RunResult(
                    ok=ok,
                    spec_key=spec.key,
                    backend="simulate",
                    method=bundle.get("method", ""),
                    canonical=canonical,
                    raw=bundle.get("raw"),
                    meta=bundle.get("meta", {}),
                    warnings=list(bundle.get("warnings", [])),
                    error="" if ok else "Simulation produced no result.",
                    latency_ms=0.0,
                    url="",
                    request_payload=payload,
                    simulated=True,
                )
                if not ok and allow_fallback:
                    last_error = result.error
                    continue
                if last_error:
                    result.warnings.insert(
                        0,
                        "Real backends were unavailable or failed, so this result came from the "
                        f"built-in simulation. Last error: {last_error}",
                    )
                return self._finish(result, spec, attempts, simulated=True)

        # Nothing worked.
        return RunResult(
            ok=False,
            spec_key=spec.key,
            backend=order[-1] if order else "none",
            error=last_error or "No backend available.",
            warnings=["All backends failed or are unavailable."],
            request_payload=payload,
            fallback_chain=attempts,
        )

    # -- helpers -------------------------------------------------------
    def _from_call(
        self, call: NimCallResult, spec: catalog.NimSpec, attempts: List[str], simulated: bool
    ) -> RunResult:
        result = RunResult(
            ok=call.ok,
            spec_key=spec.key,
            backend=call.backend,
            method=f"{spec.name} NIM over HTTP",
            canonical=call.canonical,
            raw=call.raw,
            warnings=list(call.warnings),
            error=call.error,
            latency_ms=call.latency_ms,
            url=call.url,
            status=call.status,
            request_payload=call.request_payload,
            headers=call.headers,
            attempts=call.attempts,
            simulated=simulated,
        )
        return self._finish(result, spec, attempts, simulated=simulated)

    def _finish(
        self, result: RunResult, spec: catalog.NimSpec, attempts: List[str], simulated: bool
    ) -> RunResult:
        result.fallback_chain = attempts
        result.simulated = simulated
        if len(attempts) > 1:
            previous = ", ".join(attempts[:-1])
            result.warnings.append(
                f"Fell back to the '{attempts[-1]}' backend after trying: {previous}."
            )
        # Enrich meta with downstream analysis where it is cheap and useful.
        if spec.kind == "fold" and result.canonical.get("pdbs"):
            from ..bio import structure as st_mod

            try:
                structure = st_mod.parse_pdb(
                    result.canonical["pdbs"][0],
                    source=result.backend,
                    simulated=simulated,
                )
                result.meta.setdefault("structure", structure)
                result.meta.setdefault("structure_summary", structure.to_dict(profiles=True))
            except Exception as exc:  # pragma: no cover
                result.warnings.append(f"Could not re-parse the returned structure: {exc}")
        if spec.kind == "dock" and result.canonical.get("ligand_positions"):
            result.meta.setdefault(
                "n_poses", len(result.canonical.get("ligand_positions") or [])
            )
        return result

    def _run_framework(self, spec: catalog.NimSpec, payload: Dict[str, Any]) -> RunResult:
        """Try the in-process BioNeMo / PyTorch path (ESM-2 embeddings today)."""
        env = self.environment()
        if not env.get("torch"):
            return RunResult(
                ok=False, spec_key=spec.key, backend="framework",
                error="PyTorch is not installed in this environment.",
                warnings=["Install torch/transformers to enable the local runtime backend."],
                request_payload=payload,
            )
        if spec.key == "esm2":
            from .local_runtime import esm2_embeddings_torch

            sequence = payload.get("sequence") or ""
            out = esm2_embeddings_torch(sequence, self.settings.get("local_model_id", "facebook/esm2_t12_35M_UR50D"))
            if out is None:
                return RunResult(
                    ok=False, spec_key=spec.key, backend="framework",
                    error="The local model could not be loaded (no weights available offline?).",
                    request_payload=payload,
                )
            return RunResult(
                ok=True,
                spec_key=spec.key,
                backend="framework",
                method="transformers ESM-2 forward pass (local runtime)",
                canonical={"embeddings": out["embeddings"], "mean": out["mean"]},
                raw={"embeddings": out["embeddings"]},
                meta={"model_id": out["model_id"], "device": out["device"]},
                request_payload=payload,
                latency_ms=out.get("latency_ms", 0.0),
            )
        return RunResult(
            ok=False, spec_key=spec.key, backend="framework",
            error=f"No local framework runner is implemented for '{spec.key}' "
                  "(BioNeMo recipes are launched as containers or via the framework CLI).",
            request_payload=payload,
        )

    # -- diagnostics ---------------------------------------------------
    def probe_model(self, spec: catalog.NimSpec, backends: Optional[List[str]] = None) -> Dict[str, ProbeResult]:
        out: Dict[str, ProbeResult] = {}
        for backend in backends or ["local", "hosted"]:
            out[backend] = self.client.probe(spec, backend)
        return out

    def probe_endpoint(self, base_url: str) -> Tuple[ProbeResult, List[Dict[str, str]]]:
        """Probe an arbitrary base URL and list the operations it advertises.

        Used by the Settings page's connection tester: paste a URL, learn whether
        a NIM is listening there and which paths it exposes.
        """
        spec = catalog.NimSpec(
            key="_probe", name="Endpoint probe", vendor="", task="", kind="fold",
            description="", hosted_path="", local_path="",
        )
        saved = self.client.local_endpoints.get("_probe")
        self.client.local_endpoints["_probe"] = base_url.rstrip("/")
        try:
            probe = self.client.probe(spec, "local")
            operations = self.client.list_operations(base_url.rstrip("/")) if probe.reachable else []
        finally:
            if saved is None:
                self.client.local_endpoints.pop("_probe", None)
            else:
                self.client.local_endpoints["_probe"] = saved
        return probe, operations


def _probe_url(client: NimClient, base_url: str) -> ProbeResult:
    """Probe a raw base URL (used by the Settings page's connection tester)."""
    from .catalog import NimSpec

    spec = NimSpec(key="_probe", name="Endpoint probe", vendor="", task="", kind="fold",
                   description="", hosted_path="", local_path="")
    client.local_endpoints["_probe"] = base_url
    try:
        return client.probe(spec, "local")
    finally:
        client.local_endpoints.pop("_probe", None)
