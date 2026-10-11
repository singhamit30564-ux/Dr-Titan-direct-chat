"""HTTP client for BioNeMo NIM microservices.

Handles the three things that actually break when you wire a UI to NIM
containers, instead of assuming they do not:

1. **Readiness.**  A NIM can answer on port 8000 while the model is still
   loading.  :meth:`NimClient.probe` reads ``/v1/health/ready``, ``/v1/models``,
   ``/v1/metadata`` and ``/v1/version`` and reports what it found.
2. **Route drift.**  NIM route prefixes differ between containers and versions
   (``/biology/mit/esmfold`` in a local container vs ``/biology/mit/esmfold/...``
   hosted, ``/generate`` vs ``/biology/nvidia/molmim/generate``).  Rather than
   hard-failing, the client can read ``/openapi.json`` and match operations by
   keyword, then remember the answer.
3. **Response shape drift.**  The same model returns ``molecules`` hosted and
   ``generated`` locally.  :func:`normalise` maps every known shape onto one
   canonical result the UI understands.
"""

from __future__ import annotations

import json
import logging
import re
import ssl
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests

from .catalog import HOSTED_BASE, NimSpec, get as get_spec

log = logging.getLogger("bionemo_web.nims")

DEFAULT_TIMEOUT = 120
HEALTH_TIMEOUT = 4
DISCOVERY_TIMEOUT = 6


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------


@dataclass
class ProbeResult:
    """Outcome of a connectivity / readiness probe."""

    backend: str          # "local" | "hosted"
    base_url: str
    reachable: bool = False
    ready: bool = False
    status: Optional[int] = None
    latency_ms: Optional[float] = None
    model_versions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    message: str = ""
    detail: Dict[str, Any] = field(default_factory=dict)

    @property
    def state(self) -> str:
        if not self.reachable:
            return "unreachable"
        if self.ready:
            return "ready"
        return "reachable, not ready"


@dataclass
class NimCallResult:
    """A single model invocation, successful or not."""

    ok: bool
    backend: str
    url: str
    spec_key: str
    latency_ms: float
    status: Optional[int] = None
    canonical: Dict[str, Any] = field(default_factory=dict)
    raw: Any = None
    error: str = ""
    warnings: List[str] = field(default_factory=list)
    request_payload: Dict[str, Any] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    attempts: int = 1

    def curl(self) -> str:
        """A copy-pasteable curl command for debugging the same call by hand."""
        body = json.dumps(self.request_payload, indent=2)
        headers = "".join(f" \\\n  -H '{k}: {v}'" for k, v in self.headers.items())
        return f"curl -X POST '{self.url}'{headers} \\\n  -d '{body}'"


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------


def _first_list(d: Dict[str, Any], keys: Sequence[str]) -> Optional[List]:
    for k in keys:
        v = d.get(k)
        if isinstance(v, list) and v:
            return v
    return None


def normalise(kind: str, raw: Any) -> Dict[str, Any]:
    """Map a NIM response onto the console's canonical result shapes.

    Canonical shapes::

        fold      {"pdbs": [str], "plddt": [float]|None, "ptm": float|None, "pae": ...}
        embeddings{"embeddings": [[float]], "mean": [float]|None}
        molecules {"molecules": [{"smiles": str, "score": float|None, "extra": {}}]}
        dock      {"ligand_positions": [[x,y,z]], "position_confidence": [float]}
        design    {"sequences": [{"sequence": str, "score": float|None}], "pdb": str|None}
        genomics  {"sequence": str, "probs": [float]|None, "scores": {...}}
        segmentation {"mask": ..., "meta": {...}}
    """
    if raw is None:
        return {}
    if isinstance(raw, str):
        # Some endpoints answer with bare text (a PDB or a FASTA).
        text = raw.strip()
        if text.startswith("HEADER") or "ATOM" in text[:2000]:
            return {"pdbs": [text]}
        if text.startswith(">"):
            return {"sequences": [{"sequence": _clean_fasta(text), "score": None}]}
        return {"text": text}
    if not isinstance(raw, dict):
        return {"raw": raw}

    if kind == "fold":
        pdbs = _first_list(raw, ["pdbs", "pdb", "structures", "structure", "predictions"]) or []
        if pdbs and isinstance(pdbs[0], dict):
            pdbs = [p.get("pdb") or p.get("structure") or "" for p in pdbs]
        plddt = raw.get("plddt") or raw.get("confidence") or raw.get("mean_plddt")
        out: Dict[str, Any] = {
            "pdbs": [p for p in pdbs if isinstance(p, str) and p.strip()],
            "plddt": plddt if isinstance(plddt, list) else None,
            "mean_plddt": (sum(plddt) / len(plddt)) if isinstance(plddt, list) and plddt else plddt,
            "ptm": raw.get("ptm") or raw.get("pTM"),
            "pae": raw.get("pae") or raw.get("predicted_aligned_error"),
            "affinity": raw.get("affinity_pred_value") or raw.get("affinity"),
        }
        return out

    if kind == "embeddings":
        emb = _first_list(raw, ["embeddings", "embedding", "data", "features"])
        mean = raw.get("mean_embeddings") or raw.get("mean_embedding")
        if emb is None and isinstance(raw.get("embedding"), list):
            emb = [raw["embedding"]]
        return {"embeddings": emb or [], "mean": mean}

    if kind == "molecules":
        rows: List[Dict[str, Any]] = []
        for key in ("molecules", "generated", "results", "samples"):
            block = raw.get(key)
            if isinstance(block, list):
                for item in block:
                    if isinstance(item, str):
                        rows.append({"smiles": item, "score": None, "extra": {}})
                    elif isinstance(item, dict):
                        smi = (
                            item.get("smiles")
                            or item.get("sample")
                            or item.get("smi")
                            or item.get("molecule")
                            or ""
                        )
                        score = item.get("score", item.get("qed", item.get("property")))
                        extra = {k: v for k, v in item.items()
                                 if k not in ("smiles", "sample", "smi", "molecule", "score", "qed")}
                        rows.append({"smiles": smi, "score": score, "extra": extra})
                break
        if not rows and isinstance(raw.get("smiles"), str):
            rows = [{"smiles": raw["smiles"], "score": raw.get("score"), "extra": {}}]
        return {"molecules": rows, "status": raw.get("status")}

    if kind == "dock":
        positions = raw.get("ligand_positions") or raw.get("positions") or raw.get("poses")
        conf = raw.get("position_confidence") or raw.get("confidence") or raw.get("scores")
        if isinstance(positions, list) and positions and isinstance(positions[0], dict):
            conf = [p.get("confidence") for p in positions]
            positions = [p.get("positions") or p.get("ligand_positions") for p in positions]
        return {
            "ligand_positions": positions or [],
            "position_confidence": conf or [],
            "ligand": raw.get("ligand"),
            "protein": raw.get("protein"),
        }

    if kind == "design":
        rows = []
        for key in ("sequences", "results", "designs"):
            block = raw.get(key)
            if isinstance(block, list):
                for item in block:
                    if isinstance(item, str):
                        rows.append({"sequence": item, "score": None})
                    elif isinstance(item, dict):
                        rows.append(
                            {
                                "sequence": item.get("sequence") or item.get("seq") or "",
                                "score": item.get("score", item.get("global_score", item.get("ll"))),
                                "local_scores": item.get("local_scores") or item.get("scores"),
                                "recovery": item.get("recovery"),
                            }
                        )
                break
        if not rows and isinstance(raw.get("sequence"), str):
            rows = [{"sequence": raw["sequence"], "score": raw.get("score")}]
        return {"sequences": rows, "pdb": raw.get("pdb") or raw.get("structure")}

    if kind == "genomics":
        seq = raw.get("sequence") or raw.get("generated_sequence") or raw.get("output") or ""
        return {
            "sequence": seq,
            "probs": raw.get("probs") or raw.get("sampled_probs"),
            "scores": {k: v for k, v in raw.items() if isinstance(v, (int, float))},
        }

    if kind == "segmentation":
        return {"mask": raw.get("mask") or raw.get("segmentation"), "meta": raw}

    return {"raw": raw}


def _clean_fasta(text: str) -> str:
    return "".join(
        line.strip() for line in text.splitlines() if line and not line.startswith(">")
    )


# --------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------


class NimClient:
    """Talks to local and hosted NIM endpoints, logging every call it makes."""

    def __init__(
        self,
        *,
        local_endpoints: Optional[Dict[str, str]] = None,
        path_overrides: Optional[Dict[str, str]] = None,
        hosted_api_key: str = "",
        hosted_base: str = "",
        timeout: int = DEFAULT_TIMEOUT,
        verify_ssl: bool = True,
        on_call: Optional[callable] = None,
    ) -> None:
        self.local_endpoints = dict(local_endpoints or {})
        self.path_overrides = dict(path_overrides or {})
        self.hosted_api_key = hosted_api_key
        if hosted_base:
            global HOSTED_BASE  # module-level default used by url_for()
            HOSTED_BASE = hosted_base.rstrip("/")
        self.hosted_base = hosted_base.rstrip("/") if hosted_base else ""
        #: Shown as the placeholder/default in the Administration page.
        self.default_local = (
            self.local_endpoints.get("esmfold") or self.local_endpoints.get("*") or "http://localhost:8000"
        )
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        self.on_call = on_call  # callback(NimCallResult) for the inspector/benchmarks
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "dr-titan-bionemo-console/1.0"})
        self._discovered: Dict[Tuple[str, str], str] = {}

    # -- addressing ----------------------------------------------------
    def base_url(self, spec: NimSpec, backend: str) -> str:
        if backend == "hosted":
            return HOSTED_BASE
        url = self.local_endpoints.get(spec.key) or self.local_endpoints.get("*") or ""
        return url.rstrip("/")

    def path_for(self, spec: NimSpec, backend: str) -> str:
        override = self.path_overrides.get(spec.key)
        if override:
            return override if override.startswith("/") else "/" + override
        discovered = self._discovered.get((spec.key, backend))
        if discovered:
            return discovered
        if backend == "hosted":
            return spec.hosted_path
        return spec.local_path or spec.hosted_path

    def url_for(self, spec: NimSpec, backend: str) -> str:
        base = self.base_url(spec, backend)
        path = self.path_for(spec, backend)
        if not base:
            return ""
        if base.endswith("/v1") and path.startswith("/v1/"):
            path = path[3:]
        return base + path

    def headers_for(self, backend: str) -> Dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if backend == "hosted" and self.hosted_api_key:
            headers["Authorization"] = f"Bearer {self.hosted_api_key}"
        return headers

    # -- probing -------------------------------------------------------
    def probe(self, spec: NimSpec, backend: str) -> ProbeResult:
        base = self.base_url(spec, backend)
        result = ProbeResult(backend=backend, base_url=base)
        if not base:
            result.message = "No endpoint configured."
            return result
        if backend == "hosted" and not self.hosted_api_key:
            result.reachable = False
            result.message = "No NVIDIA API key configured - hosted NIMs need one (NGC_API_KEY)."
            return result

        started = time.time()
        session = self._session
        try:
            resp = session.get(base + "/v1/health/ready", timeout=HEALTH_TIMEOUT,
                               verify=self.verify_ssl)
            result.status = resp.status_code
            result.latency_ms = round((time.time() - started) * 1000, 1)
            result.reachable = True
            result.ready = resp.status_code == 200
            body = _safe_json(resp)
            if isinstance(body, dict):
                result.message = str(body.get("status") or body.get("message") or resp.reason)
        except requests.exceptions.SSLError as exc:
            result.message = f"TLS error: {exc}"
            return result
        except requests.exceptions.ConnectionError as exc:
            result.message = f"Connection refused/unreachable: {_short(exc)}"
            return result
        except requests.exceptions.Timeout:
            result.message = f"No response within {HEALTH_TIMEOUT}s."
            return result
        except Exception as exc:  # pragma: no cover - defensive
            result.message = _short(exc)
            return result

        # Optional extras; absence is normal for older containers.
        for path, key in (("/v1/models", "model_versions"), ("/v1/metadata", "metadata"),
                          ("/v1/version", "metadata")):
            try:
                r = session.get(base + path, timeout=HEALTH_TIMEOUT, verify=self.verify_ssl,
                                headers=self.headers_for(backend))
                if r.status_code != 200:
                    continue
                data = _safe_json(r)
                if key == "model_versions" and isinstance(data, dict):
                    items = data.get("data") or data.get("models") or []
                    for item in items if isinstance(items, list) else []:
                        if isinstance(item, dict) and item.get("id"):
                            result.model_versions.append(str(item["id"]))
                        elif isinstance(item, str):
                            result.model_versions.append(item)
                elif isinstance(data, dict):
                    result.metadata.update(
                        {k: v for k, v in data.items() if isinstance(v, (str, int, float, list))}
                    )
            except Exception:
                continue
        if not result.message:
            result.message = "ready" if result.ready else "no readiness signal"
        return result

    def list_operations(self, base_url: str, backend: str = "local") -> List[Dict[str, str]]:
        """Read ``/openapi.json`` and return every POST operation it advertises."""
        if not base_url:
            return []
        try:
            resp = self._session.get(
                base_url.rstrip("/") + "/openapi.json",
                timeout=DISCOVERY_TIMEOUT,
                verify=self.verify_ssl,
                headers=self.headers_for(backend),
            )
            if resp.status_code != 200:
                return []
            spec = resp.json()
        except Exception:
            return []
        ops: List[Dict[str, str]] = []
        for path, methods in (spec.get("paths") or {}).items():
            if not isinstance(methods, dict):
                continue
            for method, body in methods.items():
                if method.lower() != "post":
                    continue
                summary = ""
                operation_id = ""
                if isinstance(body, dict):
                    summary = str(body.get("summary") or "")
                    operation_id = str(body.get("operationId") or "")
                ops.append({"path": path, "summary": summary, "operation_id": operation_id})
        return ops

    def discover_path(self, spec: NimSpec, base_url: str, backend: str = "local") -> Optional[str]:
        """Find this model's route in the deployment's own OpenAPI document.

        Matching is by keyword score over path + operationId + summary, so it
        works even when a container renames its prefix.
        """
        ops = self.list_operations(base_url, backend)
        if not ops:
            return None
        keywords = [
            spec.key.replace("_", ""),
            spec.key.replace("_", "-"),
            spec.name.lower().replace(" ", ""),
            spec.name.lower().replace(" ", "-"),
            spec.name.lower().replace(" (", "-").replace(")", "").replace(" ", ""),
        ]
        keywords = [k for k in keywords if len(k) > 2]
        intent_words = {
            "fold": ["fold", "structure", "predict"],
            "embeddings": ["embed", "embedding"],
            "molecules": ["generate", "sampl", "molecule"],
            "dock": ["dock", "docking", "generate"],
            "design": ["generate", "mpnn", "design", "predict"],
            "genomics": ["generate", "forward", "score"],
        }.get(spec.kind, ["generate"])

        best: Tuple[float, Optional[str]] = (0.0, None)
        for op in ops:
            haystack = " ".join([op["path"], op.get("summary", ""), op.get("operation_id", "")]).lower()
            score = 0.0
            for kw in keywords:
                if kw in haystack:
                    score += 3.0
            for word in intent_words:
                if word in haystack:
                    score += 1.0
            if op["path"].rstrip("/").endswith(("health/ready", "/models", "/metadata", "/version")):
                continue
            if score > best[0]:
                best = (score, op["path"])
        if best[1] and best[0] >= 3.0:
            self._discovered[(spec.key, backend)] = best[1]
            log.info("discovered route for %s: %s", spec.key, best[1])
            return best[1]
        return None

    # -- calling -------------------------------------------------------
    def call(
        self,
        spec: NimSpec,
        payload: Dict[str, Any],
        *,
        backend: str = "local",
        timeout: Optional[int] = None,
        retries: int = 2,
        url_override: str = "",
    ) -> NimCallResult:
        """POST a payload to a NIM and normalise the response."""
        url = url_override or self.url_for(spec, backend)
        headers = self.headers_for(backend)
        started = time.time()
        attempts = 0
        last_error = ""
        last_status: Optional[int] = None
        raw: Any = None

        if not url:
            return NimCallResult(
                ok=False, backend=backend, url="", spec_key=spec.key, latency_ms=0.0,
                error="No endpoint configured for this model.", request_payload=payload,
                headers=headers,
            )

        for attempt in range(retries + 1):
            attempts += 1
            try:
                resp = self._session.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=timeout or self.timeout,
                    verify=self.verify_ssl,
                )
                last_status = resp.status_code
                if resp.status_code >= 500 and attempt < retries:
                    last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                    time.sleep(1.5 * (attempt + 1))
                    continue
                if resp.status_code >= 400:
                    last_error = f"HTTP {resp.status_code}: {resp.text[:400]}"
                    raw = _safe_json(resp, default=resp.text[:2000])
                    break
                raw = _safe_json(resp, default=resp.text)
                break
            except requests.exceptions.Timeout:
                last_error = f"Timed out after {timeout or self.timeout}s."
                if attempt < retries:
                    continue
            except requests.exceptions.SSLError as exc:
                last_error = f"TLS error: {_short(exc)}"
                break
            except requests.exceptions.ConnectionError as exc:
                last_error = f"Connection error: {_short(exc)}"
                if attempt < retries:
                    time.sleep(0.7)
                    continue
                break
            except Exception as exc:  # pragma: no cover
                last_error = _short(exc)
                break

        latency = round((time.time() - started) * 1000, 1)
        ok = raw is not None and (last_status is None or last_status < 400)
        canonical = normalise(spec.kind, raw) if ok else {}
        warnings: List[str] = []
        if ok and not canonical:
            warnings.append("The service answered but no recognised result fields were present.")
        if ok and url.endswith("/") is False and backend == "hosted" and not self.hosted_api_key:
            warnings.append("No API key set: the hosted endpoint may reject this call.")

        result = NimCallResult(
            ok=bool(ok),
            backend=backend,
            url=url,
            spec_key=spec.key,
            latency_ms=latency,
            status=last_status,
            canonical=canonical,
            raw=raw,
            error="" if ok else last_error,
            warnings=warnings,
            request_payload=payload,
            headers={k: (v if k != "Authorization" else "Bearer ***") for k, v in headers.items()},
            attempts=attempts,
        )
        if self.on_call:
            try:
                self.on_call(result)
            except Exception:  # pragma: no cover - logging must never break a run
                pass
        return result


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _safe_json(resp: requests.Response, default: Any = None) -> Any:
    try:
        return resp.json()
    except Exception:
        return default if default is not None else resp.text


def _short(exc: Exception, limit: int = 160) -> str:
    text = str(exc).replace("\n", " ")
    return text[:limit] + ("..." if len(text) > limit else "")
