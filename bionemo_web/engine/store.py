"""SQLite persistence: conversations, runs, artifacts, call logs and settings.

Everything the console knows is durable.  A run record carries the full
provenance manifest (backend, URL, parameters, warnings, timing), which is what
lets the app answer "which model produced this file, with which settings, and
was it real or simulated?" months later.

The database lives in ``.runtime/console.db`` (git-ignored) and is created on
first use.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

RUNTIME_DIR = Path(os.environ.get("BIONEMO_CONSOLE_HOME", Path(__file__).resolve().parents[2] / ".runtime"))
DB_PATH = RUNTIME_DIR / "console.db"
ARTIFACT_DIR = RUNTIME_DIR / "artifacts"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    pinned INTEGER DEFAULT 0,
    tags TEXT DEFAULT '',
    meta TEXT DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT DEFAULT '',
    created_at REAL NOT NULL,
    run_id TEXT,
    tool TEXT,
    payload TEXT DEFAULT '{}',
    feedback INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, created_at);
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    conversation_id TEXT,
    message_id TEXT,
    model_key TEXT NOT NULL,
    backend TEXT NOT NULL,
    ok INTEGER NOT NULL,
    simulated INTEGER DEFAULT 0,
    created_at REAL NOT NULL,
    latency_ms REAL DEFAULT 0,
    summary TEXT DEFAULT '',
    canonical TEXT DEFAULT '{}',
    meta TEXT DEFAULT '{}',
    request TEXT DEFAULT '{}',
    warnings TEXT DEFAULT '[]',
    error TEXT DEFAULT '',
    provenance TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_runs_created ON runs(created_at DESC);
CREATE TABLE IF NOT EXISTS call_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL NOT NULL,
    model_key TEXT,
    backend TEXT,
    ok INTEGER,
    status INTEGER,
    latency_ms REAL,
    attempts INTEGER,
    url TEXT,
    error TEXT DEFAULT '',
    request TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_calllog_created ON call_log(created_at DESC);
CREATE TABLE IF NOT EXISTS artifacts (
    id TEXT PRIMARY KEY,
    run_id TEXT,
    conversation_id TEXT,
    kind TEXT,
    name TEXT,
    path TEXT,
    size INTEGER DEFAULT 0,
    created_at REAL NOT NULL,
    meta TEXT DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at REAL
);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    conversation_id TEXT,
    kind TEXT,
    title TEXT,
    status TEXT,
    progress REAL DEFAULT 0,
    message TEXT DEFAULT '',
    created_at REAL,
    started_at REAL,
    finished_at REAL,
    params TEXT DEFAULT '{}',
    result_summary TEXT DEFAULT '{}',
    run_id TEXT,
    error TEXT DEFAULT ''
);
"""


def _now() -> float:
    return time.time()


def _json(value: Any) -> str:
    return json.dumps(value, default=_fallback, ensure_ascii=False)


def _fallback(obj: Any) -> Any:
    """Make odd objects JSON-safe instead of exploding mid-write."""
    if hasattr(obj, "to_dict"):
        try:
            return obj.to_dict()
        except Exception:
            pass
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, (set, tuple)):
        return list(obj)
    return str(obj)


@dataclass
class JobRecord:
    id: str
    kind: str
    title: str
    status: str = "queued"      # queued|running|done|failed|cancelled
    progress: float = 0.0
    message: str = ""
    created_at: float = field(default_factory=_now)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    params: Dict[str, Any] = field(default_factory=dict)
    result_summary: Dict[str, Any] = field(default_factory=dict)
    run_id: str = ""
    error: str = ""
    conversation_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "title": self.title,
            "status": self.status,
            "progress": round(self.progress, 3),
            "message": self.message,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "params": self.params,
            "result_summary": self.result_summary,
            "run_id": self.run_id,
            "error": self.error,
            "conversation_id": self.conversation_id,
            "duration_s": round((self.finished_at or _now()) - (self.started_at or self.created_at), 2),
        }

    @property
    def is_active(self) -> bool:
        return self.status in ("queued", "running")


class Store:
    """Thread-safe SQLite wrapper."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path or DB_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            # Databases written before artifact registration was made idempotent
            # can hold several rows for one path; keep the newest row per path so
            # an upgrade does not leave a History page full of duplicates.
            self._conn.execute(
                """DELETE FROM artifacts WHERE path <> '' AND rowid NOT IN (
                       SELECT MAX(rowid) FROM artifacts WHERE path <> '' GROUP BY path
                   )"""
            )
            self._conn.commit()

    # -- low level -----------------------------------------------------
    def execute(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return cur

    def query(self, sql: str, params: Sequence[Any] = ()) -> List[sqlite3.Row]:
        with self._lock:
            return list(self._conn.execute(sql, params))

    def query_one(self, sql: str, params: Sequence[Any] = ()) -> Optional[sqlite3.Row]:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    # -- conversations -------------------------------------------------
    def create_conversation(self, title: str = "New conversation", *, tags: str = "") -> str:
        cid = uuid.uuid4().hex[:12]
        now = _now()
        self.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at, pinned, tags) VALUES (?,?,?,?,0,?)",
            (cid, title, now, now, tags),
        )
        return cid

    def list_conversations(self, limit: int = 100, search: str = "") -> List[Dict[str, Any]]:
        if search:
            rows = self.query(
                """SELECT c.*, (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS n_messages
                   FROM conversations c WHERE c.title LIKE ? OR c.tags LIKE ?
                   ORDER BY c.pinned DESC, c.updated_at DESC LIMIT ?""",
                (f"%{search}%", f"%{search}%", limit),
            )
        else:
            rows = self.query(
                """SELECT c.*, (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS n_messages
                   FROM conversations c ORDER BY c.pinned DESC, c.updated_at DESC LIMIT ?""",
                (limit,),
            )
        return [dict(r) for r in rows]

    def rename_conversation(self, cid: str, title: str) -> None:
        self.execute("UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?", (title, _now(), cid))

    def set_pinned(self, cid: str, pinned: bool) -> None:
        self.execute("UPDATE conversations SET pinned = ?, updated_at = ? WHERE id = ?",
                     (1 if pinned else 0, _now(), cid))

    def delete_conversation(self, cid: str) -> None:
        self.execute("DELETE FROM messages WHERE conversation_id = ?", (cid,))
        self.execute("DELETE FROM conversations WHERE id = ?", (cid,))

    def touch_conversation(self, cid: str) -> None:
        self.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (_now(), cid))

    def auto_title(self, cid: str) -> None:
        """Title an untitled conversation from its first user message."""
        row = self.query_one("SELECT title FROM conversations WHERE id = ?", (cid,))
        if row and row["title"] not in ("", "New conversation"):
            return
        first = self.query_one(
            "SELECT content FROM messages WHERE conversation_id = ? AND role = 'user' "
            "ORDER BY created_at LIMIT 1",
            (cid,),
        )
        if first and first["content"]:
            title = " ".join(str(first["content"]).split())[:70]
            self.rename_conversation(cid, title or "New conversation")

    # -- messages ------------------------------------------------------
    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        *,
        run_id: str = "",
        tool: str = "",
        payload: Optional[Dict[str, Any]] = None,
    ) -> str:
        mid = uuid.uuid4().hex[:12]
        self.execute(
            """INSERT INTO messages (id, conversation_id, role, content, created_at, run_id, tool, payload)
               VALUES (?,?,?,?,?,?,?,?)""",
            (mid, conversation_id, role, content, _now(), run_id, tool, _json(payload or {})),
        )
        self.touch_conversation(conversation_id)
        return mid

    def messages(self, conversation_id: str, limit: int = 500) -> List[Dict[str, Any]]:
        rows = self.query(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at LIMIT ?",
            (conversation_id, limit),
        )
        out = []
        for row in rows:
            d = dict(row)
            try:
                d["payload"] = json.loads(d.get("payload") or "{}")
            except Exception:
                d["payload"] = {}
            d["runs"] = self.runs_for_message(d["id"])
            out.append(d)
        return out

    def set_feedback(self, message_id: str, value: int) -> None:
        self.execute("UPDATE messages SET feedback = ? WHERE id = ?", (value, message_id))

    # -- runs ----------------------------------------------------------
    def record_run(
        self,
        *,
        model_key: str,
        backend: str,
        ok: bool,
        simulated: bool,
        latency_ms: float,
        summary: str = "",
        canonical: Optional[Dict[str, Any]] = None,
        meta: Optional[Dict[str, Any]] = None,
        request: Optional[Dict[str, Any]] = None,
        warnings: Optional[Iterable[str]] = None,
        error: str = "",
        provenance: Optional[Dict[str, Any]] = None,
        conversation_id: str = "",
        message_id: str = "",
    ) -> str:
        rid = uuid.uuid4().hex[:12]
        # Keep persisted payloads small: structures and poses stay in memory,
        # only their summaries are written.
        safe_meta = {k: v for k, v in (meta or {}).items() if k not in ("structure", "poses", "rows", "designs")}
        self.execute(
            """INSERT INTO runs (id, conversation_id, message_id, model_key, backend, ok, simulated,
                                 created_at, latency_ms, summary, canonical, meta, request, warnings,
                                 error, provenance)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                rid, conversation_id, message_id, model_key, backend, 1 if ok else 0,
                1 if simulated else 0, _now(), latency_ms, summary,
                _json(canonical or {}), _json(safe_meta), _json(request or {}),
                _json(list(warnings or [])), error, _json(provenance or {}),
            ),
        )
        return rid

    def runs_for_message(self, message_id: str) -> List[Dict[str, Any]]:
        rows = self.query("SELECT * FROM runs WHERE message_id = ? ORDER BY created_at", (message_id,))
        return [_run_row(r) for r in rows]

    def runs(self, limit: int = 200, conversation_id: str = "") -> List[Dict[str, Any]]:
        if conversation_id:
            rows = self.query(
                "SELECT * FROM runs WHERE conversation_id = ? ORDER BY created_at DESC LIMIT ?",
                (conversation_id, limit),
            )
        else:
            rows = self.query("SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,))
        return [_run_row(r) for r in rows]

    def run(self, run_id: str) -> Optional[Dict[str, Any]]:
        row = self.query_one("SELECT * FROM runs WHERE id = ?", (run_id,))
        return _run_row(row) if row else None

    # -- call log & benchmarks ----------------------------------------
    def log_call(
        self, *, model_key: str, backend: str, ok: bool, status: Optional[int],
        latency_ms: float, attempts: int, url: str, error: str = "",
        request: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.execute(
            """INSERT INTO call_log (created_at, model_key, backend, ok, status, latency_ms, attempts,
                                     url, error, request)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (_now(), model_key, backend, 1 if ok else 0, status, latency_ms, attempts, url,
             error[:500], _json(request or {})),
        )

    def call_log(self, limit: int = 200) -> List[Dict[str, Any]]:
        rows = self.query("SELECT * FROM call_log ORDER BY created_at DESC LIMIT ?", (limit,))
        return [dict(r) for r in rows]

    def benchmark_summary(self) -> List[Dict[str, Any]]:
        """Latency/throughput per model+backend, for the performance panel."""
        rows = self.query(
            """SELECT model_key, backend, COUNT(*) AS calls,
                      SUM(ok) AS successes,
                      AVG(latency_ms) AS avg_latency,
                      MIN(latency_ms) AS min_latency,
                      MAX(latency_ms) AS max_latency
               FROM call_log GROUP BY model_key, backend ORDER BY calls DESC"""
        )
        out = []
        for r in rows:
            calls = r["calls"] or 1
            out.append(
                {
                    "model_key": r["model_key"],
                    "backend": r["backend"],
                    "calls": r["calls"],
                    "success_rate": round((r["successes"] or 0) / calls, 3),
                    "avg_latency_ms": round(r["avg_latency"] or 0, 1),
                    "min_latency_ms": round(r["min_latency"] or 0, 1),
                    "max_latency_ms": round(r["max_latency"] or 0, 1),
                }
            )
        return out

    # -- artifacts -----------------------------------------------------
    def add_artifact(
        self, *, kind: str, name: str, path: str, run_id: str = "", conversation_id: str = "",
        meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Register a written file, keyed by its path.

        Registration is deliberately idempotent: Streamlit re-runs the script on
        every interaction, so the same finished run is often re-rendered several
        times. A plain INSERT would list the same file once per re-render, and the
        History page would show each artifact up to four times. Re-registering a
        known path therefore refreshes its metadata and returns the existing id.
        """
        aid = uuid.uuid4().hex[:12]
        try:
            size = Path(path).stat().st_size
        except Exception:
            size = 0
        path_str = str(path)
        existing = (
            self.query_one("SELECT id FROM artifacts WHERE path = ?", (path_str,))
            if path_str
            else None
        )
        if existing is not None:
            self.execute(
                """UPDATE artifacts SET kind = ?, name = ?, size = ?, meta = ?,
                   run_id = COALESCE(NULLIF(?, ''), run_id),
                   conversation_id = COALESCE(NULLIF(?, ''), conversation_id)
                   WHERE id = ?""",
                (kind, name, size, _json(meta or {}), run_id, conversation_id, existing["id"]),
            )
            return existing["id"]
        self.execute(
            """INSERT INTO artifacts (id, run_id, conversation_id, kind, name, path, size, created_at, meta)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (aid, run_id, conversation_id, kind, name, path_str, size, _now(), _json(meta or {})),
        )
        return aid

    def artifacts(self, limit: int = 200, conversation_id: str = "") -> List[Dict[str, Any]]:
        if conversation_id:
            rows = self.query(
                "SELECT * FROM artifacts WHERE conversation_id = ? ORDER BY created_at DESC LIMIT ?",
                (conversation_id, limit),
            )
        else:
            rows = self.query("SELECT * FROM artifacts ORDER BY created_at DESC LIMIT ?", (limit,))
        return [dict(r) for r in rows]

    # -- settings ------------------------------------------------------
    def get_settings(self) -> Dict[str, Any]:
        rows = self.query("SELECT key, value FROM settings")
        out: Dict[str, Any] = {}
        for row in rows:
            try:
                out[row["key"]] = json.loads(row["value"])
            except Exception:
                out[row["key"]] = row["value"]
        return out

    def set_setting(self, key: str, value: Any) -> None:
        self.execute(
            "INSERT INTO settings (key, value, updated_at) VALUES (?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
            (key, _json(value), _now()),
        )

    def set_settings(self, values: Dict[str, Any]) -> None:
        for key, value in values.items():
            self.set_setting(key, value)

    # -- jobs ----------------------------------------------------------
    def upsert_job(self, job: JobRecord) -> None:
        self.execute(
            """INSERT INTO jobs (id, conversation_id, kind, title, status, progress, message, created_at,
                                 started_at, finished_at, params, result_summary, run_id, error)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET status=excluded.status, progress=excluded.progress,
                 message=excluded.message, started_at=excluded.started_at, finished_at=excluded.finished_at,
                 result_summary=excluded.result_summary, run_id=excluded.run_id, error=excluded.error""",
            (
                job.id, job.conversation_id, job.kind, job.title, job.status, job.progress,
                job.message, job.created_at, job.started_at, job.finished_at,
                _json(job.params), _json(job.result_summary), job.run_id, job.error,
            ),
        )

    def jobs(self, limit: int = 100, conversation_id: str = "") -> List[Dict[str, Any]]:
        if conversation_id:
            rows = self.query(
                "SELECT * FROM jobs WHERE conversation_id = ? ORDER BY created_at DESC LIMIT ?",
                (conversation_id, limit),
            )
        else:
            rows = self.query("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,))
        return [dict(r) for r in rows]

    # -- reporting -----------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        counts = {}
        for table in ("conversations", "messages", "runs", "artifacts", "jobs", "call_log"):
            row = self.query_one(f"SELECT COUNT(*) AS n FROM {table}")
            counts[table] = row["n"] if row else 0
        row = self.query_one("SELECT SUM(simulated) AS sim, COUNT(*) AS n FROM runs")
        sim = (row["sim"] or 0) if row else 0
        total = (row["n"] or 0) if row else 0
        return {
            "counts": counts,
            "simulated_runs": int(sim),
            "real_runs": int(total - sim),
            "db_path": str(self.path),
            "db_size_kb": round(self.path.stat().st_size / 1024, 1) if self.path.exists() else 0.0,
        }

    def export_conversation(self, conversation_id: str, *, include_runs: bool = True) -> Dict[str, Any]:
        """JSON-serialisable export used by the Markdown/JSON download buttons."""
        conv = self.query_one("SELECT * FROM conversations WHERE id = ?", (conversation_id,))
        if not conv:
            return {}
        data = {
            "conversation": dict(conv),
            "messages": self.messages(conversation_id),
            "exported_at": _now(),
            "app": "Dr-Titan BioNeMo Web Console",
        }
        if include_runs:
            data["runs"] = self.runs(limit=1000, conversation_id=conversation_id)
        return data


def _run_row(row: sqlite3.Row) -> Dict[str, Any]:
    d = dict(row)
    for key in ("canonical", "meta", "request", "warnings", "provenance"):
        try:
            d[key] = json.loads(d.get(key) or "{}")
        except Exception:
            d[key] = {} if key not in ("warnings",) else []
    d["ok"] = bool(d.get("ok"))
    d["simulated"] = bool(d.get("simulated"))
    return d


# --------------------------------------------------------------------------
# Process-wide singleton (survives Streamlit reruns)
# --------------------------------------------------------------------------

_STORE: Optional[Store] = None
_STORE_LOCK = threading.Lock()


def get_store() -> Store:
    global _STORE
    if _STORE is None:
        with _STORE_LOCK:
            if _STORE is None:
                _STORE = Store()
    return _STORE
