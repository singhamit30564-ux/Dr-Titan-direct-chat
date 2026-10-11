"""Background job queue.

Folding a 300-residue protein, docking 40 poses or running a virtual-screening
pipeline takes seconds to minutes.  Streamlit's execution model reruns the whole
script on every interaction, so that work cannot live inside a widget callback:
it needs a queue that outlives the rerun.

This module is a process-wide singleton thread pool.  Jobs are persisted to
SQLite as they move through ``queued -> running -> done|failed|cancelled``, and
the full result object is kept in memory for the session so the UI can render
rich objects (structures, poses, molecule tables) without re-serialising them.

Cancellation is cooperative: a job checks :meth:`JobContext.check_cancelled`
between steps, which is why long pipelines are written as a sequence of small
steps rather than one blocking call.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .store import JobRecord, get_store

log = logging.getLogger("bionemo_web.jobs")

MAX_WORKERS = 2
MAX_HISTORY = 200


class JobCancelled(Exception):
    """Raised inside a job when the user cancels it."""


@dataclass
class JobContext:
    """Handed to job functions so they can report progress and honour cancel."""

    job_id: str
    _cancel: threading.Event
    _record: JobRecord
    _store: Any = None

    def progress(self, value: float, message: str = "") -> None:
        self._record.progress = max(0.0, min(1.0, float(value)))
        if message:
            self._record.message = message
        self._persist()

    def log(self, message: str) -> None:
        self._record.message = message
        self._persist()

    def check_cancelled(self) -> None:
        if self._cancel.is_set():
            raise JobCancelled("Cancelled by user.")

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def _persist(self) -> None:
        if self._store is not None:
            try:
                self._store.upsert_job(self._record)
            except Exception:  # pragma: no cover
                pass


@dataclass
class Job:
    record: JobRecord
    func: Optional[Callable[[JobContext], Any]] = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    context: Optional[JobContext] = None
    logs: List[str] = field(default_factory=list)
    future: Any = None

    def to_dict(self) -> Dict[str, Any]:
        d = self.record.to_dict()
        d["logs"] = self.logs[-40:]
        return d


class JobQueue:
    """A small, dependency-free background queue."""

    def __init__(self, max_workers: int = MAX_WORKERS) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="bionemo-job")
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.RLock()
        self._store = get_store()

    # -- submission ----------------------------------------------------
    def submit(
        self,
        kind: str,
        title: str,
        func: Callable[[JobContext], Any],
        *,
        params: Optional[Dict[str, Any]] = None,
        conversation_id: str = "",
    ) -> str:
        job_id = uuid.uuid4().hex[:12]
        record = JobRecord(
            id=job_id, kind=kind, title=title, status="queued", params=params or {},
            conversation_id=conversation_id,
        )
        cancel = threading.Event()
        job = Job(record=record, func=func, cancel_event=cancel)
        with self._lock:
            self._jobs[job_id] = job
            self._trim()
        self._store.upsert_job(record)
        job.future = self._executor.submit(self._run, job_id)
        return job_id

    def _run(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None or job.func is None:
            return
        record = job.record
        record.status = "running"
        record.started_at = time.time()
        record.message = "starting"
        ctx = JobContext(job_id=job_id, _cancel=job.cancel_event, _record=record, _store=self._store)
        job.context = ctx
        ctx._persist()
        try:
            result = job.func(ctx)
            if job.cancel_event.is_set():
                record.status = "cancelled"
                record.message = "cancelled"
            else:
                job.result = result
                record.status = "done"
                record.progress = 1.0
                record.message = record.message or "complete"
                summary = getattr(result, "result_summary", None)
                if callable(summary):
                    record.result_summary = summary()
                elif isinstance(result, dict):
                    record.result_summary = {
                        k: v for k, v in result.items()
                        if isinstance(v, (str, int, float, bool, list, dict, type(None)))
                    }
        except JobCancelled:
            record.status = "cancelled"
            record.message = "Cancelled by user."
        except Exception as exc:  # noqa: BLE001 - a job failure must not kill the app
            log.exception("job %s failed", job_id)
            record.status = "failed"
            record.error = f"{type(exc).__name__}: {exc}"
            record.message = "failed"
        finally:
            record.finished_at = time.time()
            ctx._persist()

    # -- inspection ----------------------------------------------------
    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def status(self, job_id: str) -> Optional[Dict[str, Any]]:
        job = self.get(job_id)
        return job.to_dict() if job else None

    def list(self, *, active_only: bool = False, conversation_id: str = "") -> List[Dict[str, Any]]:
        with self._lock:
            jobs = list(self._jobs.values())
        jobs.sort(key=lambda j: -j.record.created_at)
        out = []
        for job in jobs:
            if conversation_id and job.record.conversation_id != conversation_id:
                continue
            if active_only and not job.record.is_active:
                continue
            out.append(job.to_dict())
        return out

    def active_count(self) -> int:
        with self._lock:
            return sum(1 for j in self._jobs.values() if j.record.is_active)

    def aggregate_progress(self) -> float:
        """Overall completion of everything currently queued/running (0-1)."""
        with self._lock:
            active = [j for j in self._jobs.values() if j.record.is_active]
            if not active:
                return 1.0
            return sum(j.record.progress for j in active) / len(active)

    # -- control -------------------------------------------------------
    def cancel(self, job_id: str) -> bool:
        job = self.get(job_id)
        if job is None or not job.record.is_active:
            return False
        job.cancel_event.set()
        job.record.message = "cancelling..."
        self._store.upsert_job(job.record)
        return True

    def cancel_all(self) -> int:
        count = 0
        for job in self.list(active_only=True):
            if self.cancel(job["id"]):
                count += 1
        return count

    def wait(self, job_id: str, timeout: float = 300.0, poll: float = 0.15) -> Optional[Dict[str, Any]]:
        """Block until a job finishes (used by tests and by synchronous callers)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            job = self.get(job_id)
            if job is None:
                return None
            if not job.record.is_active:
                return job.to_dict()
            time.sleep(poll)
        return self.status(job_id)

    def clear_finished(self) -> int:
        with self._lock:
            finished = [jid for jid, j in self._jobs.items() if not j.record.is_active]
            for jid in finished:
                self._jobs.pop(jid, None)
        return len(finished)

    def _trim(self) -> None:
        if len(self._jobs) <= MAX_HISTORY:
            return
        finished = sorted(
            (j for j in self._jobs.values() if not j.record.is_active),
            key=lambda j: j.record.created_at,
        )
        for job in finished[: len(self._jobs) - MAX_HISTORY]:
            self._jobs.pop(job.record.id, None)


_QUEUE: Optional[JobQueue] = None
_QUEUE_LOCK = threading.Lock()


def get_queue() -> JobQueue:
    """Process-wide queue (survives Streamlit reruns)."""
    global _QUEUE
    if _QUEUE is None:
        with _QUEUE_LOCK:
            if _QUEUE is None:
                _QUEUE = JobQueue()
    return _QUEUE
