"""The chat agent: one user turn in, one grounded answer out.

Sequence for every message:

1. persist the user message,
2. rebuild the session context (last sequence, last structure, last molecule),
3. plan tool calls with the deterministic router (an LLM may plan only as a
   last resort),
4. execute the plan -- inline for quick work, through the job queue for long
   pipelines,
5. narrate the outcome: a language model if one is connected, otherwise
   :class:`~bionemo_web.engine.llm.OfflineAnalyst`, which writes the summary
   straight from the result objects.

The agent never invents a number. Every statement it makes is derived from a
recorded run with a provenance manifest.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..nims.service import NimService, RunResult
from . import tools as tools_mod
from .jobs import JobContext, get_queue
from .llm import LLMClient, OfflineAnalyst, context_for_llm
from .router import Plan, plan as build_plan
from .store import get_store

log = logging.getLogger("bionemo_web.agent")

#: Tools that can take long enough to need the background queue.
HEAVY_TOOLS = {"virtual_screening"}


@dataclass
class TurnResult:
    """What one turn produced, for the UI to render."""

    conversation_id: str
    user_message_id: str = ""
    assistant_message_id: str = ""
    kind: str = "answer"          # answer | pending | clarification | chat
    text: str = ""
    outcomes: List[Any] = field(default_factory=list)
    plan: Optional[Plan] = None
    job_id: str = ""
    run_ids: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    elapsed_s: float = 0.0
    narration_source: str = "offline"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "assistant_message_id": self.assistant_message_id,
            "kind": self.kind,
            "text": self.text,
            "job_id": self.job_id,
            "run_ids": self.run_ids,
            "notes": self.notes,
            "narration_source": self.narration_source,
            "n_outcomes": len(self.outcomes),
            "elapsed_s": round(self.elapsed_s, 2),
        }


class ChatAgent:
    """Coordinates store, router, tools, service, LLM and jobs."""

    def __init__(
        self,
        service: NimService,
        *,
        llm: Optional[LLMClient] = None,
        store: Any = None,
    ) -> None:
        self.service = service
        self.llm = llm or LLMClient()
        self.store = store or get_store()
        self.queue = get_queue()

    # -- session context ----------------------------------------------
    def session_context(self, conversation_id: str) -> Dict[str, Any]:
        """Derive what the session already knows, for follow-up messages."""
        ctx: Dict[str, Any] = {
            "last_sequence": "",
            "last_structure_pdb": "",
            "previous_structure_pdb": "",
            "last_smiles": "",
            "last_report": None,
            "structures": [],
        }
        runs = self.store.runs(limit=25, conversation_id=conversation_id)
        for run in runs:
            meta = run.get("meta") or {}
            canonical = run.get("canonical") or {}
            if not ctx["last_structure_pdb"] and canonical.get("pdbs"):
                ctx["last_structure_pdb"] = canonical["pdbs"][0]
                ctx["structures"].append({"run_id": run["id"], "model": run["model_key"]})
            if ctx["last_structure_pdb"] and not ctx["previous_structure_pdb"] and len(ctx["structures"]) > 1:
                # A second, older structure (used by compare_structures).
                for older in runs[runs.index(run) + 1:]:
                    old_pdbs = (older.get("canonical") or {}).get("pdbs")
                    if old_pdbs:
                        ctx["previous_structure_pdb"] = old_pdbs[0]
                        break
            request = run.get("request") or {}
            if not ctx["last_sequence"]:
                seq = request.get("sequence") or meta.get("sequence") or ""
                if seq and set(str(seq).upper()) <= set("ACDEFGHIKLMNPQRSTVWY"):
                    ctx["last_sequence"] = str(seq)
            if not ctx["last_smiles"]:
                smi = request.get("smi") or request.get("smiles") or request.get("ligand") or ""
                if smi and len(str(smi)) < 400:
                    ctx["last_smiles"] = str(smi)
            if not ctx["last_report"] and meta.get("report"):
                ctx["last_report"] = meta["report"]
        if not ctx["last_sequence"] and ctx["last_structure_pdb"]:
            try:
                from ..bio import structure as st_mod

                ctx["last_sequence"] = st_mod.parse_pdb(ctx["last_structure_pdb"]).sequence()
            except Exception:
                pass
        return ctx

    # -- main entry ----------------------------------------------------
    def handle_message(
        self,
        conversation_id: str,
        text: str,
        *,
        backend: str = "auto",
        attachments: Optional[Dict[str, str]] = None,
        prefer_background: bool = True,
    ) -> TurnResult:
        started = time.time()
        attachments = attachments or {}
        if not conversation_id:
            conversation_id = self.store.create_conversation(text[:60] or "New conversation")
        user_message_id = self.store.add_message(conversation_id, "user", text)
        self.store.auto_title(conversation_id)

        ctx = self.session_context(conversation_id)
        ctx["conversation_id"] = conversation_id
        # Attachments take precedence: they are explicit user input.
        if attachments.get("pdb"):
            ctx["last_structure_pdb"] = attachments["pdb"]
        if attachments.get("sequence"):
            ctx["last_sequence"] = attachments["sequence"]
        if attachments.get("smiles"):
            ctx["last_smiles"] = attachments["smiles"]

        combined = text
        for kind, value in attachments.items():
            if kind == "smiles":
                combined += f"\n{value}"
            elif kind in ("sequence", "pdb") and value not in text:
                combined += f"\n{value[:400]}"

        plan = build_plan(combined, context=ctx, llm=self.llm)
        turn = TurnResult(conversation_id=conversation_id, user_message_id=user_message_id, plan=plan)

        if plan.chat_only or not plan.steps:
            answer, source = self._answer_question(text, ctx, plan)
            turn.kind = "chat"
            turn.text = answer
            turn.narration_source = source
            turn.assistant_message_id = self.store.add_message(
                conversation_id, "assistant", answer,
                payload={"plan": plan.to_dict(), "kind": "chat", "narration_source": source},
            )
            turn.notes.extend(plan.notes)
            turn.elapsed_s = time.time() - started
            return turn

        if plan.clarifications and any(step.needs for step in plan.steps):
            missing = ", ".join(sorted({m for step in plan.steps for m in step.needs}))
            answer = (
                f"I can run **{plan.steps[0].tool.replace('_', ' ')}**, but I still need: {missing}. "
                "Paste the missing input (or attach a file) and I will continue."
            )
            turn.kind = "clarification"
            turn.text = answer
            turn.assistant_message_id = self.store.add_message(
                conversation_id, "assistant", answer, payload={"plan": plan.to_dict(), "kind": "clarification"}
            )
            turn.elapsed_s = time.time() - started
            return turn

        heavy = prefer_background and any(step.tool in HEAVY_TOOLS for step in plan.steps)
        if heavy:
            job_id = self.queue.submit(
                "pipeline",
                f"Chat request: {plan.steps[0].tool.replace('_', ' ')}",
                lambda ctx_job: self._run_plan_job(ctx_job, plan, backend=backend, ctx=ctx,
                                                   conversation_id=conversation_id,
                                                   user_message_id=user_message_id),
                params={"text": text[:400], "tools": [s.tool for s in plan.steps], "backend": backend},
                conversation_id=conversation_id,
            )
            turn.kind = "pending"
            turn.job_id = job_id
            turn.text = (
                "Running this in the background: "
                + ", ".join(s.tool.replace("_", " ") for s in plan.steps)
                + ". Progress and results appear below as soon as each step finishes."
            )
            turn.assistant_message_id = self.store.add_message(
                conversation_id, "assistant", turn.text,
                payload={"plan": plan.to_dict(), "kind": "pending", "job_id": job_id},
            )
            turn.elapsed_s = time.time() - started
            return turn

        outcomes = self._execute_steps(plan, ctx, backend=backend)
        turn.outcomes = outcomes
        turn.run_ids = [o.run_id for o in outcomes if o.run_id]
        narration, source = self._narrate(plan, outcomes, ctx)
        turn.text = narration
        turn.narration_source = source
        turn.assistant_message_id = self.store.add_message(
            conversation_id, "assistant", narration,
            payload={
                "plan": plan.to_dict(),
                "kind": "result",
                "outcomes": [o.to_dict() for o in outcomes],
                "run_ids": turn.run_ids,
                "narration_source": source,
            },
        )
        turn.notes.extend(plan.notes)
        turn.elapsed_s = time.time() - started
        return turn

    # -- execution -----------------------------------------------------
    def _execute_steps(self, plan: Plan, ctx: Dict[str, Any], *, backend: str,
                       progress: Optional[Any] = None) -> List[Any]:
        outcomes: List[Any] = []
        for index, step in enumerate(plan.steps):
            args = dict(step.args)
            # Chain: a step that needs a protein can use the structure produced
            # by the previous step in this same turn.
            for key in ("pdb", "protein"):
                if key in tools_mod.TOOLS[step.tool].parameters and not args.get(key):
                    if outcomes:
                        previous = outcomes[-1]
                        previous_pdb = (previous.payload or {}).get("pdb") or (
                            previous.run.meta.get("structure").pdb_text
                            if previous.run and previous.run.meta.get("structure") else ""
                        )
                        if previous_pdb:
                            args[key] = previous_pdb
                    elif ctx.get("last_structure_pdb"):
                        args[key] = ctx["last_structure_pdb"]
                    elif getattr(previous, "run", None) is not None and key == "pdb":
                        structure = (previous.run.meta or {}).get("structure")
                        if structure is not None and getattr(structure, "pdb_text", ""):
                            args[key] = structure.pdb_text
            if progress:
                progress(index / max(1, len(plan.steps)), f"step {index + 1}: {step.tool}")
            outcome = tools_mod.execute(step.tool, args, self.service, backend=backend, context=ctx,
                                        progress=progress)
            outcome.run_id = self._record_outcome(outcome, ctx, args)
            outcomes.append(outcome)
            # Keep the derived context fresh for the next step.
            payload = outcome.payload or {}
            if payload.get("pdb"):
                ctx["last_structure_pdb"] = payload["pdb"]
            elif getattr(outcome, "run", None) is not None:
                # A fold tool keeps the parsed structure in run.meta rather than in
                # the payload, so a follow-up step in the same turn ("now design
                # sequences for that backbone") still has a backbone to work from.
                structure = (outcome.run.meta or {}).get("structure")
                if structure is not None and getattr(structure, "pdb_text", ""):
                    ctx["last_structure_pdb"] = structure.pdb_text
            if payload.get("sequence"):
                ctx["last_sequence"] = payload["sequence"]
            if args.get("smiles") or args.get("seed_smiles") or args.get("ligand"):
                ctx["last_smiles"] = args.get("smiles") or args.get("seed_smiles") or args.get("ligand")
        return outcomes

    def _record_outcome(self, outcome: Any, ctx: Dict[str, Any], args: Dict[str, Any]) -> str:
        """Persist the run behind an outcome and log the HTTP call if there was one."""
        run: Optional[RunResult] = getattr(outcome, "run", None)
        try:
            if run is not None:
                provenance = run.provenance()
                run_id = self.store.record_run(
                    model_key=run.spec_key,
                    backend=run.backend,
                    ok=run.ok,
                    simulated=run.simulated,
                    latency_ms=run.latency_ms,
                    summary=outcome.summary[:4000],
                    canonical=run.canonical,
                    meta=run.meta,
                    request=run.request_payload,
                    warnings=run.warnings,
                    error=run.error,
                    provenance=provenance,
                    conversation_id=ctx.get("conversation_id", ""),
                )
                if run.backend in ("local", "hosted"):
                    self.store.log_call(
                        model_key=run.spec_key, backend=run.backend, ok=run.ok, status=run.status,
                        latency_ms=run.latency_ms, attempts=run.attempts, url=run.url, error=run.error,
                        request=run.request_payload,
                    )
                return run_id
            # Local analysis: still record a run so the history is complete.
            return self.store.record_run(
                model_key=outcome.tool,
                backend="local analysis",
                ok=outcome.ok,
                simulated=False,
                latency_ms=outcome.elapsed_s * 1000,
                summary=outcome.summary[:4000],
                canonical={},
                meta={"payload": {k: v for k, v in (outcome.payload or {}).items()
                                  if isinstance(v, (str, int, float, bool, list, dict, type(None)))}},
                request=args,
                provenance={
                    "app": "Dr-Titan BioNeMo Web Console",
                    "runner": outcome.runner,
                    "method": "local analysis (no model call)",
                },
                conversation_id=ctx.get("conversation_id", ""),
            )
        except Exception as exc:  # pragma: no cover - persistence must never break a turn
            log.warning("could not record run: %s", exc)
            return ""

    # -- background job body ------------------------------------------
    def _run_plan_job(
        self, job: JobContext, plan: Plan, *, backend: str, ctx: Dict[str, Any],
        conversation_id: str, user_message_id: str,
    ) -> "JobPayload":
        ctx = dict(ctx)
        ctx["conversation_id"] = conversation_id
        step_progress = {"i": 0, "n": max(1, len(plan.steps))}

        def progress(fraction: float, message: str) -> None:
            job.check_cancelled()
            job.progress((step_progress["i"] + fraction) / step_progress["n"], message)

        outcomes = []
        for index, step in enumerate(plan.steps):
            step_progress["i"] = index
            job.check_cancelled()
            outcome = self._execute_steps_guarded(step, plan, ctx, backend=backend, progress=progress)
            outcomes.append(outcome)
        narration, source = self._narrate(plan, outcomes, ctx)
        job.progress(1.0, "done")
        return JobPayload(
            conversation_id=conversation_id,
            user_message_id=user_message_id,
            text=narration,
            narration_source=source,
            outcomes=outcomes,
        )

    def _execute_steps_guarded(self, step: Any, plan: Plan, ctx: Dict[str, Any], *, backend: str,
                               progress: Any) -> Any:
        """Execute one step with the same chaining rules as the inline path."""
        args = dict(step.args)
        for key in ("pdb", "protein"):
            if key in tools_mod.TOOLS[step.tool].parameters and not args.get(key):
                if ctx.get("last_structure_pdb"):
                    args[key] = ctx["last_structure_pdb"]
        outcome = tools_mod.execute(step.tool, args, self.service, backend=backend, context=ctx,
                                    progress=progress)
        outcome.run_id = self._record_outcome(outcome, ctx, args)
        payload = outcome.payload or {}
        if payload.get("pdb"):
            ctx["last_structure_pdb"] = payload["pdb"]
        elif getattr(outcome, "run", None) is not None:
            structure = (outcome.run.meta or {}).get("structure")
            if structure is not None and getattr(structure, "pdb_text", ""):
                ctx["last_structure_pdb"] = structure.pdb_text
        if args.get("seed_smiles"):
            ctx["last_smiles"] = args["seed_smiles"]
        return outcome

    # -- narration -----------------------------------------------------
    def _narrate(self, plan: Plan, outcomes: List[Any], ctx: Dict[str, Any]) -> Tuple[str, str]:
        """Write the assistant's prose from the computed outcomes."""
        base = " ".join(o.summary for o in outcomes if o.summary).strip()
        failures = [o for o in outcomes if not o.ok]
        if failures:
            base += (" " if base else "") + " ".join(
                f"The {o.tool.replace('_', ' ')} step failed: {o.error}" for o in failures
            )
        if not base:
            base = "The run finished but produced no summary."

        blocked = any(
            (getattr(o.run, "simulated", False) if getattr(o, "run", None) else False) for o in outcomes
        )
        if self.llm.status().available and outcomes:
            context = "\n\n".join(context_for_llm(o.run) if getattr(o, "run", None) else json.dumps(
                {"tool": o.tool, "summary": o.summary, "payload_keys": sorted(o.payload.keys())}, default=str)
                for o in outcomes)
            instruction = (
                "Summarise these BioNeMo results for the scientist who asked. Lead with the number "
                "that matters, mention the top hit or the key structural feature, and state any caveat "
                "that affects interpretation."
                + (" Say plainly that the results are simulated." if blocked else "")
            )
            answer = self.llm.summarise(instruction, context)
            if answer:
                return answer, "llm"
        return base, "offline"

    def _answer_question(self, question: str, ctx: Dict[str, Any], plan: Plan) -> Tuple[str, str]:
        """Answer a question that needs no tool call."""
        if self.llm.status().available:
            stats = self.store.stats()
            context = json.dumps(
                {
                    "last_sequence": (ctx.get("last_sequence") or "")[:400],
                    "has_structure": bool(ctx.get("last_structure_pdb")),
                    "last_smiles": ctx.get("last_smiles", ""),
                    "last_report": ctx.get("last_report"),
                    "session_stats": stats.get("counts"),
                },
                default=str,
            )[:6000]
            answer = self.llm.complete(
                [
                    {"role": "system", "content": self.llm.config.system_prompt},
                    {
                        "role": "user",
                        "content": (
                            f"Question: {question}\n\nSession context (use only this):\n{context}\n\n"
                            "Answer in at most 6 sentences. If the session lacks the data needed, say "
                            "what to run instead."
                        ),
                    },
                ]
            ).strip()
            if answer:
                return answer, "llm"
        return OfflineAnalyst.answer_question(question, {"stats": self.store.stats(), "last_run": {}}), "offline"


@dataclass
class JobPayload:
    """Result object returned by a background chat job."""

    conversation_id: str
    user_message_id: str
    text: str
    narration_source: str
    outcomes: List[Any] = field(default_factory=list)
    kind: str = "result"

    def result_summary(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "text": self.text[:600],
            "n_outcomes": len(self.outcomes),
            "tools": [o.tool for o in self.outcomes],
            "ok": all(o.ok for o in self.outcomes) if self.outcomes else False,
        }
