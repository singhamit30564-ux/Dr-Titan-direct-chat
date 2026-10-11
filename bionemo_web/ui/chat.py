"""The chat surface: the console's primary interface.

Design notes that matter for behaviour:

* ``st.chat_input`` is rendered at the top level of the page function, so it stays
  pinned to the bottom of the viewport like a real chat app.
* Long work runs in the background job queue (see ``engine/jobs.py``); the active
  -jobs panel is a Streamlit *fragment* that re-polls itself every second, so
  progress moves without re-running the whole app or losing the conversation.
* Assistant messages persist their full payload, so reopening a conversation
  re-renders the same structure viewers, tables and downloads as the live run.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from ..engine.llm import LLMClient
from ..nims import catalog
from . import cards
from .theme import PALETTE, badge_html, empty_state

QUICK_ACTIONS: List[Dict[str, str]] = [
    {
        "label": "Fold a protein",
        "prompt": "fold MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKR",
        "hint": "ESMFold / AlphaFold2",
    },
    {
        "label": "Design sequences",
        "prompt": "design 8 sequences for the last backbone",
        "hint": "ProteinMPNN inverse folding",
    },
    {
        "label": "Generate analogues",
        "prompt": "generate 12 analogues of CC(=O)Oc1ccccc1C(=O)O optimising QED",
        "hint": "MolMIM / GenMol",
    },
    {
        "label": "Dock a ligand",
        "prompt": "dock CC(=O)Oc1ccccc1C(=O)O into the last structure",
        "hint": "DiffDock",
    },
    {
        "label": "Virtual screening",
        "prompt": "virtual screen CC(=O)Oc1ccccc1C(=O)O against the last structure --num_molecules 6 --num_poses 3",
        "hint": "Full generative pipeline",
    },
    {
        "label": "Analyse a sequence",
        "prompt": "analyse MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ",
        "hint": "Local biophysics",
    },
]


# --------------------------------------------------------------------------
# Sidebar-piece helpers
# --------------------------------------------------------------------------


def render_quick_actions() -> Optional[str]:
    """Clickable prompt starters; returns a prompt if one was clicked."""
    st.markdown(
        '<div class="bm-card-meta" style="margin-bottom:6px">Start here - every action below is a '
        "real tool call, and the result states which backend answered.</div>",
        unsafe_allow_html=True,
    )
    columns = st.columns(3)
    for index, action in enumerate(QUICK_ACTIONS):
        with columns[index % 3]:
            if st.button(f"{action['label']}\n\n{action['hint']}", key=f"quick_{index}",
                         width="stretch"):
                return action["prompt"]
    return None


def render_active_jobs(queue: Any, *, fragment_seconds: float = 1.0) -> None:
    """Live progress for queued/running work (self-polling fragment)."""
    active = queue.list(active_only=True)
    if not active:
        return

    @st.fragment(run_every=fragment_seconds)
    def _panel() -> None:
        jobs = queue.list(active_only=True)
        if not jobs:
            st.rerun()
        for job in jobs:
            status = job["status"]
            progress = float(job.get("progress") or 0.0)
            label = f"{job['title']} — {job.get('message') or status}"
            st.progress(min(1.0, max(0.0, progress)), text=label)
            info_columns = st.columns([1, 1, 2])
            with info_columns[0]:
                st.markdown(badge_html(status.upper(),
                                       "warn" if status == "running" else "info"), unsafe_allow_html=True)
            with info_columns[1]:
                if st.button("Cancel", key=f"cancel_{job['id']}", width="stretch"):
                    queue.cancel(job["id"])
                    st.rerun()
            with info_columns[2]:
                elapsed = time.time() - (job.get("started_at") or job.get("created_at") or time.time())
                st.markdown(f'<div class="bm-card-meta">{elapsed:.0f}s elapsed · {progress * 100:.0f}%</div>',
                            unsafe_allow_html=True)

    _panel()


# --------------------------------------------------------------------------
# Message rendering
# --------------------------------------------------------------------------


def render_user_message(message: Dict[str, Any]) -> None:
    st.markdown(
        f'<div class="bm-user-msg"><div class="bm-msg-role">You</div>'
        f'{_escape(message.get("content", ""))}</div>',
        unsafe_allow_html=True,
    )


def render_assistant_message(message: Dict[str, Any], *, conversation_id: str) -> None:
    payload = message.get("payload") or {}
    kind = payload.get("kind", "chat")
    st.markdown('<div class="bm-assistant-msg"><div class="bm-msg-role">Console</div></div>',
                unsafe_allow_html=True)
    st.markdown(message.get("content", ""))

    source = payload.get("narration_source")
    if source:
        tone = {"llm": "violet", "offline": "accent"}.get(source, "info")
        label = {"llm": "LANGUAGE MODEL NARRATION", "offline": "DETERMINISTIC NARRATION"}.get(
            source, source.upper()
        )
        st.markdown(badge_html(label, tone, dot=False), unsafe_allow_html=True)

    plan = payload.get("plan") or {}
    if plan.get("steps"):
        with st.expander("What the console decided to run", expanded=False):
            st.dataframe(
                pd.DataFrame([
                    {"tool": s.get("tool"), "confidence": s.get("confidence"),
                     "why": s.get("rationale"),
                     "arguments": ", ".join(f"{k}={_short(v)}" for k, v in (s.get("args") or {}).items())}
                    for s in plan["steps"]
                ]),
                width="stretch", hide_index=True,
            )
            st.caption(f"Planner: {plan.get('planner', 'rules')}. "
                       + (" ".join(plan.get("notes") or []) if plan.get("notes") else ""))

    outcomes = payload.get("outcomes") or []
    for index, outcome_payload in enumerate(outcomes):
        _render_persisted_outcome(outcome_payload, key=f"{message['id']}_{index}")

    if payload.get("job_id"):
        st.caption(f"Background job `{payload['job_id']}`. Progress appears above while it runs.")

    if outcomes or kind in ("chat", "clarification"):
        _message_footer(message)


def _render_persisted_outcome(data: Dict[str, Any], *, key: str) -> None:
    """Re-render an outcome from its JSON payload (after a reload or rerun)."""
    from ..engine.tools import ToolOutcome

    outcome = ToolOutcome(
        tool=data.get("tool", ""),
        runner=data.get("runner", ""),
        ok=bool(data.get("ok")),
        title=data.get("title", ""),
        renderer=data.get("renderer", "text"),
        payload=data.get("payload") or {},
        summary=data.get("summary", ""),
        steps=data.get("steps") or [],
        elapsed_s=float(data.get("elapsed_s") or 0.0),
        error=data.get("error", ""),
        run_id=data.get("run_id", ""),
        provenance=data.get("provenance") or {},
    )
    # The header carries the backend badge (LOCAL NIM / NVIDIA HOSTED / SIMULATED),
    # which has to survive a page reload: it is rebuilt from the provenance
    # snapshot recorded with the message, never guessed.
    st.markdown(
        f'<div class="bm-card-head"><span class="bm-card-title">{outcome.title}</span>'
        f'<span>{cards.backend_badges(outcome)}</span></div>'
        f'<div class="bm-summary">{outcome.summary}</div>',
        unsafe_allow_html=True,
    )
    if outcome.renderer in cards.RENDERERS:
        with st.expander("Open result",
                         expanded=bool(outcome.payload.get("rows") or outcome.payload.get("pockets"))):
            cards.render_outcome(outcome, key=key, show_header=False)
    else:
        with st.expander("Details"):
            st.json(data, expanded=False)


def _message_footer(message: Dict[str, Any]) -> None:
    columns = st.columns([1, 1, 1, 6])
    message_id = message.get("id", "")
    with columns[0]:
        if st.button("👍", key=f"up_{message_id}", help="Mark this answer useful"):
            st.session_state.setdefault("feedback", {})[message_id] = 1
            st.toast("Feedback recorded.")
    with columns[1]:
        if st.button("👎", key=f"down_{message_id}", help="Mark this answer unhelpful"):
            st.session_state.setdefault("feedback", {})[message_id] = -1
            st.toast("Feedback recorded.")
    with columns[2]:
        st.caption("·")
    with columns[3]:
        st.caption((message.get("content", "")[:0] or "") + "")


def _escape(text: str) -> str:
    return (str(text or "")
            .replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace("\n", "<br>"))


def _short(value: Any, limit: int = 46) -> str:
    text = str(value)
    return text if len(text) <= limit else text[:limit] + "…"


# --------------------------------------------------------------------------
# Composer
# --------------------------------------------------------------------------


def render_composer(*, conversation_id: str, placeholder: str) -> Optional[Dict[str, Any]]:
    """Chat input plus the attachment row. Returns ``{"text":..., "attachments": {...}}``."""
    prompt = st.chat_input(placeholder)
    attachments: Dict[str, str] = {}

    with st.expander("Attach input (PDB, FASTA, SMILES)", expanded=False):
        columns = st.columns(3)
        with columns[0]:
            pdb_file = st.file_uploader("PDB", type=["pdb", "ent", "cif"], key=f"pdb_up_{conversation_id}")
            if pdb_file is not None:
                try:
                    attachments["pdb"] = pdb_file.read().decode("utf-8", "replace")
                    st.caption(f"{len(attachments['pdb'].splitlines())} lines loaded")
                except Exception as exc:
                    st.warning(f"Could not read the PDB: {exc}")
        with columns[1]:
            fasta_file = st.file_uploader("FASTA / sequence", type=["fasta", "fa", "txt", "seq"],
                                          key=f"fa_up_{conversation_id}")
            if fasta_file is not None:
                try:
                    text = fasta_file.read().decode("utf-8", "replace")
                    from ..bio.sequence import parse_fasta

                    records = parse_fasta(text)
                    attachments["sequence"] = records[0]["sequence"] if records else text.strip().splitlines()[-1]
                    st.caption(f"{len(attachments['sequence'])} residues loaded")
                except Exception as exc:
                    st.warning(f"Could not read the FASTA: {exc}")
        with columns[2]:
            smiles_file = st.file_uploader("SMILES / SDF", type=["smi", "smiles", "sdf", "csv"],
                                            key=f"smi_up_{conversation_id}")
            if smiles_file is not None:
                try:
                    text = smiles_file.read().decode("utf-8", "replace")
                    if smiles_file.name.lower().endswith(".sdf"):
                        from rdkit import Chem

                        supplier = Chem.SDMolSupplier()
                        supplier.SetData(text, sanitize=False)
                        mols = [m for m in supplier if m is not None]
                        smiles = [Chem.MolToSmiles(m) for m in mols]
                    else:
                        tokens = [t.strip() for line in text.splitlines() for t in line.split() if t.strip()]
                        from ..bio.chem import parse_smiles

                        smiles = [t for t in tokens if parse_smiles(t)[0] is not None]
                    if smiles:
                        attachments["smiles"] = "\n".join(smiles[:500])
                        st.caption(f"{len(smiles)} molecule(s) loaded")
                    else:
                        st.warning("No molecules could be parsed from that file.")
                except Exception as exc:
                    st.warning(f"Could not read the molecule file: {exc}")
        st.caption(
            "Attachments take priority over text: if you attach a PDB and type \"find its pockets\", "
            "the attached structure is used."
        )

    # Natural-language payload detection: a long sequence or SMILES in the text
    # does not need an attachment at all, and the router already extracts it.
    if prompt is None or not str(prompt).strip():
        return None
    return {"text": str(prompt).strip(), "attachments": attachments}


def handle_turn(agent: Any, *, conversation_id: str, turn_input: Dict[str, Any],
                backend: str) -> None:
    """Run one turn, streaming the narration when a model is available."""
    text = turn_input.get("text", "")
    attachments = turn_input.get("attachments") or {}
    placeholder = st.chat_input  # placeholder to keep flake quiet if unused
    del placeholder

    with st.chat.message("assistant") if hasattr(st, "chat") and hasattr(st.chat, "message") else _nullcontext():
        with st.status("Working…", expanded=True) as status:
            status.write("Reading the request")
            result = agent.handle_message(
                conversation_id, text, backend=backend, attachments=attachments,
                prefer_background=True,
            )
            if result.kind == "pending":
                status.update(label="Running in the background — progress is shown above",
                              state="running", expanded=False)
            else:
                status.update(label=f"Done in {result.elapsed_s:.1f}s", state="complete", expanded=False)
    st.session_state["_last_turn"] = result.to_dict()
    st.rerun()


class _nullcontext:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# --------------------------------------------------------------------------
# Page body
# --------------------------------------------------------------------------


def render_chat_page(
    *,
    agent: Any,
    store: Any,
    queue: Any,
    llm: LLMClient,
    conversation_id: str,
    backend: str,
) -> None:
    """Full chat page: history, live jobs, composer."""
    messages = store.messages(conversation_id)
    if not messages:
        st.markdown(
            empty_state(
                "Ask for a fold, a molecule, a dock or a full screen — or attach a file.",
                "Example: “fold MKTAYIAKQRQ…”, “generate 12 analogues of aspirin optimising QED”, "
                "“run a virtual screen against the last structure”.",
            ),
            unsafe_allow_html=True,
        )
        quick = render_quick_actions()
        if quick:
            st.session_state["_pending_prompt"] = quick
            st.rerun()
        st.caption(
            "Every run records which backend answered it: a **LOCAL NIM** container, **NVIDIA HOSTED** "
            "endpoints, a **LOCAL RUNTIME** model, **LOCAL ANALYSIS** code, or the **SIMULATED** "
            "backends that make the console usable without a GPU."
        )

    for message in messages:
        if message["role"] == "user":
            render_user_message(message)
        else:
            render_assistant_message(message, conversation_id=conversation_id)

    render_active_jobs(queue)

    # A finished background job is folded into the conversation exactly once.
    _ingest_finished_jobs(store=store, queue=queue, conversation_id=conversation_id)

    pending = st.session_state.pop("_pending_prompt", None)
    if pending:
        handle_turn(agent, conversation_id=conversation_id,
                    turn_input={"text": pending, "attachments": {}}, backend=backend)

    turn_input = render_composer(conversation_id=conversation_id,
                                 placeholder="Ask for a fold, molecule, dock, design or screen…")
    if turn_input:
        handle_turn(agent, conversation_id=conversation_id, turn_input=turn_input, backend=backend)


def _ingest_finished_jobs(*, store: Any, queue: Any, conversation_id: str) -> None:
    """Move completed chat jobs into the persisted conversation."""
    jobs = queue.list(conversation_id=conversation_id)
    done = [j for j in jobs if j["status"] in ("done", "failed", "cancelled")]
    handled = st.session_state.setdefault("_handled_jobs", set())
    for job in done:
        if job["id"] in handled:
            continue
        handled.add(job["id"])
        record = queue.get(job["id"])
        payload = getattr(record, "result", None) if record else None
        if payload is None:
            continue
        summary = payload.result_summary() if hasattr(payload, "result_summary") else {}
        store.add_message(
            conversation_id,
            "assistant",
            payload.text,
            payload={
                "kind": "result",
                "narration_source": payload.narration_source,
                "job_id": job["id"],
                "outcomes": [o.to_dict() for o in getattr(payload, "outcomes", [])],
                "plan": {"steps": [{"tool": o.tool} for o in getattr(payload, "outcomes", [])]},
                "summary": summary,
            },
        )
        st.rerun()
