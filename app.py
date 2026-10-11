"""Dr-Titan BioNeMo Web Console — entry point.

Run it:

    streamlit run app.py

Everything else (models, backends, storage, jobs) lives in the ``bionemo_web``
package. This file is deliberately thin: configuration bootstrap, sidebar, and
page routing, nothing more.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bionemo_web.console import pages as console_pages  # noqa: E402
from bionemo_web.engine.agent import ChatAgent  # noqa: E402
from bionemo_web.engine.jobs import get_queue  # noqa: E402
from bionemo_web.engine.llm import LLMClient, LLMConfig  # noqa: E402
from bionemo_web.engine.store import get_store  # noqa: E402
from bionemo_web.nims import catalog  # noqa: E402
from bionemo_web.nims.client import NimClient  # noqa: E402
from bionemo_web.nims.service import BACKEND_LABELS, NimService  # noqa: E402
from bionemo_web.ui import chat as chat_ui  # noqa: E402
from bionemo_web.ui.theme import PALETTE, badge_html, css  # noqa: E402

APP_TITLE = "Dr Titan · BioNeMo Console"
PAGES = [
    "Chat",
    "Structure Lab",
    "Molecule Workbench",
    "Screening Pipeline",
    "Administration",
    "History",
]

DEFAULTS = {
    "prefer_backend": os.environ.get("BIONEMO_BACKEND", "auto"),
    "allow_hosted": True,
    "simulation_seed": 20241011,
    "llm_provider": os.environ.get("BIONEMO_LLM_PROVIDER", "auto"),
    "llm_model": "",
    "llm_base_url": "",
    "llm_api_key": "",
    "llm_temperature": 0.2,
    "local_endpoints": {},
}


# --------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def _bootstrap() -> dict:
    """Build the long-lived singletons once per server process."""
    store = get_store()
    queue = get_queue()

    settings = dict(DEFAULTS)
    settings.update(store.get_settings())

    # Environment always wins over stored values for secrets and endpoints.
    endpoints = dict(settings.get("local_endpoints") or {})
    default_local = os.environ.get("BIONEMO_NIM_URL", "http://localhost:8000")
    for key in ("esmfold", "alphafold2", "openfold3", "boltz2", "molmim", "genmol", "diffdock",
                "proteinmpnn", "esm2", "evo2", "rfdiffusion", "msa_search"):
        endpoints.setdefault(key, default_local)
    settings["local_endpoints"] = endpoints

    def _on_call(record: dict) -> None:
        store.log_call(**record)

    client = NimClient(
        local_endpoints=endpoints,
        hosted_api_key=os.environ.get("NVIDIA_API_KEY") or os.environ.get("NGC_API_KEY", ""),
        hosted_base=os.environ.get("BIONEMO_HOSTED_URL", catalog.HOSTED_BASE),
        on_call=_on_call,
    )
    service = NimService(client, settings)
    llm = LLMClient(LLMConfig.from_settings(settings))
    agent = ChatAgent(service, llm=llm, store=store)
    return {"store": store, "queue": queue, "settings": settings, "client": client,
            "service": service, "llm": llm, "agent": agent}


def _session_defaults() -> None:
    st.session_state.setdefault("page", PAGES[0])
    st.session_state.setdefault("backend_override", "")
    st.session_state.setdefault("_handled_jobs", set())


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------


def _sidebar(context: dict) -> tuple:
    store, queue, settings = context["store"], context["queue"], context["settings"]
    service, llm = context["service"], context["llm"]

    with st.sidebar:
        st.markdown(
            '<div class="bm-brand"><div class="bm-brand-mark">DT</div><div>'
            '<div class="bm-brand-title">Dr Titan</div>'
            '<div class="bm-brand-sub">BioNeMo Console</div></div></div>',
            unsafe_allow_html=True,
        )

        page = st.radio("Navigate", PAGES, index=PAGES.index(st.session_state["page"]),
                        label_visibility="collapsed", key="nav_radio")
        if page != st.session_state["page"]:
            st.session_state["page"] = page
            st.rerun()

        st.divider()
        env = service.environment()
        env_chips = [
            badge_html("GPU" if env["gpu_available"] else "NO GPU",
                       "good" if env["gpu_available"] else "warn"),
            badge_html("NGC KEY" if env["ngc_key_present"] else "NO KEY",
                       "good" if env["ngc_key_present"] else "info"),
        ]
        st.markdown(" ".join(env_chips), unsafe_allow_html=True)

        backend = settings.get("prefer_backend", "auto")
        override = st.selectbox(
            "Backend for this session", ["", *BACKEND_LABELS.keys()],
            format_func=lambda k: "Use the default" if k == "" else BACKEND_LABELS[k],
            key="backend_select",
            help="Overrides the Administration default until you change it back.",
        )
        st.session_state["backend_override"] = override
        effective = override or backend
        st.caption(f"Effective backend: **{BACKEND_LABELS.get(effective, effective)}**")

        llm_status = llm.status()
        st.markdown(
            badge_html("LLM CONNECTED" if llm_status.available else "NO LLM",
                       "good" if llm_status.available else "info")
            + f'<div class="bm-card-meta" style="margin-top:4px">{llm_status.message}</div>',
            unsafe_allow_html=True,
        )

        if queue.active_count():
            st.markdown(
                badge_html(f"{queue.active_count()} JOB(S) RUNNING", "warn"), unsafe_allow_html=True
            )

        st.divider()
        st.markdown('<div class="bm-card-meta">Conversations</div>', unsafe_allow_html=True)
        if st.button("＋ New conversation", width="stretch", key="new_conv"):
            conversation_id = store.create_conversation("New conversation")
            st.session_state["conversation_id"] = conversation_id
            st.session_state["page"] = "Chat"
            st.rerun()
        search = st.text_input("Search", key="conv_search", placeholder="filter conversations…",
                               label_visibility="collapsed")
        conversations = store.list_conversations(limit=60, search=search)
        current = st.session_state.get("conversation_id")
        if not current and conversations:
            current = conversations[0]["id"]
            st.session_state["conversation_id"] = current
        if not conversations:
            conversation_id = store.create_conversation("New conversation")
            st.session_state["conversation_id"] = conversation_id
            current = conversation_id
        for conversation in conversations[:24]:
            is_current = conversation["id"] == current
            label = ("● " if is_current else "") + conversation["title"][:38]
            if st.button(label, key=f"conv_{conversation['id']}", width="stretch",
                         help=f"{conversation['n_messages']} messages · {conversation['id']}"):
                st.session_state["conversation_id"] = conversation["id"]
                st.session_state["page"] = "Chat"
                st.rerun()

        with st.expander("Manage this conversation"):
            title = st.text_input("Title", value=next(
                (c["title"] for c in conversations if c["id"] == current), ""), key="conv_title")
            columns = st.columns(2)
            with columns[0]:
                if st.button("Rename", key="conv_rename", width="stretch") and title:
                    store.rename_conversation(current, title)
                    st.rerun()
            with columns[1]:
                pinned = next((c["pinned"] for c in conversations if c["id"] == current), 0)
                if st.button("Unpin" if pinned else "Pin", key="conv_pin", width="stretch"):
                    store.set_pinned(current, not pinned)
                    st.rerun()
            if st.button("Delete conversation", key="conv_delete", width="stretch"):
                store.delete_conversation(current)
                st.session_state.pop("conversation_id", None)
                st.rerun()

        st.divider()
        st.caption(
            "BioNeMo NIMs are NVIDIA's inference microservices. This console drives them over their "
            "HTTP API, and falls back to its own documented models when no GPU is present — never "
            "silently."
        )

    return st.session_state.get("conversation_id"), effective


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={"about": "Dr-Titan BioNeMo Web Console · local Streamlit front end for NVIDIA BioNeMo NIMs"},
    )
    st.markdown(css(), unsafe_allow_html=True)
    _session_defaults()
    context = _bootstrap()
    conversation_id, backend = _sidebar(context)

    store, queue, service, llm = context["store"], context["queue"], context["service"], context["llm"]
    page = st.session_state["page"]

    if page == "Chat":
        st.markdown("## Chat")
        st.caption(
            "Ask in plain language. The console plans the tool calls, runs them against your "
            "configured backends, and reports exactly which backend answered each one."
        )
        chat_ui.render_chat_page(
            agent=context["agent"], store=store, queue=queue, llm=llm,
            conversation_id=conversation_id, backend=backend,
        )
    elif page == "Structure Lab":
        console_pages.structure_lab(service, store, backend)
    elif page == "Molecule Workbench":
        console_pages.molecule_workbench(service, store, backend)
    elif page == "Screening Pipeline":
        console_pages.screening_page(service, store, queue, backend, conversation_id)
    elif page == "Administration":
        console_pages.administration(service, store, context["settings"], queue)
    elif page == "History":
        console_pages.history_page(store)


if __name__ == "__main__":
    main()
