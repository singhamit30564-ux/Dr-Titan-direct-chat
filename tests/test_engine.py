"""Agent, router, store, jobs and export — the behaviour the user actually sees."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from bionemo_web.bio import chem
from bionemo_web.engine import router, tools
from bionemo_web.engine.agent import ChatAgent
from bionemo_web.engine.jobs import JobQueue
from bionemo_web.engine.llm import LLMClient, LLMConfig, OfflineAnalyst
from bionemo_web.engine.store import Store
from bionemo_web.nims.client import NimClient
from bionemo_web.nims.service import NimService


@pytest.fixture()
def store(tmp_path) -> Store:
    return Store(path=tmp_path / "test.db")


@pytest.fixture()
def service() -> NimService:
    return NimService(NimClient(), {"prefer_backend": "simulate", "allow_hosted": False,
                                    "simulation_seed": 4})


@pytest.fixture()
def agent(service, store) -> ChatAgent:
    return ChatAgent(service, llm=LLMClient(LLMConfig(provider="off")), store=store)


# ------------------------------------------------------------------ router


def test_router_reads_a_plain_fold_request(sequence):
    plan = router.plan(f"fold {sequence}", llm=None)
    assert [s.tool for s in plan.steps] == ["fold_protein"]
    assert plan.steps[0].args["sequence"] == sequence


def test_router_extracts_a_smiles_and_a_count(aspirin):
    plan = router.plan(f"generate 15 analogues of {aspirin} optimising QED", llm=None)
    assert plan.steps[0].tool == "generate_molecules"
    assert plan.steps[0].args["seed_smiles"] == aspirin
    assert plan.steps[0].args["num_molecules"] == 15
    assert plan.steps[0].args["property_name"] == "QED"


def test_router_does_not_mistake_english_for_chemistry():
    plan = router.plan("what can you do?", llm=None)
    assert plan.chat_only, "prose must not be parsed as a molecule"
    assert plan.steps == []


def test_router_prefers_an_explicit_model_mention():
    plan = router.plan("fold MKTAYIAKQRQISF with alphafold2", llm=None)
    assert plan.steps[0].args["model"] == "alphafold2"


def test_router_slash_command_with_flags():
    plan = router.plan("/molecules --num_molecules 20 CC(=O)Oc1ccccc1C(=O)O", llm=None)
    assert plan.steps[0].tool == "generate_molecules"
    assert plan.steps[0].args["num_molecules"] == 20


def test_router_builds_a_pipeline_for_a_composite_request(sequence, aspirin):
    plan = router.plan(f"virtual screen {aspirin} against {sequence}", llm=None)
    assert plan.steps[0].tool == "virtual_screening"


def test_router_resolves_common_molecule_names():
    """A chemist should be able to type a drug name instead of a SMILES."""
    plan = router.plan("dock aspirin into the last structure", llm=None)
    assert plan.entities["smiles"] == chem.COMMON_MOLECULES["aspirin"]
    assert plan.steps[0].tool == "dock_ligand"


def test_router_never_runs_a_tool_on_prose():
    """Prose that merely mentions a molecule stays conversation.

    A bare drug name is resolved as an entity, but with no analysis or docking
    intent the plan must remain chat-only: answering a question about a drug is
    not the same as running RDKit over it unasked.
    """
    for sentence in [
        "the aspirin of my eye is a fox",
        "is caffeine safe in pregnancy?",
        "my favourite colour is ibuprofen yellow",
    ]:
        plan = router.plan(sentence, llm=None)
        assert plan.chat_only, sentence
        assert plan.steps == [], sentence


def test_property_question_is_analysis_not_generation():
    """"what is the QED of caffeine?" must not silently invent analogues."""
    plan = router.plan("what is the QED of caffeine?", llm=None)
    assert plan.steps[0].tool == "analyse_molecules"


def test_generation_verb_wins_for_analogues():
    plan = router.plan("generate 10 analogues of ibuprofen optimising QED", llm=None)
    assert plan.steps[0].tool == "generate_molecules"
    assert plan.steps[0].args["num_molecules"] == 10


def test_named_molecule_routes_molecule_analysis():
    plan = router.plan("analyse the molecule paracetamol", llm=None)
    assert plan.steps[0].tool == "analyse_molecules"


def test_slash_command_is_not_second_guessed():
    """An explicit command survives disambiguation even with no verb."""
    plan = router.plan("/molecules --num_molecules 20 CCO", llm=None)
    assert plan.steps[0].tool == "generate_molecules"


def test_router_records_what_is_missing():
    plan = router.plan("dock something", llm=None)
    assert plan.clarifications, "an unanswerable request must ask for the missing input"


# ------------------------------------------------------------------- tools


def test_local_sequence_analysis_needs_no_model(service, sequence):
    outcome = tools.execute("analyse_sequence", {"sequence": sequence}, service)
    assert outcome.ok
    assert outcome.renderer == "sequence_report"
    assert outcome.payload["report"]["length"] == len(sequence)
    assert outcome.summary


def test_local_molecule_analysis(service, aspirin):
    outcome = tools.execute("analyse_molecules", {"smiles_list": f"{aspirin}, CCO"}, service)
    assert outcome.ok
    assert len(outcome.payload["reports"]) == 2


def test_tool_reports_an_error_instead_of_raising(service):
    outcome = tools.execute("no_such_tool", {}, service)
    assert outcome.ok is False
    assert outcome.error


def test_generation_tool_produces_a_usable_table(service, aspirin):
    outcome = tools.execute(
        "generate_molecules",
        {"seed_smiles": aspirin, "num_molecules": 5, "property_name": "QED"},
        service,
    )
    assert outcome.ok and outcome.run is not None
    assert outcome.run.meta.get("rows")


# -------------------------------------------------------------- the agent


def test_agent_persists_a_conversation(agent, store, sequence):
    conversation_id = store.create_conversation("test")
    turn = agent.handle_message(conversation_id, f"fold {sequence}", backend="simulate")
    assert turn.kind == "answer"
    assert turn.outcomes and turn.outcomes[0].ok
    messages = store.messages(conversation_id)
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[1]["payload"]["kind"] == "result"
    assert store.runs(conversation_id=conversation_id), "the run must be recorded"


def test_agent_uses_the_previous_structure_for_a_follow_up(agent, store, sequence):
    conversation_id = store.create_conversation("test")
    agent.handle_message(conversation_id, f"fold {sequence}", backend="simulate")
    turn = agent.handle_message(conversation_id, "design 3 sequences for that backbone",
                                backend="simulate")
    assert [s.tool for s in turn.plan.steps] == ["design_sequences"]
    assert turn.outcomes[0].ok, "the follow-up should reuse the structure from the session"
    assert len(turn.outcomes[0].payload["designs"]) == 3


def test_agent_asks_for_missing_input(agent, store):
    conversation_id = store.create_conversation("test")
    turn = agent.handle_message(conversation_id, "dock it", backend="simulate")
    assert turn.kind == "clarification"
    assert "need" in turn.text.lower() or "missing" in turn.text.lower()


def test_agent_answers_a_question_without_running_anything(agent, store):
    conversation_id = store.create_conversation("test")
    turn = agent.handle_message(conversation_id, "what can you do?", backend="simulate")
    assert turn.kind == "chat"
    assert turn.narration_source == "offline"
    assert "fold" in turn.text.lower()


def test_offline_narration_never_claims_a_real_prediction(agent, store, sequence):
    conversation_id = store.create_conversation("test")
    turn = agent.handle_message(conversation_id, f"fold {sequence}", backend="simulate")
    assert "simulat" in turn.text.lower(), "the summary must state that the result is simulated"


# -------------------------------------------------------- store and export


def test_store_records_provenance(service, store, sequence):
    result = service.run("esmfold", {"sequence": sequence})
    run_id = store.record_run(
        model_key=result.spec_key, backend=result.backend, ok=result.ok,
        simulated=result.simulated, latency_ms=result.latency_ms, summary="test",
        provenance=result.provenance(),
    )
    run = store.run(run_id)
    assert run and run["model_key"] == "esmfold"
    assert run["provenance"]["simulated"] is True


def test_store_search_and_pin(store):
    a = store.create_conversation("alpha")
    store.create_conversation("beta")
    assert [c["title"] for c in store.list_conversations(search="alph")] == ["alpha"]
    store.set_pinned(a, True)
    assert store.list_conversations()[0]["title"] == "alpha"


def test_conversation_export_round_trips(agent, store, sequence):
    conversation_id = store.create_conversation("export me")
    agent.handle_message(conversation_id, f"fold {sequence}", backend="simulate")
    exported = store.export_conversation(conversation_id)
    assert json.dumps(exported, default=str)
    assert len(exported["messages"]) == 2


def test_export_writes_files(tmp_path, service, aspirin):
    from bionemo_web.engine import export

    outcome = tools.execute(
        "generate_molecules", {"seed_smiles": aspirin, "num_molecules": 4}, service
    )
    files = export.outcome_files(outcome, run_id="testexport")
    kinds = {record["kind"] for record in files}
    assert "csv" in kinds and "sdf" in kinds and "json" in kinds
    for record in files:
        path = Path(record["path"])
        assert path.exists() and path.stat().st_size > 0
    csv_text = next(Path(r["path"]).read_text() for r in files if r["kind"] == "csv")
    assert "smiles" in csv_text.splitlines()[0].lower()


def test_screening_report_mentions_the_weighting(service, aspirin, sequence):
    from bionemo_web.engine import export

    outcome = tools.execute(
        "virtual_screening",
        {"protein": sequence, "seed_smiles": aspirin, "num_molecules": 4, "num_poses": 2},
        service,
    )
    assert outcome.ok
    report = export.screening_report(outcome)
    assert "composite" in report.lower()
    assert "provenance" in report.lower()


def test_curl_replay_is_generated_for_a_real_call():
    from bionemo_web.engine.tools import ToolOutcome
    from bionemo_web.nims.service import RunResult

    run = RunResult(
        ok=True, spec_key="molmim", backend="local", method="test",
        request_payload={"smi": "CCO", "num_molecules": 5},
        url="http://localhost:8000/v1/biology/nvidia/molmim/generate",
    )
    outcome = ToolOutcome(tool="generate_molecules", runner="molmim via local", ok=True,
                          title="t", renderer="molecule_table", run=run)
    from bionemo_web.engine import export

    curl = export.nim_request_curl(outcome)
    assert curl.startswith("curl") and "localhost:8000" in curl


# ------------------------------------------------------------------- jobs


def test_jobs_run_in_the_background_and_report_progress(tmp_path):
    queue = JobQueue(max_workers=1)

    def work(ctx):
        ctx.progress(0.5, "halfway")
        return {"answer": 42}

    job_id = queue.submit("test", "unit job", work)
    result = queue.wait(job_id, timeout=30)
    assert result["status"] == "done"
    assert result["progress"] == pytest.approx(1.0)
    assert queue.get(job_id).result == {"answer": 42}


def test_jobs_survive_a_failing_function(tmp_path):
    queue = JobQueue(max_workers=1)

    def boom(ctx):
        raise RuntimeError("expected failure")

    job_id = queue.submit("test", "failing job", boom)
    result = queue.wait(job_id, timeout=30)
    assert result["status"] == "failed"
    assert "expected failure" in result["error"]


def test_jobs_can_be_cancelled():
    queue = JobQueue(max_workers=1)

    def slow(ctx):
        for _ in range(100):
            ctx.check_cancelled()
            time.sleep(0.05)
        return "finished"

    job_id = queue.submit("test", "slow job", slow)
    time.sleep(0.3)
    assert queue.cancel(job_id)
    result = queue.wait(job_id, timeout=30)
    assert result["status"] == "cancelled"


# --------------------------------------------------------- offline analyst


def test_offline_analyst_describes_a_fold(service, sequence):
    result = service.run("esmfold", {"sequence": sequence})
    text = OfflineAnalyst.summarise_run(result)
    assert "residues" in text
    assert "simulat" in text.lower()


def test_offline_analyst_handles_a_failure(service):
    result = service.run("does-not-exist", {})
    assert "did not produce" in OfflineAnalyst.summarise_run(result)


# ----------------------------------------------------------- artifact index

def test_artifact_registration_is_idempotent(tmp_path):
    """Re-rendering a finished run must not duplicate its files in History.

    Streamlit reruns the whole script on every interaction, so the same export is
    usually registered several times. Registration is keyed by path, so the second
    (and fourth) call must return the original id rather than adding a row.
    """
    store = Store(path=tmp_path / "console.db")
    artifact = tmp_path / "pose.pdb"
    artifact.write_text("ATOM\n")

    kwargs = dict(kind="pdb", name="pose.pdb", path=str(artifact), run_id="r1")
    first = store.add_artifact(**kwargs)
    again = store.add_artifact(**kwargs)

    assert first == again
    rows = store.artifacts()
    assert len(rows) == 1
    assert rows[0]["path"] == str(artifact)
    assert rows[0]["size"] == len("ATOM\n")

    # A second, genuinely different file is still recorded separately.
    other = tmp_path / "ligand.sdf"
    other.write_text("C\n")
    store.add_artifact(kind="sdf", name="ligand.sdf", path=str(other), run_id="r1")
    assert len(store.artifacts()) == 2


def test_store_upgrade_drops_duplicate_artifact_rows(tmp_path):
    """An index written before the fix is cleaned up when the store opens."""
    db = tmp_path / "console.db"
    store = Store(path=db)
    artifact = tmp_path / "report.json"
    artifact.write_text("{}")
    store.add_artifact(kind="json", name="report.json", path=str(artifact))
    # Simulate the old behaviour: the same path indexed three more times.
    for n in range(3):
        store.execute(
            "INSERT INTO artifacts (id, run_id, conversation_id, kind, name, path, size, created_at, meta)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (f"legacy{n}", "", "", "json", "report.json", str(artifact), 2, 1.0, "{}"),
        )
    assert len(store.artifacts()) == 4

    reopened = Store(path=db)
    assert len(reopened.artifacts()) == 1


def test_digit_tokens_are_never_mistaken_for_smiles():
    """The "10" in "generate 10 analogues" is a count, not a molecule.

    RDKit answers a nonsense candidate with a parse error on stderr; users read
    that as the app crashing. Counts must be filtered before validation.
    """
    plan = router.plan("generate 10 analogues of ibuprofen optimising QED", llm=None)
    assert plan.entities["smiles"] == chem.COMMON_MOLECULES["ibuprofen"]
    assert plan.entities["smiles"] != "10"


def test_plain_english_never_yields_a_smiles():
    for text in ("the quick brown fox", "what is the pI of the backbone?",
                 "fold 8 structures", "screen 25 candidates against MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"):
        plan = router.plan(text, llm=None)
        assert plan.entities.get("smiles") is None, text


def test_letters_that_spell_elements_do_not_become_molecules():
    """Words like "in", "as" or "co" are element symbols only by accident."""
    for word in ("Co", "In", "As", "Sn", "He", "Noble"):
        plan = router.plan(f"analyse the {word} value for me", llm=None)
        assert plan.entities.get("smiles") is None, word
