from __future__ import annotations

from pathlib import Path

import pytest


pytest.importorskip("torch")

from habitus_ai.developmental_cortex import CortexConfig
from habitus_ai.developmental_runtime import BornInHabitusRuntime
from habitus_ai.gestation import gestate, load_profile
from habitus_ai.integrated_agent import (
    MEMORY_COMMIT_ABILITY,
    MEMORY_RECALL_ABILITY,
    WORKSPACE_READ_ABILITY,
    WORKSPACE_RUN_ABILITY,
    IntegratedMind,
)
from habitus_ai.types import InputTrunk, OutputTrunk, RecordType


class RecordingModel:
    def __init__(self) -> None:
        self.calls: list[list[dict[str, str]]] = []

    def generate(self, messages):
        copied = [dict(message) for message in messages]
        self.calls.append(copied)
        return f"surface reply to: {copied[-1]['content']}"


def _config() -> CortexConfig:
    return CortexConfig(
        byte_embedding_width=16,
        graph_width=16,
        direction_width=4,
        hidden_width=32,
        recurrent_layers=1,
        maximum_event_bytes=64,
        seed=71,
    )


def _runtime(database: Path, checkpoints: Path) -> BornInHabitusRuntime:
    return BornInHabitusRuntime(
        database,
        cortex_config=_config(),
        checkpoint_directory=checkpoints,
        device="cpu",
        mind_dimension=32,
    )


def _profile(runtime: BornInHabitusRuntime):
    profile = load_profile(runtime.mind)
    if profile is None:
        profile = gestate(
            runtime.mind,
            human_name="Nemo",
            agent_name="Mira",
            taste_schema="builder",
            model_backend="test",
            model_name="recording-model",
        )
    return profile


def test_current_event_renderer_has_no_transcript_or_recalled_text(tmp_path: Path) -> None:
    database = tmp_path / "mind.sqlite"
    checkpoints = tmp_path / "checkpoints"
    model = RecordingModel()
    with _runtime(database, checkpoints) as runtime:
        agent = IntegratedMind(
            runtime,
            model,
            workspace=tmp_path,
            profile=_profile(runtime),
        )
        first = agent.handle("PRIVATE_FIRST_EVENT_947")
        first_cortex_state = runtime.cortex.latest_pulse_state()
        second = agent.handle("How are you right now?")

        assert first.kind == "conversation"
        assert second.kind == "conversation"
        assert first.speech_cycle_id is not None
        assert second.speech_cycle_id is not None
        assert len(model.calls) == 2
        assert model.calls[1][-1]["content"] == "How are you right now?"
        assert "PRIVATE_FIRST_EVENT_947" not in "\n".join(
            message["content"] for message in model.calls[1]
        )
        assert "dominant_drive=" in model.calls[1][0]["content"]
        assert len(runtime.mind.open_experience_cycles(OutputTrunk.SPEAK)) == 1
        assert runtime.cortex.latest_pulse_state().state_sha256 != (
            first_cortex_state.state_sha256
        )
        output = runtime.mind.store.get_record(
            runtime.mind.open_experience_cycles(OutputTrunk.SPEAK)[0].output_record_id
        )
        assert output.metadata["transcript_records_used"] == 0
        assert output.metadata["recalled_records_used"] == 0
        assert output.metadata["automatic_text_retrieval"] is False
        assert output.metadata["developmental_self_generated"] is False
        state = agent.state()
        assert state["memory"]["transcript_window"] is False
        assert state["memory"]["automatic_text_retrieval"] is False
        assert len(state["desires"]) == 9
        assert state["graph_invariant_errors"] == []


def test_workspace_actions_require_exact_self_affordance_and_cortex_return(
    tmp_path: Path,
) -> None:
    (tmp_path / "note.txt").write_text("INTEGRATED_OPEN_OK\n", encoding="utf-8")
    (tmp_path / "probe.py").write_text(
        "print('INTEGRATED_RUN_OK')\n", encoding="utf-8"
    )
    model = RecordingModel()
    with _runtime(tmp_path / "tools.sqlite", tmp_path / "tool-checkpoints") as runtime:
        agent = IntegratedMind(
            runtime,
            model,
            workspace=tmp_path,
            profile=_profile(runtime),
        )
        opened = agent.handle("/open note.txt")
        assert opened.tool_receipt is not None
        assert opened.tool_receipt.tool_id == WORKSPACE_READ_ABILITY
        assert opened.tool_receipt.verified is True
        assert "INTEGRATED_OPEN_OK" in opened.response
        read_call = runtime.mind.store.get_record(opened.tool_receipt.output_record_id)
        read_return = runtime.mind.store.get_record(opened.tool_receipt.return_record_id)
        assert read_call.metadata["authorized_output_node_id"] == WORKSPACE_READ_ABILITY
        assert read_return.metadata["causal_trunk"] == InputTrunk.SEE.value
        assert read_return.metadata["membrane_words"] is False
        read_cycle = runtime.mind.experience_cycle(opened.tool_receipt.cycle_id)
        credited_nodes = {
            node_id
            for edge_id in read_cycle.credited_edge_ids
            for edge in (runtime.mind.store.get_edge(edge_id),)
            if edge is not None
            for node_id in (edge.source_id, edge.target_id)
        }
        motive_ids = {
            node_id
            for group in agent.foundation.values()
            for node_id in group.values()
        }
        credited_motives = credited_nodes & motive_ids
        assert credited_motives
        assert any(
            runtime.mind.store.get_node_dynamics(node_id).valence > 0.0
            for node_id in credited_motives
        )

        ran = agent.handle("/run probe.py")
        assert ran.tool_receipt is not None
        assert ran.tool_receipt.tool_id == WORKSPACE_RUN_ABILITY
        assert ran.tool_receipt.verified is True
        assert "INTEGRATED_RUN_OK" in ran.response
        run_call = runtime.mind.store.get_record(ran.tool_receipt.output_record_id)
        run_return = runtime.mind.store.get_record(ran.tool_receipt.return_record_id)
        assert run_call.metadata["authorized_output_node_id"] == WORKSPACE_RUN_ABILITY
        assert run_return.metadata["causal_trunk"] == InputTrunk.NOTICE.value
        assert run_return.metadata["membrane_words"] is False
        assert runtime.cortex.latest_pulse_state().pulse == ran.pulse
        assert model.calls == []
        assert runtime.mind.graph.validate_invariants() == []


def test_explicit_memory_is_pulse_selected_and_survives_restart(tmp_path: Path) -> None:
    database = tmp_path / "memory.sqlite"
    checkpoints = tmp_path / "memory-checkpoints"
    model = RecordingModel()
    with _runtime(database, checkpoints) as runtime:
        agent = IntegratedMind(
            runtime,
            model,
            workspace=tmp_path,
            profile=_profile(runtime),
        )
        remembered = agent.handle("remember that my launch color is ultraviolet")
        assert remembered.tool_receipt is not None
        assert remembered.tool_receipt.tool_id == MEMORY_COMMIT_ABILITY
        assert remembered.tool_receipt.verified is True
        assert remembered.evidence_record_ids
        fact = runtime.mind.store.get_record(remembered.evidence_record_ids[0])
        assert fact.record_type == RecordType.FACT
        assert fact.metadata["automatic_prompt_injection"] is False
        persisted_state = runtime.cortex.latest_pulse_state().state_sha256

    reopened_model = RecordingModel()
    with _runtime(database, checkpoints) as runtime:
        assert runtime.cortex.latest_pulse_state().state_sha256 == persisted_state
        agent = IntegratedMind(
            runtime,
            reopened_model,
            workspace=tmp_path,
            profile=_profile(runtime),
        )
        recalled = agent.handle("/recall launch color")
        assert recalled.tool_receipt is not None
        assert recalled.tool_receipt.tool_id == MEMORY_RECALL_ABILITY
        assert recalled.tool_receipt.verified is True
        assert "launch color is ultraviolet" in recalled.response
        assert fact.record_id in recalled.evidence_record_ids
        recall_return = runtime.mind.store.get_record(
            recalled.tool_receipt.return_record_id
        )
        assert recall_return.metadata["membrane_words"] is False
        assert recall_return.metadata["causal_trunk"] == InputTrunk.SEE.value
        assert reopened_model.calls == []
        assert runtime.mind.graph.validate_invariants() == []
