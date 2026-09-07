from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


pytest.importorskip("torch")

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "graph_native_live"
    / "developmental_communication_nursery.py"
)
SPEC = importlib.util.spec_from_file_location(
    "developmental_communication_nursery", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
NURSERY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = NURSERY
SPEC.loader.exec_module(NURSERY)


def test_agent_message_causes_held_out_listener_action_without_prompt_text(
    tmp_path: Path,
) -> None:
    report = NURSERY.run_communication_nursery(
        database=tmp_path / "communication.sqlite",
        checkpoints=tmp_path / "checkpoints",
        device="cpu",
        tiny=True,
        state_count=3,
        exposures=3,
        rehearsal_steps=20,
        bootstrap_rounds=1,
        behavior_steps=20,
        held_out_rounds=1,
    )

    assert sum(item["success"] for item in report["cold_trials"]) == 0
    assert report["motor_form_count"] == 3
    assert report["bootstrap_successes"] == 3
    assert report["held_out_trial_count"] == 3
    assert report["held_out_emission_rate"] == 1.0
    assert report["held_out_communication_accuracy"] == 1.0
    assert report["held_out_graph_only_accuracy"] == 1.0
    assert report["held_out_coupled_accuracy"] == 1.0
    assert report["dominant_speak_rate"] == 1.0
    assert report["all_outputs_self_generated"] is True
    assert report["all_outputs_receipted"] is True
    assert report["checkpoint_restart_match"] is True
    assert report["pretrained_model_loaded"] is False
    assert report["tokenizer_loaded"] is False
    assert report["transcript_window"] is False
    assert report["runtime_memory_retrieval_for_speech"] is False
    assert report["open_action_cycles"] == 0
    assert report["graph_invariant_errors"] == []
    for trial in report["held_out_trials"]:
        assert trial["output_payload_sha256"] == trial["expected_payload_sha256"]
        assert trial["generation_mode"] == "promoted_form_competition"
        assert trial["stop_source"] == "promoted_whole_form"
        assert trial["candidate_count"] == 3
        assert trial["output_record_id"]
        assert trial["return_record_id"]
        assert trial["internal_event_ids"]
