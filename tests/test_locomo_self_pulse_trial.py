from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


EXPERIMENT = Path(__file__).resolve().parents[1] / "experiments" / "graph_native_live"
for import_root in (Path(__file__).resolve().parents[1] / "src", EXPERIMENT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from habitus_ai import DevelopmentalGrowth, InputTrunk

MODULE_PATH = EXPERIMENT / "locomo_self_pulse_trial.py"
SPEC = importlib.util.spec_from_file_location("locomo_self_pulse_trial", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
TRIAL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TRIAL
SPEC.loader.exec_module(TRIAL)


def test_session_turns_preserve_chronology_speakers_and_modal_metadata() -> None:
    sample = {
        "conversation": {
            "session_1_date_time": "2026-09-01 10:00",
            "session_1": [
                {"speaker": "A", "text": "hello", "dia_id": "D1:1"},
                {
                    "speaker": "B",
                    "text": "look",
                    "dia_id": "D1:2",
                    "blip_caption": "a red kite",
                },
            ],
            "session_2_date_time": "2026-09-02 11:00",
            "session_2": [
                {"speaker": "A", "text": "again", "dia_id": "D2:1"}
            ],
        }
    }

    turns = TRIAL._session_turns(sample, 2)

    assert [turn["dia_id"] for turn in turns] == ["D1:1", "D1:2", "D2:1"]
    assert [turn["speaker"] for turn in turns] == ["A", "B", "A"]
    assert turns[1]["blip_caption"] == "a red kite"
    assert turns[2]["session"] == 2
    assert turns[2]["sequence"] == 0
    assert turns[2]["session_date_time"] == "2026-09-02 11:00"


def test_questions_remain_ineligible_until_every_evidence_turn_was_lived() -> None:
    sample = {
        "qa": [
            {
                "question": "Known?",
                "answer": "yes",
                "category": 1,
                "evidence": ["D1:1"],
            },
            {
                "question": "Needs future?",
                "answer": "no",
                "category": 1,
                "evidence": ["D1:1", "D2:1"],
            },
            {
                "question": "Wrong category?",
                "answer": "x",
                "category": 2,
                "evidence": ["D1:1"],
            },
        ]
    }

    questions = TRIAL._eligible_questions(
        sample, {"D1:1"}, maximum=10, categories=(1,)
    )

    assert [question["question"] for question in questions] == ["Known?"]


def test_answer_score_does_not_reward_fluent_irrelevance() -> None:
    wrong = TRIAL._answer_score(
        "I would be delighted to help with that question.",
        "transgender woman",
    )
    partial = TRIAL._answer_score("Caroline is a woman.", "transgender woman")

    assert wrong == {
        "exact": False,
        "contains_reference": False,
        "token_precision": 0.0,
        "token_recall": 0.0,
        "token_f1": 0.0,
    }
    assert partial["exact"] is False
    assert partial["contains_reference"] is False
    assert partial["token_recall"] == pytest.approx(0.5)


def test_turn_metadata_marks_replay_as_observation_not_generated_speech() -> None:
    metadata = TRIAL._turn_metadata(
        {
            "dia_id": "D3:4",
            "session": 3,
            "sequence": 3,
            "session_date_time": "tomorrow is not injected as a fact",
        },
        modality="utterance",
    )

    assert metadata["locomo_dia_id"] == "D3:4"
    assert metadata["locomo_modality"] == "utterance"
    assert metadata["developmental_replay"] is True
    assert metadata["teacher_forced"] is False


def test_growth_receipt_is_json_safe_and_preserves_its_causal_trunks() -> None:
    growth = DevelopmentalGrowth(
        record_ids=("record:1", "record:2"),
        input_trunks=(InputTrunk.SEE, InputTrunk.HEAR),
        feature_node_ids=("fiber:1", "fiber:2"),
        candidate_pattern_node_ids=("candidate:1",),
        promoted_pattern_node_ids=(),
        active_context_node_ids=("candidate:1",),
        edge_ids=("edge:1", "edge:2"),
        cross_trunk=True,
    )

    row = TRIAL._growth_row(growth)

    assert row["input_trunks"] == ["SEE", "HEAR"]
    assert row["cross_trunk"] is True
    assert json.loads(json.dumps(row)) == row
