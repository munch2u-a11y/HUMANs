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
    / "developmental_cortex_nursery.py"
)
SPEC = importlib.util.spec_from_file_location(
    "developmental_cortex_nursery", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
NURSERY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = NURSERY
SPEC.loader.exec_module(NURSERY)


def test_nursery_uses_closed_receipts_and_no_pretrained_language_state(
    tmp_path: Path,
) -> None:
    report = NURSERY.run_nursery(
        database=tmp_path / "nursery.sqlite",
        checkpoints=tmp_path / "checkpoints",
        device="cpu",
        tiny=True,
        cycles=2,
        optimization_steps=2,
    )

    assert report["random_initialization"] is True
    assert report["pretrained_model_loaded"] is False
    assert report["tokenizer_loaded"] is False
    assert report["transcript_window"] is False
    assert report["receipt_backed_action_cycles"] == 8
    assert report["open_action_cycles"] == 0
    assert report["verified_behavior_examples"] == 8
    assert sum(report["actualized_motor_trunks"].values()) == 8
    assert all(count > 0 for count in report["actualized_motor_trunks"].values())
    assert report["motor_selection_modes"][
        "developmental_stochastic_exploration"
    ] > 0
    assert report["motor_selection_modes"]["caregiver_selected_authorized"] == 0
    assert report["graph_invariant_errors"] == []
