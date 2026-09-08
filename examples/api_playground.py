#!/usr/bin/env python3
"""Run the complete integrated loop offline with a tiny cortex and fake voice.

This is a disposable developer example, not the default 18M-parameter product
configuration. It needs PyTorch but does not need Ollama, a model download, a
GPU, network access, or an existing memory database.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Sequence

from habitus_ai.developmental_cortex import CortexConfig
from habitus_ai.developmental_runtime import BornInHabitusRuntime
from habitus_ai.gestation import gestate
from habitus_ai.integrated_agent import IntegratedMind, IntegratedTurn


class EditableSpeechMotor:
    """Minimal current-event renderer intended to be replaced by developers."""

    def generate(self, messages) -> str:
        current_event = str(messages[-1]["content"])
        return f"I received the current event: {current_event}"


def tiny_config() -> CortexConfig:
    return CortexConfig(
        byte_embedding_width=16,
        graph_width=16,
        direction_width=4,
        hidden_width=32,
        recurrent_layers=1,
        maximum_event_bytes=64,
        seed=71,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="print complete turn envelopes instead of compact receipts",
    )
    return parser


def _compact_turn(message: str, turn: IntegratedTurn) -> dict[str, object]:
    receipt = turn.tool_receipt
    return {
        "input": message,
        "kind": turn.kind,
        "pulse": turn.pulse,
        "selected_outputs": list(turn.selected_outputs),
        "response_preview": turn.response[:180],
        "evidence_record_ids": list(turn.evidence_record_ids),
        "tool_receipt": (
            {
                "tool_id": receipt.tool_id,
                "status": receipt.status,
                "verified": receipt.verified,
                "output_record_id": receipt.output_record_id,
                "return_record_id": receipt.return_record_id,
            }
            if receipt is not None
            else None
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    with TemporaryDirectory(prefix="habitus-playground-") as temporary:
        root = Path(temporary)
        workspace = root / "workspace"
        workspace.mkdir()
        (workspace / "note.txt").write_text(
            "The playground signal is ultraviolet.\n",
            encoding="utf-8",
        )
        (workspace / "probe.py").write_text(
            "print('PLAYGROUND_RUN_OK')\n",
            encoding="utf-8",
        )

        with BornInHabitusRuntime(
            root / "mind.sqlite",
            cortex_config=tiny_config(),
            checkpoint_directory=root / "checkpoints",
            device="cpu",
            mind_dimension=32,
        ) as runtime:
            profile = gestate(
                runtime.mind,
                human_name="Developer",
                agent_name="Playground",
                taste_schema="builder",
                model_backend="example",
                model_name="editable-speech-motor",
            )
            mind = IntegratedMind(
                runtime,
                EditableSpeechMotor(),
                workspace=workspace,
                profile=profile,
            )
            messages = (
                "Hello from an offline integration test.",
                "remember that the launch color is ultraviolet",
                "/recall launch color",
                "/open .",
                "/open note.txt",
                "/run probe.py",
            )
            turns = []
            for message in messages:
                turn = mind.handle(message)
                turns.append(turn)
                payload = (
                    {"input": message, **turn.to_dict()}
                    if args.full
                    else _compact_turn(message, turn)
                )
                print(json.dumps(payload, sort_keys=True, default=str))

            state = mind.state()
            expected_tools = (
                "ability:memory-commit",
                "ability:memory-recall",
                "ability:workspace-read",
                "ability:workspace-read",
                "ability:workspace-run-python",
            )
            observed_tools = tuple(
                turn.tool_receipt.tool_id
                for turn in turns
                if turn.tool_receipt is not None
            )
            verified = all(
                turn.tool_receipt.verified
                for turn in turns
                if turn.tool_receipt is not None
            )
            if observed_tools != expected_tools or not verified:
                raise RuntimeError("playground ability receipt verification failed")
            if "launch color is ultraviolet" not in turns[2].response:
                raise RuntimeError("playground explicit recall verification failed")
            if "Opened folder ." not in turns[3].response:
                raise RuntimeError("playground workspace folder verification failed")
            if "playground signal is ultraviolet" not in turns[4].response.casefold():
                raise RuntimeError("playground workspace read verification failed")
            if "PLAYGROUND_RUN_OK" not in turns[5].response:
                raise RuntimeError("playground workspace run verification failed")
            if state["graph_invariant_errors"]:
                raise RuntimeError("playground graph invariant verification failed")

            final_state = state if args.full else {
                "pulse": state["pulse"],
                "cortex": state["cortex"],
                "memory": state["memory"],
                "registered_abilities": state["registered_abilities"],
                "verified_tool_returns": state["verified_tool_returns"],
                "graph_invariant_errors": state["graph_invariant_errors"],
            }
            print(json.dumps({"final_state": final_state}, sort_keys=True))

    print("Playground completed; its temporary mind was removed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
