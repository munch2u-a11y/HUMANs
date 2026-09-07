from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from habitus_ai import (
    ACTION_OPPORTUNITY_ORDER,
    BaseAgenticMemoryRAG,
    DeterministicHashEmbedder,
    GraphSide,
    InputTrunk,
    OutputTrunk,
    SelfPulseKernel,
)
from habitus_ai.open_weight import (
    NoContextPlanner,
    SPEECH_START_KIND,
    SPEECH_STOP_KIND,
)
from habitus_ai.types import ConceptNode, as_tuple


def _mind(path: Path) -> BaseAgenticMemoryRAG:
    return BaseAgenticMemoryRAG(path, embedder=DeterministicHashEmbedder(32))


def test_three_senses_collapse_in_order_under_one_durable_self_pulse(
    tmp_path: Path,
) -> None:
    database = tmp_path / "three-senses.sqlite"
    with _mind(database) as mind:
        concepts = {
            InputTrunk.NOTICE: "felt:notification",
            InputTrunk.SEE: "seen:workspace",
            InputTrunk.HEAR: "heard:person",
        }
        for lane, node_id in concepts.items():
            mind.add_concept(node_id, node_id, input_trunks=(lane,))
        kernel = SelfPulseKernel(mind, maximum_bucket_tokens=8)

        for lane in (InputTrunk.HEAR, InputTrunk.SEE, InputTrunk.NOTICE):
            kernel.enqueue_input(
                lane.value,
                lane=lane,
                concept_ids=(concepts[lane],),
                token_cost=1,
            )

        cycle = kernel.advance_cycle(cycle_id="self-cycle:one")
        assert cycle is not None
        assert mind.pulse == 1
        assert cycle.sensory.settled_order == (
            InputTrunk.NOTICE,
            InputTrunk.SEE,
            InputTrunk.HEAR,
        )
        assert [item.pulse for item in cycle.sensory.receipts] == [1, 1, 1]
        assert tuple(item.trunk for item in cycle.output_opportunities) == (
            ACTION_OPPORTUNITY_ORDER
        )
        for receipt in cycle.sensory.receipts:
            assert receipt.inward_traces
            assert all(trace.target_node_id == "SELF" for trace in receipt.inward_traces)
            assert all(trace.path_node_ids[-1] == "SELF" for trace in receipt.inward_traces)

        payload = kernel.latest_state_payload()
        assert payload is not None
        assert payload["pulse"] == 1
        assert payload["settled_order"] == ["NOTICE", "SEE", "HEAR"]
        assert [item["record_id"] for item in payload["inputs"]] == list(
            cycle.self_state.input_record_ids
        )
        persisted = mind.store.connection.execute(
            "SELECT state_json, state_sha256 FROM self_pulse_states WHERE pulse = 1"
        ).fetchone()
        assert persisted is not None
        assert hashlib.sha256(persisted["state_json"].encode()).hexdigest() == persisted[
            "state_sha256"
        ]

    with _mind(database) as reopened:
        payload = SelfPulseKernel(reopened).latest_state_payload()
        assert payload is not None
        assert payload["cycle_id"] == "self-cycle:one"
        assert payload["recurrent_nodes"]


def test_malformed_later_lane_is_rejected_before_a_pulse_is_reserved(
    tmp_path: Path,
) -> None:
    with _mind(tmp_path / "preflight.sqlite") as mind:
        mind.add_concept(
            "seen:valid", "seen:valid", input_trunks=(InputTrunk.SEE,)
        )
        kernel = SelfPulseKernel(mind)
        kernel.enqueue_input(
            "valid", lane=InputTrunk.SEE, concept_ids=("seen:valid",)
        )
        kernel.enqueue_input(
            "invalid", lane=InputTrunk.HEAR, concept_ids=("missing:concept",)
        )

        with pytest.raises(KeyError, match="unknown sensory concept"):
            kernel.advance_cycle()

        assert mind.pulse == 0
        assert len(kernel.pending()) == 2
        assert mind.store.connection.execute(
            "SELECT COUNT(*) FROM records"
        ).fetchone()[0] == 0


def test_failed_reconciliation_rolls_back_recurrent_state_and_self_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _mind(tmp_path / "atomic-state.sqlite") as mind:
        mind.add_concept(
            "heard:valid", "heard:valid", input_trunks=(InputTrunk.HEAR,)
        )
        mind.recurrent.register("heard:valid", persistence=0.9)
        before = mind.recurrent.snapshot(pulse=mind.pulse).state_sha256
        kernel = SelfPulseKernel(mind)
        bucket = kernel.enqueue_input(
            "valid", lane=InputTrunk.HEAR, concept_ids=("heard:valid",)
        )

        def fail_after_recurrent_transition(*_args, **_kwargs):
            raise RuntimeError("injected output valuation failure")

        monkeypatch.setattr(kernel, "_rank_outputs", fail_after_recurrent_transition)
        with pytest.raises(RuntimeError, match="injected output valuation failure"):
            kernel.advance_cycle(item_ids=(bucket.item_id,))

        assert mind.pulse == 1
        assert mind.recurrent.snapshot(pulse=mind.pulse).state_sha256 == before
        assert mind.store.connection.execute(
            "SELECT COUNT(*) FROM recurrent_pulses"
        ).fetchone()[0] == 0
        assert mind.store.connection.execute(
            "SELECT COUNT(*) FROM self_pulse_states"
        ).fetchone()[0] == 0
        row = mind.store.connection.execute(
            "SELECT status, error FROM self_pulse_inbox WHERE item_id = ?",
            (bucket.item_id,),
        ).fetchone()
        assert row["status"] == "failed"
        assert "injected output valuation failure" in row["error"]


def test_pulse_extension_state_commits_and_rolls_back_with_self(tmp_path: Path) -> None:
    with _mind(tmp_path / "extension-atomic.sqlite") as mind:
        mind.add_concept(
            "heard:extension", "heard:extension", input_trunks=(InputTrunk.HEAR,)
        )
        kernel = SelfPulseKernel(mind)
        mind.store.connection.execute(
            "CREATE TABLE extension_state(pulse INTEGER PRIMARY KEY, value TEXT NOT NULL)"
        )
        first = kernel.enqueue_input(
            "first", lane=InputTrunk.HEAR, concept_ids=("heard:extension",)
        )

        def commit_extension(connection, frame):
            connection.execute(
                "INSERT INTO extension_state(pulse, value) VALUES (?, ?)",
                (frame.pulse, frame.input_records[0].record_id),
            )
            return {"test_extension": {"pulse": frame.pulse}}

        receipt = kernel.advance_cycle(
            item_ids=(first.item_id,), pulse_state_observer=commit_extension
        )
        assert receipt is not None
        assert mind.store.connection.execute(
            "SELECT value FROM extension_state WHERE pulse = 1"
        ).fetchone()[0] == receipt.self_state.input_record_ids[0]
        assert kernel.latest_state_payload()["extensions"]["test_extension"] == {
            "pulse": 1
        }

        second = kernel.enqueue_input(
            "second", lane=InputTrunk.HEAR, concept_ids=("heard:extension",)
        )

        def fail_extension(connection, frame):
            connection.execute(
                "INSERT INTO extension_state(pulse, value) VALUES (?, 'transient')",
                (frame.pulse,),
            )
            raise RuntimeError("extension failed after write")

        with pytest.raises(RuntimeError, match="extension failed after write"):
            kernel.advance_cycle(
                item_ids=(second.item_id,), pulse_state_observer=fail_extension
            )
        assert mind.store.connection.execute(
            "SELECT COUNT(*) FROM extension_state WHERE pulse = 2"
        ).fetchone()[0] == 0
        assert mind.store.connection.execute(
            "SELECT COUNT(*) FROM self_pulse_states WHERE pulse = 2"
        ).fetchone()[0] == 0


def _add_ability(
    mind: BaseAgenticMemoryRAG,
    node_id: str,
    *,
    input_trunk: InputTrunk = InputTrunk.HEAR,
    output_trunk: OutputTrunk = OutputTrunk.SPEAK,
) -> None:
    mind.add_concept(
        node_id,
        node_id,
        input_trunks=(input_trunk,),
        output_trunks=(output_trunk,),
        kind="ability",
    )
    mind.recurrent.register(node_id, persistence=0.9)


def _teach_return(
    mind: BaseAgenticMemoryRAG,
    node_id: str,
    context_id: str,
    value: float,
) -> None:
    decision = mind.classify_output(
        node_id,
        target_concept_id=node_id,
        required_output_trunk=OutputTrunk.SPEAK,
    )
    cycle = mind.begin_output_cycle(
        node_id,
        decision,
        metadata={"semantic_node_ids": [context_id]},
    )
    mind.record_cycle_return(
        cycle.cycle_id,
        "verified consequence",
        input_trunk=InputTrunk.HEAR,
        status="observed",
        stability_delta=value,
        verified=True,
        allow_growth=False,
        embedding=[0.0] * mind.embedder.dimension,
    )


def test_output_value_is_context_conditioned_and_authorization_is_one_use(
    tmp_path: Path,
) -> None:
    with _mind(tmp_path / "conditioned-output.sqlite") as mind:
        for context_id in ("context:x", "context:y"):
            mind.add_concept(
                context_id, context_id, input_trunks=(InputTrunk.HEAR,)
            )
        for node_id in ("answer:a", "answer:b"):
            _add_ability(mind, node_id)

        _teach_return(mind, "answer:a", "context:x", 0.9)
        _teach_return(mind, "answer:a", "context:y", -0.9)
        _teach_return(mind, "answer:b", "context:x", -0.9)
        _teach_return(mind, "answer:b", "context:y", 0.9)

        kernel = SelfPulseKernel(mind)
        kernel.enqueue_input(
            "current x situation",
            lane=InputTrunk.HEAR,
            concept_scores={
                "context:x": 1.0,
                "answer:a": 0.75,
                "answer:b": 0.75,
            },
            token_cost=2,
        )
        cycle = kernel.advance_cycle()
        assert cycle is not None
        by_node = {item.node_id: item for item in cycle.output_candidates}
        assert by_node["answer:a"].expected_stability_delta > 0.0
        assert by_node["answer:b"].expected_stability_delta < 0.0
        assert cycle.selected_output is not None
        assert cycle.selected_output.node_id == "answer:a"

        pulse_before_output = mind.pulse
        output = kernel.actualize_output(
            cycle.selected_output,
            "context-sensitive response",
            metadata={"semantic_node_ids": ["context:x"]},
        )
        assert mind.pulse == pulse_before_output
        authorization = mind.store.connection.execute(
            """SELECT status, cycle_id FROM self_output_authorizations
               WHERE authorization_id = ?""",
            (cycle.selected_output.authorization_id,),
        ).fetchone()
        assert authorization["status"] == "consumed"
        assert authorization["cycle_id"] == output.cycle_id
        with pytest.raises(ValueError, match="already consumed"):
            kernel.actualize_output(
                cycle.selected_output, "cannot externalize this route twice"
            )


def test_all_motor_lanes_actualize_and_their_returns_share_the_next_pulse(
    tmp_path: Path,
) -> None:
    with _mind(tmp_path / "motor-return-frame.sqlite") as mind:
        paths = (
            (InputTrunk.NOTICE, OutputTrunk.DO, "ability:do"),
            (InputTrunk.SEE, OutputTrunk.LOOK, "ability:look"),
            (InputTrunk.HEAR, OutputTrunk.SPEAK, "ability:speak"),
        )
        for input_trunk, output_trunk, node_id in paths:
            _add_ability(
                mind,
                node_id,
                input_trunk=input_trunk,
                output_trunk=output_trunk,
            )

        kernel = SelfPulseKernel(mind)
        for input_trunk, _, node_id in reversed(paths):
            kernel.enqueue_input(
                node_id,
                lane=input_trunk,
                concept_ids=(node_id,),
                token_cost=1,
            )
        proposal = kernel.advance_cycle(cycle_id="motor:proposal")
        assert proposal is not None
        assert tuple(item.trunk for item in proposal.selected_outputs) == (
            OutputTrunk.DO,
            OutputTrunk.LOOK,
            OutputTrunk.SPEAK,
        )
        assert all(item.authorization_id for item in proposal.selected_outputs)

        opened = {}
        for affordance in proposal.selected_outputs:
            opened[affordance.trunk] = kernel.actualize_output(
                affordance, f"actualized {affordance.trunk.value}"
            )
        assert mind.pulse == proposal.self_state.pulse == 1

        with pytest.raises(ValueError, match="DO returns through NOTICE"):
            kernel.enqueue_cycle_return(
                opened[OutputTrunk.DO].cycle_id,
                "misrouted consequence",
                lane=InputTrunk.SEE,
                status="accepted",
                stability_delta=0.8,
                verified=True,
            )

        return_items = []
        for input_trunk, output_trunk, _ in paths:
            return_items.append(
                kernel.enqueue_cycle_return(
                    opened[output_trunk].cycle_id,
                    f"verified {output_trunk.value} consequence",
                    lane=input_trunk,
                    status="accepted",
                    stability_delta=0.8,
                    verified=True,
                    token_cost=1,
                )
            )
        returned = kernel.advance_cycle(
            cycle_id="motor:returns",
            item_ids=tuple(item.item_id for item in return_items),
        )
        assert returned is not None
        assert returned.self_state.pulse == 2
        assert returned.sensory.settled_order == (
            InputTrunk.NOTICE,
            InputTrunk.SEE,
            InputTrunk.HEAR,
        )
        assert {item.kind for item in returned.sensory.receipts} == {
            "cycle_return"
        }
        assert all(
            mind.experience_cycle(cycle.cycle_id).status == "closed"
            for cycle in opened.values()
        )
        assert all(
            mind.store.get_node_dynamics(node_id).valence > 0.0
            for _, _, node_id in paths
        )
        assert mind.store.connection.execute(
            "SELECT COUNT(*) FROM recurrent_pulses"
        ).fetchone()[0] == 2
        assert mind.graph.validate_invariants() == []


def test_idle_poll_cannot_repeat_an_output_without_a_new_self_pulse(
    tmp_path: Path,
) -> None:
    with _mind(tmp_path / "idle-output.sqlite") as mind:
        _add_ability(mind, "ability:wait")
        kernel = SelfPulseKernel(mind)
        kernel.enqueue_input(
            "wait", lane=InputTrunk.HEAR, concept_ids=("ability:wait",)
        )
        first = kernel.advance_cycle()
        assert first is not None and first.selected_output is not None

        assert kernel.advance_cycle() is None
        assert mind.pulse == first.self_state.pulse

        idle = kernel.advance_cycle(advance_when_idle=True)
        assert idle is not None
        assert idle.self_state.pulse == first.self_state.pulse + 1
        with pytest.raises(ValueError, match="superseded self pulse"):
            kernel.actualize_output(first.selected_output, "stale output")


def test_no_context_planner_uses_kernel_receipt_and_gated_speech(
    tmp_path: Path,
) -> None:
    with _mind(tmp_path / "planner-kernel.sqlite") as mind:
        mind.add_concept(
            "drive:answer",
            "opaque-drive",
            input_trunks=(InputTrunk.HEAR,),
            output_trunks=(OutputTrunk.SPEAK,),
            kind="drive",
        )
        for node_id, kind, embedding in (
            (
                "boundary:start",
                SPEECH_START_KIND,
                [0.0] * mind.embedder.dimension,
            ),
            (
                "boundary:stop",
                SPEECH_STOP_KIND,
                [0.0] * mind.embedder.dimension,
            ),
            (
                "lexeme:answer",
                "lexeme",
                mind.embedder.embed("answer surface"),
            ),
        ):
            mind.store.add_concept(
                ConceptNode(
                    concept_id=node_id,
                    label=node_id,
                    kind=kind,
                    embedding=as_tuple(embedding),
                    terms=(),
                    vault_id=None,
                    created_pulse=0,
                    last_active_pulse=0,
                )
            )
        mind.graph.add_relation(
            "drive:answer", "boundary:start", side=GraphSide.OUTPUT
        )
        mind.graph.add_relation(
            "boundary:start", "lexeme:answer", side=GraphSide.OUTPUT
        )
        mind.graph.add_relation(
            "lexeme:answer", "boundary:stop", side=GraphSide.OUTPUT
        )
        mind.recurrent.register(
            "drive:answer",
            pressure=0.85,
            expression_threshold=0.5,
            persistence=0.9,
        )
        planner = NoContextPlanner(mind)

        plan = planner.hear("opaque-drive")
        assert plan.self_pulse is not None
        assert plan.output_authorization_id is not None
        assert mind.pulse == 1
        assert all(trace.target_node_id == "SELF" for trace in plan.input_traces)
        assert mind.store.connection.execute(
            "SELECT COUNT(*) FROM self_pulse_states"
        ).fetchone()[0] == 1

        speech = planner.compose_speech(plan, minimum_words=1, maximum_words=2)
        assert mind.pulse == 1
        assert len(speech.steps) == 2
        assert mind.store.connection.execute(
            """SELECT COUNT(*) FROM self_internal_output_events
               WHERE parent_pulse = 1"""
        ).fetchone()[0] == len(speech.steps)
        hashes = mind.store.connection.execute(
            """SELECT e.state_after_sha256, p.state_sha256
               FROM self_internal_output_events e
               JOIN recurrent_pulses p ON p.pulse_id = e.event_id
               WHERE e.parent_pulse = 1"""
        ).fetchall()
        assert hashes
        assert all(
            row["state_after_sha256"] == row["state_sha256"] for row in hashes
        )

        cycle = planner.begin_speech_cycle(
            plan, "authorized graph speech", speech=speech
        )
        assert mind.pulse == 1
        assert cycle.metadata["self_pulse_authorization_id"] == (
            plan.output_authorization_id
        )
        with pytest.raises(ValueError, match="already consumed"):
            planner.begin_speech_cycle(plan, "duplicate graph speech")
