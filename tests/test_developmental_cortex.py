from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from habitus_ai import (
    BaseAgenticMemoryRAG,
    BornInHabitusRuntime,
    DeterministicHashEmbedder,
    DevelopmentalCurriculum,
    DevelopmentalInput,
    DevelopmentalMetrics,
    EpisodeKind,
)
from habitus_ai.developmental_cortex import (
    CORTEX_INITIALIZATION,
    CortexConfig,
    CortexGeneration,
    CortexLineageError,
    CortexNumericalError,
    CortexProposal,
    CortexTrainingEpisode,
    DevelopmentalCortex,
)
from habitus_ai.self_pulse import SelfPulseKernel
from habitus_ai.types import InputTrunk, OutputTrunk


def _config() -> CortexConfig:
    return CortexConfig(
        byte_embedding_width=16,
        graph_width=16,
        direction_width=4,
        hidden_width=32,
        recurrent_layers=2,
        maximum_event_bytes=32,
        seed=19,
    )


def _mind(path: Path) -> BaseAgenticMemoryRAG:
    mind = BaseAgenticMemoryRAG(path, embedder=DeterministicHashEmbedder(32))
    mind.add_concept(
        "lived:signal",
        "opaque-signal",
        input_trunks=(InputTrunk.HEAR,),
        output_trunks=(OutputTrunk.SPEAK,),
        semantic_embedding=False,
    )
    mind.recurrent.register("lived:signal", persistence=0.9)
    return mind


def test_cortex_state_is_atomic_and_survives_restart_without_text_replay(
    tmp_path: Path,
) -> None:
    database = tmp_path / "cortex.sqlite"
    checkpoints = tmp_path / "checkpoints"
    with _mind(database) as mind:
        cortex = DevelopmentalCortex(
            mind, config=_config(), checkpoint_directory=checkpoints, device="cpu"
        )
        kernel = SelfPulseKernel(mind)
        bucket = kernel.enqueue_input(
            "current bytes only",
            lane=InputTrunk.HEAR,
            concept_ids=("lived:signal",),
        )
        cycle = kernel.advance_cycle(
            item_ids=(bucket.item_id,), pulse_state_observer=cortex.pulse_observer
        )
        assert cycle is not None
        state = cortex.latest_pulse_state()
        assert state is not None
        assert state.pulse == cycle.self_state.pulse
        assert state.sensed_byte_count == len(b"current bytes only")
        extension = kernel.latest_state_payload()["extensions"]["developmental_cortex"]
        assert extension["transcript_window"] is False
        expected_state = state.state_sha256
        expected_model = cortex.model_sha256

    with _mind(database) as reopened:
        cortex = DevelopmentalCortex(
            reopened,
            config=_config(),
            checkpoint_directory=checkpoints,
            device="cpu",
        )
        assert cortex.latest_pulse_state().state_sha256 == expected_state
        assert cortex.model_sha256 == expected_model

        def forbidden(*_args, **_kwargs):
            raise AssertionError("generation cannot read canonical transcript text")

        reopened.store.list_records = forbidden  # type: ignore[method-assign]
        reopened.store.get_records = forbidden  # type: ignore[method-assign]
        reopened.store.get_record = forbidden  # type: ignore[method-assign]
        field = [0.0] * _config().graph_width
        generated = cortex.generate_bytes(field, maximum_bytes=3)
        assert len(generated.payload) == 3


def test_cortex_proposal_is_read_only_and_bound_to_numeric_current_state(
    tmp_path: Path,
) -> None:
    with _mind(tmp_path / "proposal.sqlite") as mind:
        cortex = DevelopmentalCortex(
            mind,
            config=_config(),
            checkpoint_directory=tmp_path / "proposal-checkpoints",
            device="cpu",
        )
        before_model = cortex.model_sha256
        field = tuple([0.125] * _config().graph_width)
        proposal = cortex.propose(field, heard_payloads=(b"current signal",))

        assert proposal.proposal_id.startswith("cortex-proposal:")
        assert proposal.model_sha256 == before_model
        assert proposal.sensed_byte_count == len(b"current signal")
        assert set(proposal.action_probabilities) == {"DO", "LOOK", "SPEAK"}
        assert sum(proposal.action_probabilities.values()) == pytest.approx(1.0)
        assert 0.0 <= proposal.confidence <= 1.0
        assert cortex.model_sha256 == before_model
        assert cortex.latest_pulse_state() is None


def test_cortical_proposal_causally_changes_self_output_ranking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with BornInHabitusRuntime(
        tmp_path / "proposal-routing.sqlite",
        cortex_config=_config(),
        checkpoint_directory=tmp_path / "proposal-routing-checkpoints",
        device="cpu",
        mind_dimension=32,
    ) as runtime:
        original = runtime.cortex.propose

        def favor_speech(graph_field, *, heard_payloads=()):
            observed = original(graph_field, heard_payloads=heard_payloads)
            return replace(
                observed,
                action_probabilities={"DO": 0.005, "LOOK": 0.005, "SPEAK": 0.99},
                predicted_consequence=0.5,
                confidence=1.0,
            )

        monkeypatch.setattr(runtime.cortex, "propose", favor_speech)
        receipt = runtime.advance(
            (
                DevelopmentalInput(
                    content="visible current state",
                    lane=InputTrunk.SEE,
                    source_id="test-camera",
                    embedding=tuple([0.125] * 32),
                ),
            )
        )

        assert isinstance(receipt.cortex_proposal, CortexProposal)
        assert receipt.cycle.selected_output is not None
        assert receipt.cycle.selected_output.trunk == OutputTrunk.SPEAK
        assert receipt.cycle.selected_output.proposal_id == (
            receipt.cortex_proposal.proposal_id
        )
        assert receipt.cycle.selected_output.learned_support == pytest.approx(0.99)
        persisted = runtime.kernel.latest_state_payload()
        assert persisted["extensions"]["developmental_cortex_proposal"][
            "proposal_id"
        ] == receipt.cortex_proposal.proposal_id
        candidates = persisted["output_candidates"]
        assert {item["proposal_id"] for item in candidates} == {
            receipt.cortex_proposal.proposal_id
        }


def test_self_generated_speech_is_the_exact_receipted_and_trainable_action(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with BornInHabitusRuntime(
        tmp_path / "self-speech.sqlite",
        cortex_config=_config(),
        checkpoint_directory=tmp_path / "self-speech-checkpoints",
        device="cpu",
        mind_dimension=32,
    ) as runtime:
        original_proposal = runtime.cortex.propose

        def favor_speech(graph_field, *, heard_payloads=()):
            return replace(
                original_proposal(graph_field, heard_payloads=heard_payloads),
                action_probabilities={"DO": 0.005, "LOOK": 0.005, "SPEAK": 0.99},
                confidence=1.0,
            )

        monkeypatch.setattr(runtime.cortex, "propose", favor_speech)
        receipt = runtime.advance(
            (
                DevelopmentalInput(
                    content="current visible need",
                    lane=InputTrunk.SEE,
                    source_id="test-camera",
                    embedding=tuple([0.125] * 32),
                ),
                DevelopmentalInput(
                    content="what now?",
                    lane=InputTrunk.HEAR,
                    source_id="test-user",
                    motor_eligible=False,
                ),
            )
        )
        original_generate = runtime.cortex.generate_bytes

        def generated_from_cortex(graph_field, **kwargs):
            native = original_generate(graph_field, **kwargs)
            return CortexGeneration(
                payload=b"mava",
                step_probabilities=(0.8, 0.8, 0.8, 0.8),
                initial_state_sha256=native.initial_state_sha256,
                final_hidden=native.final_hidden,
                final_hidden_shape=native.final_hidden_shape,
                stopped=True,
                stop_probability=0.99,
            )

        monkeypatch.setattr(runtime.cortex, "generate_bytes", generated_from_cortex)
        spoken = runtime.actualize_speech(receipt)
        output = runtime.mind.store.get_record(spoken.cycle.output_record_id)
        assert output is not None
        assert output.text == "mava"
        assert output.metadata["developmental_self_generated"] is True
        assert output.metadata["cortex_payload_sha256"] == hashlib.sha256(
            b"mava"
        ).hexdigest()
        assert len(spoken.internal_event_ids) == len(b"mava") + 1
        assert runtime.mind.store.connection.execute(
            """SELECT COUNT(*) FROM self_internal_output_events
               WHERE parent_pulse = ?""",
            (receipt.cycle.self_state.pulse,),
        ).fetchone()[0] == len(b"mava") + 1

        runtime.queue_return(
            spoken.cycle,
            "listener acted on the message",
            status="understood",
            stability_delta=0.8,
            verified=True,
        )
        runtime.advance(
            (
                DevelopmentalInput(
                    content="observed listener action",
                    lane=InputTrunk.SEE,
                    source_id="test-camera",
                    embedding=tuple([-0.125] * 32),
                ),
            )
        )
        episodes = runtime.training_episodes(
            receipt, experience_cycle_id=spoken.cycle.cycle_id
        )
        self_speech = [episode for episode in episodes if episode.direction == "speak"]
        assert len(self_speech) == 1
        assert self_speech[0].payload == b"mava"
        assert self_speech[0].conditioning_payloads == (b"what now?",)
        assert self_speech[0].verified is True
        assert self_speech[0].selected_action == OutputTrunk.SPEAK
        with pytest.raises(
            ValueError, match="speech conditioning is not exact current HEAR evidence"
        ):
            runtime.cortex.consolidate(
                (
                    replace(
                        self_speech[0],
                        conditioning_payloads=(b"forged remembered question",),
                    ),
                ),
                curriculum_stage=runtime.curriculum.stage.value,
            )
        update = runtime.cortex.consolidate(
            self_speech,
            curriculum_stage=runtime.curriculum.stage.value,
        )
        assert update.verified_behavior_examples == 1


def test_grounded_motor_rehearsal_teaches_bytes_and_a_learned_stop(
    tmp_path: Path,
) -> None:
    config = replace(
        _config(), learning_rate=0.02, learned_stop_threshold=0.70, seed=23
    )
    with BornInHabitusRuntime(
        tmp_path / "speech-rehearsal.sqlite",
        cortex_config=config,
        checkpoint_directory=tmp_path / "speech-rehearsal-checkpoints",
        device="cpu",
        mind_dimension=32,
    ) as runtime:
        runtime.curriculum.update_metrics(
            DevelopmentalMetrics(
                nonverbal_discrimination=1.0,
                consequence_prediction=1.0,
            )
        )
        receipt = runtime.advance(
            (
                DevelopmentalInput(
                    content="grounded visible state",
                    lane=InputTrunk.SEE,
                    source_id="test-camera",
                    embedding=tuple([0.25] * 32),
                ),
                DevelopmentalInput(
                    content="mava",
                    lane=InputTrunk.HEAR,
                    source_id="caregiver",
                    episode_kind=EpisodeKind.GROUNDED_LABEL,
                ),
            )
        )
        rehearsals = runtime.speech_rehearsal_episodes(receipt)
        assert len(rehearsals) == 1
        assert rehearsals[0].direction == "speak"
        assert rehearsals[0].motor_rehearsal is True
        assert rehearsals[0].verified is False

        runtime.cortex.consolidate(
            rehearsals,
            curriculum_stage=runtime.curriculum.stage.value,
            optimization_steps=160,
        )
        scored = runtime.cortex.score_sequences(
            receipt.graph_field, (b"mava", b"telu")
        )
        assert sum(item.competition_probability for item in scored) == pytest.approx(1.0)
        assert scored[0].competition_probability > scored[1].competition_probability
        motorized = runtime.cortex.motorize_bytes(
            receipt.graph_field, b"mava"
        )
        assert motorized.payload == b"mava"
        assert motorized.stopped is True
        assert motorized.stop_source == "promoted_whole_form"
        generated = runtime.cortex.generate_bytes(
            receipt.graph_field, maximum_bytes=8, minimum_bytes=1
        )
        assert generated.stopped is True
        assert generated.payload == b"mava"
        assert generated.stop_source == "learned_neural"


def test_recognized_grounded_form_opens_annealed_speech_exploration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with BornInHabitusRuntime(
        tmp_path / "speech-exploration.sqlite",
        cortex_config=_config(),
        checkpoint_directory=tmp_path / "speech-exploration-checkpoints",
        device="cpu",
        mind_dimension=32,
    ) as runtime:
        runtime.curriculum.update_metrics(
            DevelopmentalMetrics(
                nonverbal_discrimination=1.0,
                consequence_prediction=1.0,
            )
        )
        visible = tuple([0.25] * 32)
        contrasting = tuple([-0.25] * 32)
        for exposure in range(4):
            runtime.advance(
                (
                    DevelopmentalInput(
                        content=f"opaque visible carrier {exposure}",
                        lane=InputTrunk.SEE,
                        source_id="test-camera",
                        embedding=visible,
                    ),
                    DevelopmentalInput(
                        content="mava",
                        lane=InputTrunk.HEAR,
                        source_id="caregiver",
                        episode_kind=EpisodeKind.GROUNDED_LABEL,
                    ),
                )
            )
            runtime.advance(
                (
                    DevelopmentalInput(
                        content=f"contrasting visible carrier {exposure}",
                        lane=InputTrunk.SEE,
                        source_id="test-camera",
                        embedding=contrasting,
                    ),
                    DevelopmentalInput(
                        content="telu",
                        lane=InputTrunk.HEAR,
                        source_id="caregiver",
                        episode_kind=EpisodeKind.GROUNDED_LABEL,
                    ),
                )
            )

        original_proposal = runtime.cortex.propose

        def strongly_favor_do(graph_field, *, heard_payloads=()):
            return replace(
                original_proposal(graph_field, heard_payloads=heard_payloads),
                action_probabilities={
                    "DO": 0.998,
                    "LOOK": 0.001,
                    "SPEAK": 0.001,
                },
                predicted_consequence=0.8,
                confidence=1.0,
            )

        monkeypatch.setattr(runtime.cortex, "propose", strongly_favor_do)
        probe = runtime.advance(
            (
                DevelopmentalInput(
                    content="opaque visible probe",
                    lane=InputTrunk.SEE,
                    source_id="test-camera",
                    embedding=visible,
                ),
                DevelopmentalInput(
                    content="mava",
                    lane=InputTrunk.HEAR,
                    source_id="caregiver",
                    episode_kind=EpisodeKind.GROUNDED_LABEL,
                    motor_eligible=False,
                ),
            )
        )

        assert probe.recognized_forms
        assert any(
            item.trunk == OutputTrunk.SPEAK
            for item in probe.cycle.selected_outputs
        )
        state = runtime.kernel.latest_state_payload()
        assert state is not None
        pressure = state["extensions"]["developmental_growth"][
            "speech_exploration_pressure"
        ]
        assert pressure > 0.0
        speak_candidate = next(
            item
            for item in probe.cycle.output_candidates
            if item.trunk == OutputTrunk.SPEAK
        )
        assert speak_candidate.transient_pull >= pressure


def test_speech_selector_influence_is_calibrated_by_verified_agreement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with BornInHabitusRuntime(
        tmp_path / "selector-calibration.sqlite",
        cortex_config=_config(),
        checkpoint_directory=tmp_path / "selector-calibration-checkpoints",
        device="cpu",
        mind_dimension=32,
    ) as runtime:
        runtime.curriculum.update_metrics(
            DevelopmentalMetrics(
                nonverbal_discrimination=1.0,
                consequence_prediction=1.0,
            )
        )
        for exposure in range(4):
            for form, embedding in (
                ("mava", tuple([0.25] * 32)),
                ("telu", tuple([-0.25] * 32)),
            ):
                runtime.advance(
                    (
                        DevelopmentalInput(
                            content=f"calibration ground {form} {exposure}",
                            lane=InputTrunk.SEE,
                            source_id="test-camera",
                            embedding=embedding,
                        ),
                        DevelopmentalInput(
                            content=form,
                            lane=InputTrunk.HEAR,
                            source_id="caregiver",
                            episode_kind=EpisodeKind.GROUNDED_LABEL,
                        ),
                    )
                )
        successful_form_id = (
            "byte-form:" + hashlib.sha256(b"mava").hexdigest()[:32]
        )
        alternative_form_id = (
            "byte-form:" + hashlib.sha256(b"telu").hexdigest()[:32]
        )
        assert runtime.mind.store.connection.execute(
            "SELECT 1 FROM developmental_motor_forms WHERE form_id = ?",
            (successful_form_id,),
        ).fetchone()

        original_proposal = runtime.cortex.propose

        def favor_speech(graph_field, *, heard_payloads=()):
            return replace(
                original_proposal(graph_field, heard_payloads=heard_payloads),
                action_probabilities={
                    "DO": 0.005,
                    "LOOK": 0.005,
                    "SPEAK": 0.99,
                },
                confidence=1.0,
            )

        monkeypatch.setattr(runtime.cortex, "propose", favor_speech)
        for observation in range(3):
            receipt = runtime.advance(
                (
                    DevelopmentalInput(
                        content=f"visible calibration state {observation}",
                        lane=InputTrunk.SEE,
                        source_id="test-camera",
                        embedding=tuple([0.125] * 32),
                    ),
                )
            )
            cycle = runtime.actualize(
                receipt,
                "mava",
                trunk=OutputTrunk.SPEAK,
                metadata={
                    "developmental_self_generated": True,
                    "developmental_motor_form_id": successful_form_id,
                    "developmental_motor_candidate_scores": [
                        {
                            "form_id": successful_form_id,
                            "graph_score": 0.9,
                            "cortex_probability": 0.1,
                        },
                        {
                            "form_id": alternative_form_id,
                            "graph_score": 0.1,
                            "cortex_probability": 0.9,
                        },
                    ],
                },
            )
            runtime.queue_return(
                cycle,
                f"understood calibration return {observation}",
                status="understood",
                stability_delta=0.9,
                verified=True,
                source_id="test-listener",
            )
            runtime.advance(
                (
                    DevelopmentalInput(
                        content=f"visible settled state {observation}",
                        lane=InputTrunk.SEE,
                        source_id="test-camera",
                        embedding=tuple([-0.125] * 32),
                    ),
                )
            )

        graph_reliability, cortex_reliability, cortex_weight, evidence = (
            runtime._speech_selector_calibration()
        )
        assert evidence == 3
        assert graph_reliability > cortex_reliability
        assert cortex_weight < 0.30


def test_self_selects_a_promoted_motor_form_from_current_lived_routes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with BornInHabitusRuntime(
        tmp_path / "motor-form.sqlite",
        cortex_config=_config(),
        checkpoint_directory=tmp_path / "motor-form-checkpoints",
        device="cpu",
        mind_dimension=32,
    ) as runtime:
        runtime.curriculum.update_metrics(
            DevelopmentalMetrics(
                nonverbal_discrimination=1.0,
                consequence_prediction=1.0,
            )
        )
        original_proposal = runtime.cortex.propose

        def favor_speech(graph_field, *, heard_payloads=()):
            return replace(
                original_proposal(graph_field, heard_payloads=heard_payloads),
                action_probabilities={"DO": 0.005, "LOOK": 0.005, "SPEAK": 0.99},
                confidence=1.0,
            )

        monkeypatch.setattr(runtime.cortex, "propose", favor_speech)
        def opaque_sensor(key: str) -> tuple[float, ...]:
            digest = hashlib.sha256(key.encode("utf-8")).digest()
            values = [((value / 255.0) - 0.5) for value in digest]
            norm = sum(value * value for value in values) ** 0.5
            return tuple(value / norm for value in values)

        states = (
            ("mava", opaque_sensor("state-a")),
            ("telu", opaque_sensor("bar")),
        )
        for exposure in range(3):
            for form, embedding in states:
                runtime.advance(
                    (
                        DevelopmentalInput(
                            content=f"sensor-carrier:{exposure}:{form}",
                            lane=InputTrunk.SEE,
                            source_id="test-camera",
                            embedding=embedding,
                        ),
                        DevelopmentalInput(
                            content=form,
                            lane=InputTrunk.HEAR,
                            source_id="caregiver",
                            episode_kind=EpisodeKind.GROUNDED_LABEL,
                        ),
                    )
                )

        probe = runtime.advance(
            (
                DevelopmentalInput(
                    content="held-out sensor carrier",
                    lane=InputTrunk.SEE,
                    source_id="test-camera",
                    embedding=states[0][1],
                ),
            )
        )
        original_get_record = runtime.mind.store.get_record

        def forbidden_record_retrieval(*_args, **_kwargs):
            raise AssertionError("speech motor cannot retrieve canonical record text")

        runtime.mind.store.get_record = forbidden_record_retrieval  # type: ignore[method-assign]
        try:
            spoken = runtime.actualize_speech(probe)
        finally:
            runtime.mind.store.get_record = original_get_record  # type: ignore[method-assign]
        output = runtime.mind.store.get_record(spoken.cycle.output_record_id)

        assert spoken.payload == b"mava"
        assert spoken.generation_mode == "promoted_form_competition"
        assert spoken.stop_source == "promoted_whole_form"
        assert spoken.motor_form_id is not None
        assert spoken.motor_candidates[0].form_id == spoken.motor_form_id
        assert len(spoken.internal_event_ids) == len(b"mava") + 2
        assert spoken.cortex_output_state.stop_source == "promoted_whole_form"
        assert output is not None
        assert output.text == "mava"
        assert output.metadata["developmental_generation_mode"] == (
            "promoted_form_competition"
        )
        assert output.metadata["cortex_stop_source"] == "promoted_whole_form"


def test_cortex_write_rolls_back_when_later_extension_work_fails(tmp_path: Path) -> None:
    with _mind(tmp_path / "rollback.sqlite") as mind:
        cortex = DevelopmentalCortex(
            mind,
            config=_config(),
            checkpoint_directory=tmp_path / "rollback-checkpoints",
            device="cpu",
        )
        kernel = SelfPulseKernel(mind)
        bucket = kernel.enqueue_input(
            "transient", lane=InputTrunk.HEAR, concept_ids=("lived:signal",)
        )

        def fail_after_cortex(connection, frame):
            cortex.pulse_observer(connection, frame)
            raise RuntimeError("later extension failure")

        with pytest.raises(RuntimeError, match="later extension failure"):
            kernel.advance_cycle(
                item_ids=(bucket.item_id,), pulse_state_observer=fail_after_cortex
            )
        assert mind.store.connection.execute(
            "SELECT COUNT(*) FROM cortex_pulse_states"
        ).fetchone()[0] == 0
        assert mind.store.connection.execute(
            "SELECT COUNT(*) FROM self_pulse_states"
        ).fetchone()[0] == 0


def test_grounded_consolidation_changes_weights_and_restores_own_lineage(
    tmp_path: Path,
) -> None:
    database = tmp_path / "learning.sqlite"
    checkpoints = tmp_path / "learning-checkpoints"
    with _mind(database) as mind:
        cortex = DevelopmentalCortex(
            mind, config=_config(), checkpoint_directory=checkpoints, device="cpu"
        )
        curriculum = DevelopmentalCurriculum(mind)
        kernel = SelfPulseKernel(mind)
        bucket = kernel.enqueue_input(
            "signal",
            lane=InputTrunk.HEAR,
            concept_ids=("lived:signal",),
        )
        cycle = kernel.advance_cycle(
            item_ids=(bucket.item_id,), pulse_state_observer=cortex.pulse_observer
        )
        assert cycle is not None
        state = cortex.latest_pulse_state()
        assert state is not None
        record = mind.store.get_record(cycle.self_state.input_record_ids[0])
        assert record is not None
        admission = curriculum.admit(
            record,
            episode_kind=EpisodeKind.SENSORY,
            route_ids=state.route_ids,
        )
        assert admission.accepted
        field = state.graph_field
        negative = tuple(-value for value in field)
        before = cortex.model_sha256
        receipt = cortex.consolidate(
            (
                CortexTrainingEpisode(
                    episode_id="episode:grounded",
                    pulse=cycle.self_state.pulse,
                    record_ids=(record.record_id,),
                    route_ids=admission.route_ids,
                    graph_field=field,
                    negative_graph_field=negative,
                    payload=b"signal",
                    direction="hear",
                    kind=admission.episode_kind.value,
                ),
            ),
            curriculum_stage="grounded_forms",
        )
        assert receipt.before_model_sha256 == before
        assert receipt.after_model_sha256 != before
        assert Path(receipt.checkpoint_path).is_file()
        assert hashlib.sha256(Path(receipt.checkpoint_path).read_bytes()).hexdigest() == (
            receipt.checkpoint_sha256
        )
        evidence_row = mind.store.connection.execute(
            """SELECT episode_evidence_json, optimization_steps
               FROM cortex_plasticity_receipts WHERE update_id = ?""",
            (receipt.update_id,),
        ).fetchone()
        assert evidence_row is not None
        assert receipt.evidence_sha256 == hashlib.sha256(
            evidence_row["episode_evidence_json"].encode("utf-8")
        ).hexdigest()
        assert evidence_row["optimization_steps"] == 1
        expected = receipt.after_model_sha256

    with _mind(database) as reopened:
        restored = DevelopmentalCortex(
            reopened,
            config=_config(),
            checkpoint_directory=checkpoints,
            device="cpu",
        )
        assert restored.model_sha256 == expected
        lineage = reopened.store.connection.execute(
            "SELECT initialization, pretrained_source FROM cortex_lineage"
        ).fetchone()
        assert lineage["initialization"] == CORTEX_INITIALIZATION
        assert lineage["pretrained_source"] is None


def test_behavioral_heads_require_a_real_authorized_and_returned_cycle(
    tmp_path: Path,
) -> None:
    with BornInHabitusRuntime(
        tmp_path / "receipt-backed.sqlite",
        cortex_config=_config(),
        checkpoint_directory=tmp_path / "receipt-checkpoints",
        device="cpu",
        mind_dimension=32,
    ) as runtime:
        receipt = runtime.advance(
            (
                DevelopmentalInput(
                    content="visible state",
                    lane=InputTrunk.SEE,
                    source_id="test-camera",
                    embedding=tuple([0.125] * 32),
                ),
            )
        )
        action = runtime.actualize(
            receipt,
            "test motor carrier",
            trunk=OutputTrunk.DO,
        )
        with pytest.raises(ValueError, match="terminal return"):
            runtime.training_episodes(
                receipt, experience_cycle_id=action.cycle_id
            )

        runtime.queue_return(
            action,
            "test consequence carrier",
            status="completed",
            stability_delta=0.75,
            verified=True,
            embedding=tuple([0.25] * 32),
        )
        runtime.advance(
            (
                DevelopmentalInput(
                    content="next visible state",
                    lane=InputTrunk.SEE,
                    source_id="test-camera",
                    embedding=tuple([-0.125] * 32),
                ),
            )
        )
        episodes = runtime.training_episodes(
            receipt, experience_cycle_id=action.cycle_id
        )
        assert episodes
        assert all(episode.verified for episode in episodes)
        assert all(episode.selected_action == OutputTrunk.DO for episode in episodes)
        assert all(episode.consequence == 0.75 for episode in episodes)
        cycle = runtime.mind.store.get_experience_cycle(action.cycle_id)
        assert cycle is not None and cycle.terminal_return_record_id is not None
        assert all(
            cycle.output_record_id in episode.record_ids
            and cycle.terminal_return_record_id in episode.record_ids
            for episode in episodes
        )

        with pytest.raises(ValueError, match="differs from verified return"):
            runtime.cortex.consolidate(
                (replace(episodes[0], consequence=-0.75),),
                curriculum_stage=runtime.curriculum.stage.value,
            )
        altered_field = list(episodes[0].graph_field)
        altered_field[0] += 0.25
        with pytest.raises(ValueError, match="differs from its committed pulse"):
            runtime.cortex.consolidate(
                (replace(episodes[0], graph_field=tuple(altered_field)),),
                curriculum_stage=runtime.curriculum.stage.value,
            )
        update = runtime.cortex.consolidate(
            episodes,
            curriculum_stage=runtime.curriculum.stage.value,
        )
        assert update.verified_behavior_examples == len(episodes)


def test_nonrandom_initialization_is_rejected() -> None:
    with pytest.raises(CortexLineageError, match="random_from_seed"):
        CortexConfig(initialization="pretrained")


def test_developmental_training_uses_stable_precision_on_every_backend(
    tmp_path: Path,
) -> None:
    with _mind(tmp_path / "stable-precision.sqlite") as mind:
        cortex = DevelopmentalCortex(
            mind,
            config=_config(),
            checkpoint_directory=tmp_path / "stable-precision-checkpoints",
            device="cpu",
        )
        assert cortex.compute_dtype is torch.float32
        assert all(parameter.dtype is torch.float32 for parameter in cortex.model.parameters())


def test_nonfinite_training_loss_cannot_receive_a_plasticity_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _mind(tmp_path / "nonfinite.sqlite") as mind:
        cortex = DevelopmentalCortex(
            mind,
            config=_config(),
            checkpoint_directory=tmp_path / "nonfinite-checkpoints",
            device="cpu",
        )
        curriculum = DevelopmentalCurriculum(mind)
        kernel = SelfPulseKernel(mind)
        bucket = kernel.enqueue_input(
            "signal", lane=InputTrunk.HEAR, concept_ids=("lived:signal",)
        )
        cycle = kernel.advance_cycle(
            item_ids=(bucket.item_id,), pulse_state_observer=cortex.pulse_observer
        )
        assert cycle is not None
        state = cortex.latest_pulse_state()
        assert state is not None
        record = mind.store.get_record(cycle.self_state.input_record_ids[0])
        assert record is not None
        admission = curriculum.admit(
            record,
            episode_kind=EpisodeKind.SENSORY,
            route_ids=state.route_ids,
        )
        assert admission.accepted
        episode = CortexTrainingEpisode(
            episode_id="episode:nonfinite",
            pulse=state.pulse,
            record_ids=(record.record_id,),
            route_ids=admission.route_ids,
            graph_field=state.graph_field,
            payload=b"signal",
            direction="hear",
            kind=admission.episode_kind.value,
        )
        original_forward = cortex.model.forward

        def nonfinite_forward(*args, **kwargs):
            outputs, hidden = original_forward(*args, **kwargs)
            outputs["route"] = outputs["route"] * torch.tensor(float("nan"))
            return outputs, hidden

        monkeypatch.setattr(cortex.model, "forward", nonfinite_forward)
        with pytest.raises(CortexNumericalError, match="non-finite loss"):
            cortex.consolidate((episode,), curriculum_stage="prelinguistic")
        assert mind.store.connection.execute(
            "SELECT COUNT(*) FROM cortex_plasticity_receipts"
        ).fetchone()[0] == 0
