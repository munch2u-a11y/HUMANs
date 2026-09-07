#!/usr/bin/env python3
"""Raise and test a tiny receipt-backed communication convention.

The caregiver supplies arbitrary forms only while their nonverbal referents are
present.  During probes, the agent receives no HEAR input and no expected text.
Its emitted bytes alone reach a separately learned listener, whose resulting
world action supplies the success or failure receipt.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
EXPERIMENT_ROOT = Path(__file__).resolve().parent
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from habitus_ai import (  # noqa: E402
    BornInHabitusRuntime,
    CortexConfig,
    CurriculumStage,
    DevelopmentalInput,
    DevelopmentalMetrics,
    EpisodeKind,
    InputTrunk,
    OutputTrunk,
)


def sensor_vector(
    key: str,
    dimension: int,
    *,
    variation: int = 0,
) -> tuple[float, ...]:
    """Produce a language-free state basin with small held-out variation."""
    randomizer = random.Random(key)
    vector = [randomizer.uniform(-0.008, 0.008) for _ in range(dimension)]
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    for offset in range(10):
        index = int.from_bytes(digest[offset : offset + 2], "big") % dimension
        vector[index] += 1.0 if digest[(offset + 11) % len(digest)] & 1 else -1.0
    variation_digest = hashlib.sha256(
        f"{key}\0variation\0{variation}".encode("utf-8")
    ).digest()
    for offset in range(3):
        index = int.from_bytes(
            variation_digest[offset : offset + 2], "big"
        ) % dimension
        vector[index] += (
            0.0015 if variation_digest[offset + 8] & 1 else -0.0015
        )
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return tuple(value / norm for value in vector)


def procedural_forms(*, seed: int, count: int) -> tuple[str, ...]:
    """Generate a convention without installing a fixed vocabulary."""
    consonants = "bcdfghklmnprstvwz"
    vowels = "aeiou"
    forms = []
    nonce = 0
    while len(forms) < count:
        digest = hashlib.sha256(f"{seed}\0{nonce}".encode("ascii")).digest()
        form = "".join(
            (
                consonants[digest[0] % len(consonants)],
                vowels[digest[1] % len(vowels)],
                consonants[digest[2] % len(consonants)],
                vowels[digest[3] % len(vowels)],
                consonants[digest[4] % len(consonants)],
            )
        )
        if form not in forms:
            forms.append(form)
        nonce += 1
    return tuple(forms)


def opaque_carrier(*parts: object) -> str:
    material = "\0".join(str(item) for item in parts).encode("utf-8")
    return "carrier:" + hashlib.sha256(material).hexdigest()[:20]


def tiny_config(*, seed: int) -> CortexConfig:
    return CortexConfig(
        byte_embedding_width=32,
        graph_width=32,
        direction_width=8,
        hidden_width=64,
        recurrent_layers=2,
        maximum_event_bytes=32,
        learning_rate=0.01,
        learned_stop_threshold=0.70,
        seed=seed,
    )


class GroundedListener:
    """A receiver that learns demonstrations and sees only emitted bytes."""

    def __init__(self) -> None:
        self._counts: dict[bytes, Counter[int]] = defaultdict(Counter)

    def observe(self, payload: bytes, action_index: int) -> None:
        self._counts[bytes(payload)][int(action_index)] += 1

    def act(self, payload: bytes) -> int | None:
        counts = self._counts.get(bytes(payload))
        if not counts:
            return None
        return min(
            counts,
            key=lambda action: (-counts[action], action),
        )

    @property
    def demonstration_count(self) -> int:
        return sum(sum(counts.values()) for counts in self._counts.values())


def _feature_set(receipt: Any) -> set[str]:
    direct_records = {
        item.record_id
        for item in receipt.cycle.sensory.receipts
        if item.kind == "input"
    }
    return {
        node_id
        for growth in receipt.growth
        if not growth.cross_trunk
        and InputTrunk.HEAR not in growth.input_trunks
        and any(record_id in direct_records for record_id in growth.record_ids)
        for node_id in growth.feature_node_ids
    }


def _listener_success(
    listener: GroundedListener,
    payload: bytes,
    expected_action: int,
) -> tuple[int | None, bool]:
    action = listener.act(payload)
    return action, action == int(expected_action)


def run_communication_nursery(
    *,
    database: Path,
    checkpoints: Path,
    device: str,
    tiny: bool,
    state_count: int = 3,
    exposures: int = 4,
    rehearsal_steps: int = 80,
    bootstrap_rounds: int = 2,
    behavior_steps: int = 80,
    held_out_rounds: int = 2,
    seed: int = 2718,
) -> dict[str, Any]:
    if state_count < 2:
        raise ValueError("communication nursery requires at least two states")
    if exposures < 3:
        raise ValueError("communication nursery requires at least three exposures")
    if bootstrap_rounds < 1 or held_out_rounds < 1:
        raise ValueError("communication rounds must be positive")
    config = tiny_config(seed=seed) if tiny else CortexConfig(seed=seed)
    forms = procedural_forms(seed=seed, count=state_count)
    form_hash_to_state = {
        hashlib.sha256(form.encode("utf-8")).hexdigest(): state
        for state, form in enumerate(forms)
    }
    listener = GroundedListener()
    # The receiver learns the same public convention independently. None of
    # these demonstrations enter the agent until the caregiver phase below.
    for _ in range(exposures):
        for state, form in enumerate(forms):
            listener.observe(form.encode("utf-8"), state)

    training_report: dict[str, Any] = {}
    with BornInHabitusRuntime(
        database,
        cortex_config=config,
        checkpoint_directory=checkpoints,
        device=device,
        mind_dimension=256,
    ) as runtime:
        # This nursery starts at the grounded-language handoff. The separate
        # developmental cortex nursery owns the prelinguistic measured gate;
        # the explicit assertion is surfaced in the report rather than hidden.
        runtime.curriculum.update_metrics(
            DevelopmentalMetrics(
                nonverbal_discrimination=1.0,
                consequence_prediction=1.0,
            )
        )
        if runtime.curriculum.stage != CurriculumStage.GROUNDED_FORMS:
            raise RuntimeError("grounded-form prerequisite did not open")

        cold_trials = []
        cold_cycles = []
        for state in range(state_count):
            receipt = runtime.advance(
                (
                    DevelopmentalInput(
                        content=opaque_carrier("cold", state),
                        lane=InputTrunk.SEE,
                        source_id="nursery-camera",
                        embedding=sensor_vector(
                            f"referent:{state}",
                            runtime.mind.embedder.dimension,
                            variation=-1,
                        ),
                    ),
                )
            )
            try:
                spoken = runtime.actualize_speech(
                    receipt,
                    require_dominant=False,
                    maximum_bytes=max(len(item) for item in forms) + 2,
                )
            except ValueError as error:
                cold_trials.append(
                    {
                        "state": state,
                        "emitted": False,
                        "success": False,
                        "reason": str(error),
                    }
                )
                continue
            action, success = _listener_success(
                listener, spoken.payload, state
            )
            cold_cycles.append(spoken.cycle)
            cold_trials.append(
                {
                    "state": state,
                    "emitted": True,
                    "payload_sha256": hashlib.sha256(spoken.payload).hexdigest(),
                    "listener_action": action,
                    "success": success,
                    "generation_mode": spoken.generation_mode,
                }
            )
        for cycle in cold_cycles:
            runtime.queue_return(
                cycle,
                opaque_carrier("cold-return", cycle.cycle_id),
                status="baseline_observed",
                stability_delta=0.0,
                verified=False,
                source_id="nursery-listener",
            )
        if cold_cycles:
            runtime.advance(
                (
                    DevelopmentalInput(
                        content=opaque_carrier("cold-settle"),
                        lane=InputTrunk.NOTICE,
                        source_id="nursery-clock",
                        embedding=sensor_vector(
                            "cold-settle",
                            runtime.mind.embedder.dimension,
                        ),
                    ),
                )
            )

        rehearsal_episodes = []
        exposure_receipts: list[list[Any]] = [
            [] for _ in range(state_count)
        ]
        for exposure in range(exposures):
            exposure_order = list(range(state_count))
            random.Random(f"{seed}:exposure:{exposure}").shuffle(
                exposure_order
            )
            for state in exposure_order:
                form = forms[state]
                receipt = runtime.advance(
                    (
                        DevelopmentalInput(
                            content=opaque_carrier("exposure", exposure, state),
                            lane=InputTrunk.SEE,
                            source_id="nursery-camera",
                            embedding=sensor_vector(
                                f"referent:{state}",
                                runtime.mind.embedder.dimension,
                                variation=exposure,
                            ),
                        ),
                        DevelopmentalInput(
                            content=form,
                            lane=InputTrunk.HEAR,
                            source_id="caregiver",
                            episode_kind=EpisodeKind.GROUNDED_LABEL,
                        ),
                    )
                )
                exposure_receipts[state].append(receipt)
                rehearsal_episodes.extend(
                    runtime.speech_rehearsal_episodes(receipt)
                )

        expected_form_ids = {
            "byte-form:"
            + hashlib.sha256(form.encode("utf-8")).hexdigest()[:32]
            for form in forms
        }
        statistics = [
            runtime.forms.statistics(form_id)
            for form_id in sorted(expected_form_ids)
        ]
        lexical_recall = sum(
            item.kind == "byte_lexeme" for item in statistics
        ) / len(statistics)
        lexical_gap = sum(item.grounding_gap for item in statistics) / len(
            statistics
        )
        runtime.curriculum.update_metrics(
            DevelopmentalMetrics(
                nonverbal_discrimination=1.0,
                consequence_prediction=1.0,
                lexical_recall_at_5=lexical_recall,
                lexical_shuffled_gap=lexical_gap,
            )
        )
        if not rehearsal_episodes:
            raise RuntimeError("caregiver experience produced no motor rehearsal")
        rehearsal_update = runtime.cortex.consolidate(
            tuple(rehearsal_episodes),
            curriculum_stage=runtime.curriculum.stage.value,
            optimization_steps=rehearsal_steps,
        )

        bootstrap_trials = []
        behavior_episodes = []
        for round_index in range(bootstrap_rounds):
            bootstrap_order = list(range(state_count))
            random.Random(f"{seed}:bootstrap:{round_index}").shuffle(
                bootstrap_order
            )
            for state in bootstrap_order:
                receipt = runtime.advance(
                    (
                        DevelopmentalInput(
                            content=opaque_carrier(
                                "bootstrap", round_index, state
                            ),
                            lane=InputTrunk.SEE,
                            source_id="nursery-camera",
                            embedding=sensor_vector(
                                f"referent:{state}",
                                runtime.mind.embedder.dimension,
                                variation=exposures + round_index,
                            ),
                        ),
                    )
                )
                try:
                    spoken = runtime.actualize_speech(
                        receipt, require_dominant=False
                    )
                except ValueError as error:
                    bootstrap_trials.append(
                        {
                            "state": state,
                            "emitted": False,
                            "success": False,
                            "reason": str(error),
                        }
                    )
                    continue
                action, success = _listener_success(
                    listener, spoken.payload, state
                )
                runtime.queue_return(
                    spoken.cycle,
                    opaque_carrier(
                        "listener-result", round_index, state, action
                    ),
                    status="understood" if success else "misunderstood",
                    stability_delta=0.9 if success else -0.9,
                    verified=True,
                    source_id="nursery-listener",
                    metadata={
                        "listener_action": action,
                        "world_state": state,
                        "world_match": success,
                    },
                )
                runtime.advance(
                    (
                        DevelopmentalInput(
                            content=opaque_carrier(
                                "bootstrap-settle", round_index, state
                            ),
                            lane=InputTrunk.NOTICE,
                            source_id="nursery-clock",
                            embedding=sensor_vector(
                                "bootstrap-settle",
                                runtime.mind.embedder.dimension,
                                variation=round_index * state_count + state,
                            ),
                        ),
                    )
                )
                settled_cycle = runtime.mind.store.get_experience_cycle(
                    spoken.cycle.cycle_id
                )
                derived = runtime.training_episodes(
                    receipt,
                    experience_cycle_id=spoken.cycle.cycle_id,
                )
                behavior_episodes.extend(derived)
                bootstrap_trials.append(
                    {
                        "state": state,
                        "emitted": True,
                        "success": success,
                        "listener_action": action,
                        "payload_sha256": hashlib.sha256(
                            spoken.payload
                        ).hexdigest(),
                        "cycle_id": spoken.cycle.cycle_id,
                        "return_record_id": (
                            settled_cycle.terminal_return_record_id
                            if settled_cycle is not None
                            else None
                        ),
                        "generation_mode": spoken.generation_mode,
                        "stop_source": spoken.stop_source,
                        "speak_was_dominant": (
                            receipt.cycle.selected_output is not None
                            and receipt.cycle.selected_output.trunk
                            == OutputTrunk.SPEAK
                        ),
                    }
                )
        if not behavior_episodes:
            raise RuntimeError("no closed communication cycles were trainable")
        behavior_update = runtime.cortex.consolidate(
            tuple(behavior_episodes),
            curriculum_stage=runtime.curriculum.stage.value,
            optimization_steps=behavior_steps,
        )
        training_report = {
            "cold_trials": cold_trials,
            "lexical_recall": lexical_recall,
            "lexical_shuffled_gap": lexical_gap,
            "motor_form_count": runtime.mind.store.connection.execute(
                "SELECT COUNT(*) FROM developmental_motor_forms"
            ).fetchone()[0],
            "rehearsal_episode_count": len(rehearsal_episodes),
            "rehearsal_update_id": rehearsal_update.update_id,
            "rehearsal_evidence_sha256": rehearsal_update.evidence_sha256,
            "bootstrap_trials": bootstrap_trials,
            "bootstrap_successes": sum(
                item["success"] for item in bootstrap_trials
            ),
            "behavior_episode_count": len(behavior_episodes),
            "behavior_update_id": behavior_update.update_id,
            "behavior_evidence_sha256": behavior_update.evidence_sha256,
            "model_sha256_before_restart": runtime.cortex.model_sha256,
            "graph_invariant_errors_before_restart": (
                runtime.mind.graph.validate_invariants()
            ),
        }

    held_out_trials = []
    pending_returns = []
    with BornInHabitusRuntime(
        database,
        cortex_config=config,
        checkpoint_directory=checkpoints,
        device=device,
        mind_dimension=256,
    ) as runtime:
        model_sha256_after_restart = runtime.cortex.model_sha256
        for round_index in range(held_out_rounds):
            held_out_order = list(range(state_count))
            random.Random(f"{seed}:held-out:{round_index}").shuffle(
                held_out_order
            )
            for state in held_out_order:
                receipt = runtime.advance(
                    (
                        DevelopmentalInput(
                            content=opaque_carrier(
                                "held-out", round_index, state
                            ),
                            lane=InputTrunk.SEE,
                            source_id="nursery-camera",
                            embedding=sensor_vector(
                                f"referent:{state}",
                                runtime.mind.embedder.dimension,
                                variation=(
                                    exposures
                                    + bootstrap_rounds
                                    + 100
                                    + round_index
                                ),
                            ),
                        ),
                    )
                )
                dominant = (
                    receipt.cycle.selected_output is not None
                    and receipt.cycle.selected_output.trunk == OutputTrunk.SPEAK
                )
                trial: dict[str, Any] = {
                    "round": round_index,
                    "state": state,
                    "speak_was_dominant": dominant,
                    "proposal_id": (
                        receipt.cortex_proposal.proposal_id
                        if receipt.cortex_proposal is not None
                        else None
                    ),
                    "proposal_speak_probability": (
                        receipt.cortex_proposal.support_for(OutputTrunk.SPEAK)
                        if receipt.cortex_proposal is not None
                        else None
                    ),
                }
                try:
                    spoken = runtime.actualize_speech(receipt)
                except ValueError as error:
                    trial.update(
                        {
                            "emitted": False,
                            "success": False,
                            "reason": str(error),
                        }
                    )
                    held_out_trials.append(trial)
                    continue
                action, success = _listener_success(
                    listener, spoken.payload, state
                )
                candidates = spoken.motor_candidates
                graph_choice = (
                    max(candidates, key=lambda item: (item.graph_score, item.form_id))
                    if candidates
                    else None
                )
                cortex_choice = (
                    max(
                        candidates,
                        key=lambda item: (
                            item.cortex_probability,
                            item.form_id,
                        ),
                    )
                    if candidates
                    else None
                )
                output_record = runtime.mind.store.get_record(
                    spoken.cycle.output_record_id
                )
                trial.update(
                    {
                        "emitted": True,
                        "success": success,
                        "listener_action": action,
                        "cycle_id": spoken.cycle.cycle_id,
                        "output_record_id": spoken.cycle.output_record_id,
                        "output_payload_sha256": hashlib.sha256(
                            spoken.payload
                        ).hexdigest(),
                        "expected_payload_sha256": hashlib.sha256(
                            forms[state].encode("utf-8")
                        ).hexdigest(),
                        "generation_mode": spoken.generation_mode,
                        "stop_source": spoken.stop_source,
                        "motor_form_id": spoken.motor_form_id,
                        "internal_event_ids": list(spoken.internal_event_ids),
                        "cortex_output_state_sha256": (
                            spoken.cortex_output_state.state_sha256
                        ),
                        "candidate_count": len(candidates),
                        "graph_only_state": (
                            form_hash_to_state.get(graph_choice.payload_sha256)
                            if graph_choice is not None
                            else None
                        ),
                        "cortex_only_state": (
                            form_hash_to_state.get(cortex_choice.payload_sha256)
                            if cortex_choice is not None
                            else None
                        ),
                        "coupled_state": form_hash_to_state.get(
                            hashlib.sha256(spoken.payload).hexdigest()
                        ),
                        "output_metadata_self_generated": (
                            output_record is not None
                            and output_record.metadata.get(
                                "developmental_self_generated"
                            )
                            is True
                        ),
                    }
                )
                held_out_trials.append(trial)
                pending_returns.append((spoken.cycle, state, action, success))

        # Freeze the entire held-out generation sequence before any evaluation
        # return can alter graph strengths. Then close every causal cycle.
        for cycle, state, action, success in pending_returns:
            runtime.queue_return(
                cycle,
                opaque_carrier("held-out-result", cycle.cycle_id),
                status="understood" if success else "misunderstood",
                stability_delta=0.9 if success else -0.9,
                verified=True,
                source_id="nursery-listener",
                metadata={
                    "listener_action": action,
                    "world_state": state,
                    "world_match": success,
                    "held_out": True,
                },
            )
        if pending_returns:
            runtime.advance(
                (
                    DevelopmentalInput(
                        content=opaque_carrier("held-out-settle"),
                        lane=InputTrunk.NOTICE,
                        source_id="nursery-clock",
                        embedding=sensor_vector(
                            "held-out-settle",
                            runtime.mind.embedder.dimension,
                        ),
                    ),
                )
            )
            by_cycle = {
                cycle_id: runtime.mind.store.get_experience_cycle(cycle_id)
                for cycle_id in (
                    item["cycle_id"]
                    for item in held_out_trials
                    if item.get("cycle_id")
                )
            }
            for trial in held_out_trials:
                cycle = by_cycle.get(trial.get("cycle_id"))
                if cycle is not None:
                    trial["return_record_id"] = cycle.terminal_return_record_id

        emitted = [item for item in held_out_trials if item.get("emitted")]
        autonomous_successes = sum(item["success"] for item in held_out_trials)
        denominator = len(held_out_trials)
        report = {
            "schema": "habitus.developmental-communication-nursery.v1",
            "device": str(runtime.cortex.device),
            "parameter_count": runtime.cortex.parameter_count,
            "random_initialization": True,
            "pretrained_model_loaded": False,
            "tokenizer_loaded": False,
            "transcript_window": False,
            "runtime_memory_retrieval_for_speech": False,
            "prelinguistic_gate_source": (
                "scaffolded prerequisite assertion; not evidence-linked in this run"
            ),
            "state_count": state_count,
            "procedural_form_hashes": sorted(form_hash_to_state),
            "listener_demonstrations": listener.demonstration_count,
            "curriculum_stage": runtime.curriculum.stage.value,
            **training_report,
            "model_sha256_after_restart": model_sha256_after_restart,
            "checkpoint_restart_match": (
                model_sha256_after_restart
                == training_report["model_sha256_before_restart"]
            ),
            "held_out_trials": held_out_trials,
            "held_out_trial_count": denominator,
            "held_out_emission_rate": len(emitted) / max(1, denominator),
            "held_out_communication_accuracy": (
                autonomous_successes / max(1, denominator)
            ),
            "held_out_graph_only_accuracy": sum(
                item.get("graph_only_state") == item["state"]
                for item in emitted
            )
            / max(1, denominator),
            "held_out_cortex_only_accuracy": sum(
                item.get("cortex_only_state") == item["state"]
                for item in emitted
            )
            / max(1, denominator),
            "held_out_coupled_accuracy": sum(
                item.get("coupled_state") == item["state"]
                for item in emitted
            )
            / max(1, denominator),
            "dominant_speak_rate": sum(
                item["speak_was_dominant"] for item in held_out_trials
            )
            / max(1, denominator),
            "all_outputs_self_generated": bool(emitted)
            and all(
                item["output_metadata_self_generated"] for item in emitted
            ),
            "all_outputs_receipted": bool(emitted)
            and all(
                item.get("output_record_id")
                and item.get("return_record_id")
                and item.get("internal_event_ids")
                for item in emitted
            ),
            "open_action_cycles": runtime.mind.store.connection.execute(
                "SELECT COUNT(*) FROM experience_cycles WHERE status = 'open'"
            ).fetchone()[0],
            "graph_invariant_errors": runtime.mind.graph.validate_invariants(),
        }
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        type=Path,
        default=EXPERIMENT_ROOT / "communication_nursery_runs" / "MIND.sqlite",
    )
    parser.add_argument(
        "--checkpoints",
        type=Path,
        default=EXPERIMENT_ROOT
        / "communication_nursery_runs"
        / "checkpoints",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--states", type=int, default=3)
    parser.add_argument("--exposures", type=int, default=4)
    parser.add_argument("--rehearsal-steps", type=int, default=80)
    parser.add_argument("--bootstrap-rounds", type=int, default=2)
    parser.add_argument("--behavior-steps", type=int, default=80)
    parser.add_argument("--held-out-rounds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2718)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    args.database.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoints.mkdir(parents=True, exist_ok=True)
    report = run_communication_nursery(
        database=args.database,
        checkpoints=args.checkpoints,
        device=args.device,
        tiny=args.tiny,
        state_count=args.states,
        exposures=args.exposures,
        rehearsal_steps=args.rehearsal_steps,
        bootstrap_rounds=args.bootstrap_rounds,
        behavior_steps=args.behavior_steps,
        held_out_rounds=args.held_out_rounds,
        seed=args.seed,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
