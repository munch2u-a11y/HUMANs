#!/usr/bin/env python3
"""Run a procedural, byte-native nursery through the integrated SELF pulse."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Sequence


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


def unit_vector(key: str, dimension: int, *, variation: int = 0) -> tuple[float, ...]:
    randomizer = random.Random(key)
    vector = [randomizer.uniform(-0.008, 0.008) for _ in range(dimension)]
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    for offset in range(8):
        index = int.from_bytes(digest[offset : offset + 2], "big") % dimension
        vector[index] += 1.0 if digest[(offset + 8) % len(digest)] & 1 else -1.0
    # Held-out samples are independently observed but remain inside the same
    # sensor basin. This tests recurrence/generalization without making the
    # locality-sensitive receptor signature an unrelated random vector.
    variation_index = int.from_bytes(digest[-2:], "big") % dimension
    vector[variation_index] += (variation % 5 - 2) * 0.0005
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return tuple(value / norm for value in vector)


def tiny_config() -> CortexConfig:
    return CortexConfig(
        byte_embedding_width=32,
        graph_width=32,
        direction_width=8,
        hidden_width=64,
        recurrent_layers=2,
        maximum_event_bytes=64,
        seed=1701,
    )


def run_nursery(
    *,
    database: Path,
    checkpoints: Path,
    device: str,
    tiny: bool,
    cycles: int,
    optimization_steps: int,
) -> dict[str, object]:
    config = tiny_config() if tiny else CortexConfig()
    training_by_state: dict[int, list] = {index: [] for index in range(4)}
    held_out_by_state: dict[int, object] = {}
    pulse_receipts = []
    with BornInHabitusRuntime(
        database,
        cortex_config=config,
        checkpoint_directory=checkpoints,
        device=device,
        # Keep the sensory graph broad even when --tiny shrinks only the
        # trainable cortex for a fast smoke run.
        mind_dimension=256,
    ) as runtime:
        # First, only non-language senses and consequences are admissible.
        sensory_cycles = max(2, cycles)
        all_training = []
        held_out_training = []
        pending_experience = None
        action_cycle_ids = []

        def collect_settled_experience() -> None:
            nonlocal pending_experience
            if pending_experience is None:
                return
            prior_receipt, action_cycle, prior_cycle, prior_state = pending_experience
            episodes = runtime.training_episodes(
                prior_receipt,
                experience_cycle_id=action_cycle.cycle_id,
            )
            if prior_cycle == sensory_cycles - 1:
                held_out_by_state[prior_state] = prior_receipt
                held_out_training.extend(episodes)
            else:
                training_by_state[prior_state].append(prior_receipt)
                all_training.extend(episodes)
            pending_experience = None

        for cycle in range(sensory_cycles):
            for state_index in range(4):
                receipt = runtime.advance(
                    (
                        DevelopmentalInput(
                            content=f"visual-carrier:{cycle}:{state_index}",
                            lane=InputTrunk.SEE,
                            source_id="procedural-camera",
                            embedding=unit_vector(
                                f"visual-state:{state_index}",
                                runtime.mind.embedder.dimension,
                                variation=cycle,
                            ),
                        ),
                        DevelopmentalInput(
                            content=f"internal-carrier:{cycle}:{state_index}",
                            lane=InputTrunk.NOTICE,
                            source_id="procedural-body",
                            embedding=unit_vector(
                                f"internal-state:{state_index}",
                                runtime.mind.embedder.dimension,
                                variation=cycle,
                            ),
                        ),
                    )
                )
                pulse_receipts.append(receipt)
                # Any return queued by the preceding action became canonical in
                # this pulse. Only now may it become a behavioral training item.
                collect_settled_experience()
                # Early errors prevent the arbitrary first tie winner from
                # monopolizing the motor field before alternatives are sampled.
                consequence = -0.8 if state_index % 2 == 0 else 0.8
                action_cycle = runtime.actualize(
                    receipt,
                    f"motor-carrier:{cycle}:{state_index}",
                    source_id="developmental-self",
                    metadata={
                        "nursery_scaffolded_environment": True,
                        "nursery_state_index": state_index,
                    },
                )
                action_trunk = action_cycle.output_trunk
                action_cycle_ids.append(action_cycle.cycle_id)
                runtime.queue_return(
                    action_cycle,
                    f"consequence-carrier:{cycle}:{state_index}",
                    status="settled",
                    stability_delta=consequence,
                    verified=True,
                    source_id="procedural-environment",
                    embedding=unit_vector(
                        f"consequence-state:{state_index}",
                        runtime.mind.embedder.dimension,
                        variation=cycle,
                    ),
                    metadata={"nursery_observed_outcome": True},
                )
                pending_experience = (receipt, action_cycle, cycle, state_index)

        # Settle the final queued motor return without treating it as an
        # outcome merely because the procedural environment announced it.
        flush = runtime.advance(
            (
                DevelopmentalInput(
                    content="nursery-settlement-pulse",
                    lane=InputTrunk.NOTICE,
                    source_id="procedural-clock",
                    embedding=unit_vector(
                        "nursery-settlement-pulse",
                        runtime.mind.embedder.dimension,
                    ),
                ),
            )
        )
        pulse_receipts.append(flush)
        collect_settled_experience()
        first_update = runtime.cortex.consolidate(
            tuple(all_training),
            curriculum_stage=runtime.curriculum.stage.value,
            optimization_steps=optimization_steps,
        )
        plasticity_updates = [first_update]

        # Held-out graph discrimination: classify the last variation from its
        # overlap with route fibers seen in earlier variations.
        def feature_set(receipt) -> set[str]:
            direct_records = {
                item.record_id
                for item in receipt.cycle.sensory.receipts
                if item.kind == "input"
            }
            return {
                node_id
                for growth in receipt.growth
                if not growth.cross_trunk
                and any(record_id in direct_records for record_id in growth.record_ids)
                for node_id in growth.feature_node_ids
            }

        prototypes = {
            state_index: set().union(
                *(feature_set(receipt) for receipt in receipts)
            )
            for state_index, receipts in training_by_state.items()
        }
        discrimination_hits = 0
        discrimination_predictions = []
        for expected, receipt in held_out_by_state.items():
            observed = feature_set(receipt)
            scores = {
                candidate: len(observed & prototype)
                / max(1, len(observed | prototype))
                for candidate, prototype in prototypes.items()
            }
            predicted = max(scores, key=lambda item: (scores[item], -item))
            discrimination_hits += predicted == expected
            discrimination_predictions.append(
                {"expected": expected, "predicted": predicted, "scores": scores}
            )
        nonverbal_score = discrimination_hits / max(1, len(held_out_by_state))
        consequence_hits = 0
        consequence_predictions = []
        for episode in held_out_training:
            evaluation = runtime.cortex.evaluate_episode(episode)
            consequence_hits += (
                evaluation.predicted_consequence >= 0.0
            ) == (episode.consequence >= 0.0)
            consequence_predictions.append(
                {
                    "episode_id": episode.episode_id,
                    "expected": episode.consequence,
                    "predicted": evaluation.predicted_consequence,
                }
            )
        consequence_score = consequence_hits / max(1, len(held_out_training))

        promoted_patterns = runtime.mind.store.list_concepts(
            kind="experiential_pattern"
        )
        cross_pattern_ids = {
            node_id
            for pulse_receipt in pulse_receipts
            for growth in pulse_receipt.growth
            if growth.cross_trunk
            for node_id in growth.promoted_pattern_node_ids
        }
        stage = runtime.curriculum.update_metrics(
            DevelopmentalMetrics(
                nonverbal_discrimination=nonverbal_score,
                consequence_prediction=consequence_score,
            )
        )

        language_receipts = []
        language_training_examples = 0
        held_out_byte_top_1 = None
        held_out_byte_top_5 = None
        held_out_language_route_cosine = None
        if stage == CurriculumStage.GROUNDED_FORMS:
            # Arbitrary pronounceable carriers are deliberately distributed
            # across varied frames. No semantic word list or sentence object
            # is installed; repeated byte spans must earn graph promotion.
            syllables = ("mava", "telu", "sori", "paku")
            frames = ("{}", "see {}", "{} moves", "touch {}")
            language_by_cycle = []
            for cycle in range(max(4, cycles)):
                cycle_receipts = []
                for index, syllable in enumerate(syllables):
                    text = frames[cycle % len(frames)].format(syllable)
                    receipt = runtime.advance(
                        (
                            DevelopmentalInput(
                                content=f"object-carrier:{cycle}:{index}",
                                lane=InputTrunk.SEE,
                                source_id="procedural-camera",
                                embedding=unit_vector(
                                    f"object:{index}",
                                    runtime.mind.embedder.dimension,
                                    variation=cycle,
                                ),
                            ),
                            DevelopmentalInput(
                                content=text,
                                lane=InputTrunk.HEAR,
                                source_id="caregiver",
                                episode_kind=EpisodeKind.GROUNDED_LABEL,
                            ),
                        )
                    )
                    language_receipts.append(receipt)
                    cycle_receipts.append(receipt)
                language_by_cycle.append(cycle_receipts)
            language_training = tuple(
                episode
                for cycle_receipts in language_by_cycle[:-1]
                for receipt in cycle_receipts
                for episode in runtime.training_episodes(receipt)
            )
            language_training_examples = len(language_training)
            if language_training:
                plasticity_updates.append(
                    runtime.cortex.consolidate(
                        language_training,
                        curriculum_stage=runtime.curriculum.stage.value,
                        optimization_steps=optimization_steps,
                    )
                )
            held_out_language = tuple(
                episode
                for receipt in language_by_cycle[-1]
                for episode in runtime.training_episodes(receipt)
                if episode.payload
            )
            held_out_language_evaluations = [
                runtime.cortex.evaluate_episode(episode)
                for episode in held_out_language
            ]
            if held_out_language_evaluations:
                held_out_byte_top_1 = sum(
                    item.byte_top_1 or 0.0
                    for item in held_out_language_evaluations
                ) / len(held_out_language_evaluations)
                held_out_byte_top_5 = sum(
                    item.byte_top_5 or 0.0
                    for item in held_out_language_evaluations
                ) / len(held_out_language_evaluations)
                held_out_language_route_cosine = sum(
                    item.route_cosine for item in held_out_language_evaluations
                ) / len(held_out_language_evaluations)

            expected_form_ids = [
                "byte-form:" + hashlib.sha256(item.encode("utf-8")).hexdigest()[:32]
                for item in syllables
            ]
            grounded_forms = [runtime.forms.statistics(item) for item in expected_form_ids]
            lexical_recall = sum(
                item.kind == "byte_lexeme" for item in grounded_forms
            ) / len(grounded_forms)
            shuffled_gap = sum(item.grounding_gap for item in grounded_forms) / len(
                grounded_forms
            )
            runtime.curriculum.update_metrics(
                DevelopmentalMetrics(
                    nonverbal_discrimination=nonverbal_score,
                    consequence_prediction=consequence_score,
                    lexical_recall_at_5=lexical_recall,
                    lexical_shuffled_gap=shuffled_gap,
                )
            )

        promoted_lexemes = runtime.mind.store.list_concepts(kind="byte_lexeme")
        candidate_lexemes = runtime.mind.store.list_concepts(
            kind="byte_lexeme_candidate"
        )
        latest = runtime.cortex.latest_pulse_state()
        report = {
            "schema": "habitus.developmental-cortex-nursery.v1",
            "device": str(runtime.cortex.device),
            "compute_dtype": str(runtime.cortex.compute_dtype),
            "parameter_count": runtime.cortex.parameter_count,
            "random_initialization": True,
            "pretrained_model_loaded": False,
            "tokenizer_loaded": False,
            "transcript_window": False,
            "pulses": runtime.mind.pulse,
            "curriculum_stage": runtime.curriculum.stage.value,
            "held_out_nonverbal_discrimination": nonverbal_score,
            "held_out_consequence_sign_accuracy": consequence_score,
            "held_out_byte_top_1": held_out_byte_top_1,
            "held_out_byte_top_5": held_out_byte_top_5,
            "held_out_language_route_cosine": held_out_language_route_cosine,
            "language_training_examples": language_training_examples,
            "held_out_nonverbal_predictions": discrimination_predictions,
            "held_out_consequence_predictions": consequence_predictions,
            "nonverbal_patterns": len(promoted_patterns),
            "cross_trunk_patterns": len(cross_pattern_ids),
            "byte_lexeme_candidates": len(candidate_lexemes),
            "byte_lexemes_promoted": len(promoted_lexemes),
            "cortex_state_sha256": latest.state_sha256 if latest else None,
            "model_sha256": runtime.cortex.model_sha256,
            "plasticity_updates": [
                {
                    "update_id": item.update_id,
                    "before_model_sha256": item.before_model_sha256,
                    "after_model_sha256": item.after_model_sha256,
                    "checkpoint_sha256": item.checkpoint_sha256,
                    "evidence_sha256": item.evidence_sha256,
                    "optimization_steps": item.optimization_steps,
                    "verified_behavior_examples": item.verified_behavior_examples,
                }
                for item in plasticity_updates
            ],
            "plasticity_receipt_count": runtime.mind.store.connection.execute(
                "SELECT COUNT(*) FROM cortex_plasticity_receipts"
            ).fetchone()[0],
            "verified_return_receipts": runtime.mind.store.connection.execute(
                "SELECT COUNT(*) FROM experience_cycle_returns WHERE verified = 1"
            ).fetchone()[0],
            "verified_behavior_examples": first_update.verified_behavior_examples,
            "receipt_backed_action_cycles": len(action_cycle_ids),
            "actualized_motor_trunks": {
                trunk.value: sum(
                    runtime.mind.store.get_experience_cycle(cycle_id).output_trunk
                    == trunk
                    for cycle_id in action_cycle_ids
                )
                for trunk in OutputTrunk
            },
            "motor_selection_modes": {
                mode: sum(
                    runtime.mind.store.get_experience_cycle(cycle_id).metadata.get(
                        "developmental_selection_mode"
                    )
                    == mode
                    for cycle_id in action_cycle_ids
                )
                for mode in (
                    "self_ranked",
                    "developmental_stochastic_exploration",
                    "caregiver_selected_authorized",
                )
            },
            "open_action_cycles": len(runtime.mind.store.list_open_experience_cycles()),
            "graph_invariant_errors": runtime.mind.graph.validate_invariants(),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        type=Path,
        default=EXPERIMENT_ROOT / "developmental_cortex_runs" / "mind.sqlite",
    )
    parser.add_argument(
        "--checkpoints",
        type=Path,
        default=EXPERIMENT_ROOT / "developmental_cortex_runs" / "checkpoints",
    )
    parser.add_argument("--device", choices=("amd", "rocm", "cpu", "auto"), default="amd")
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--optimization-steps", type=int, default=24)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_nursery(
        database=args.database,
        checkpoints=args.checkpoints,
        device=args.device,
        tiny=args.tiny,
        cycles=args.cycles,
        optimization_steps=args.optimization_steps,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    if report["graph_invariant_errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
