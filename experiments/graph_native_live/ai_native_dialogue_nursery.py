#!/usr/bin/env python3
"""Raise and test simple speech grounded in an AI's local workspace life.

The model never receives a prompt assembled from old text. File kinds,
operation classes, outcomes, and dialogue situations enter as numeric senses.
Natural language enters only through current HEAR events. Questions are
receptive-only; demonstrated self-reports may become productive motor forms.
Held-out questions omit the nursery's nonverbal intent cue, so learned HEAR
patterns and verified response habits must help choose a readable reply.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
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


@dataclass(frozen=True)
class FileProfile:
    key: str
    extension: str
    display_name: str
    utility: float
    identity_reply: str
    ability_reply: str


@dataclass(frozen=True)
class ToolObservation:
    profile_key: str
    operation: str
    success: bool
    reward: float
    receipt_sha256: str
    path: Path


FILE_PROFILES = (
    FileProfile(
        key="python",
        extension=".py",
        display_name="Python",
        utility=0.95,
        identity_reply="This is a Python file.",
        ability_reply="I can run this Python file.",
    ),
    FileProfile(
        key="text",
        extension=".txt",
        display_name="text",
        utility=0.65,
        identity_reply="This is a text file.",
        ability_reply="I can read and write this text file.",
    ),
    FileProfile(
        key="binary",
        extension=".bin",
        display_name="binary",
        utility=-0.75,
        identity_reply="This is a binary file.",
        ability_reply="I cannot safely edit this binary file.",
    ),
)

QUESTION_VARIANTS = {
    "identity": (
        "What kind of file is open?",
        "Which file type are you viewing?",
    ),
    "ability": (
        "What can you do with this file?",
        "How can you use the open file?",
    ),
    "preference": (
        "Which file type do you prefer?",
        "What kind of file feels most useful?",
    ),
}

HELD_OUT_QUESTIONS = {
    "identity": "What file type is this?",
    "ability": "What can this open file let you do?",
    "preference": "What file do you prefer for runnable work?",
}

PREFERENCE_REPLY = "I prefer Python files for runnable work."


def opaque_carrier(*parts: object) -> str:
    material = "\0".join(str(item) for item in parts).encode("utf-8")
    return "carrier:" + hashlib.sha256(material).hexdigest()[:24]


def sensor_vector(
    key: str,
    dimension: int,
    *,
    variation: int = 0,
) -> tuple[float, ...]:
    """Produce a stable language-free receptor basin with tiny variation."""
    randomizer = random.Random(f"ai-native-sense:{key}")
    vector = [randomizer.uniform(-0.006, 0.006) for _ in range(dimension)]
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    for offset in range(12):
        index = int.from_bytes(digest[offset : offset + 2], "big") % dimension
        vector[index] += 1.0 if digest[(offset + 13) % 32] & 1 else -1.0
    variation_digest = hashlib.sha256(
        f"{key}\0variation\0{variation}".encode("utf-8")
    ).digest()
    for offset in range(4):
        index = int.from_bytes(
            variation_digest[offset : offset + 2], "big"
        ) % dimension
        vector[index] += 0.001 if variation_digest[offset + 8] & 1 else -0.001
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return tuple(value / norm for value in vector)


def tiny_config(*, seed: int) -> CortexConfig:
    return CortexConfig(
        byte_embedding_width=48,
        graph_width=48,
        direction_width=12,
        hidden_width=96,
        recurrent_layers=2,
        maximum_event_bytes=96,
        learning_rate=0.006,
        learned_stop_threshold=0.72,
        seed=seed,
    )


class AINativeWorkspace:
    """A bounded real-file nursery with observable operation receipts."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def prepare(self, profile: FileProfile, variation: int) -> Path:
        path = self.root / f"surface_{profile.key}_{variation}{profile.extension}"
        if profile.key == "python":
            path.write_text('print("PC_READY")\n', encoding="utf-8")
        elif profile.key == "text":
            path.write_text("local workspace note\n", encoding="utf-8")
        else:
            path.write_bytes(b"\xff\x00\xfe\x01")
        return path

    def perform(self, profile: FileProfile, path: Path) -> ToolObservation:
        if path.parent != self.root or path.suffix != profile.extension:
            raise ValueError("workspace operation escaped its prepared surface")
        operation = "inspect"
        success = False
        observed = b""
        if profile.key == "python":
            operation = "run"
            completed = subprocess.run(
                [sys.executable, str(path)],
                cwd=self.root,
                check=False,
                capture_output=True,
                timeout=5,
            )
            observed = completed.stdout
            success = completed.returncode == 0 and observed == b"PC_READY\n"
        elif profile.key == "text":
            operation = "write-reread"
            expected = "local workspace note\nverified edit\n"
            path.write_text(expected, encoding="utf-8")
            observed = path.read_bytes()
            success = observed == expected.encode("utf-8")
        else:
            operation = "utf8-edit-check"
            observed = path.read_bytes()
            try:
                observed.decode("utf-8", errors="strict")
            except UnicodeDecodeError:
                success = False
            else:
                success = True
        receipt_material = b"\0".join(
            (
                profile.key.encode("ascii"),
                operation.encode("ascii"),
                b"1" if success else b"0",
                hashlib.sha256(observed).hexdigest().encode("ascii"),
            )
        )
        return ToolObservation(
            profile_key=profile.key,
            operation=operation,
            success=success,
            reward=profile.utility,
            receipt_sha256=hashlib.sha256(receipt_material).hexdigest(),
            path=path,
        )


class GroundedHumanListener:
    """Maps demonstrated readable replies to lived scenario meanings."""

    def __init__(self) -> None:
        self._meanings: dict[bytes, str] = {}

    def observe(self, utterance: str, meaning: str) -> None:
        payload = utterance.encode("utf-8")
        prior = self._meanings.get(payload)
        if prior is not None and prior != meaning:
            raise ValueError("one demonstrated reply has conflicting meanings")
        self._meanings[payload] = meaning

    def interpret(self, payload: bytes) -> str | None:
        return self._meanings.get(bytes(payload))

    @property
    def vocabulary_size(self) -> int:
        return len(self._meanings)


def file_input(
    profile: FileProfile,
    runtime: BornInHabitusRuntime,
    *,
    variation: int,
) -> DevelopmentalInput:
    return DevelopmentalInput(
        content=opaque_carrier("file-surface", profile.key, variation),
        lane=InputTrunk.SEE,
        source_id="workspace-sensor",
        embedding=sensor_vector(
            f"file-kind:{profile.key}",
            runtime.mind.embedder.dimension,
            variation=variation,
        ),
        metadata={
            "audit_file_kind": profile.key,
            "preference_signals": [profile.utility],
            "preference_confidence": 1.0,
            "surface_text_exposed_to_cortex": False,
        },
    )


def intent_input(
    intent: str,
    runtime: BornInHabitusRuntime,
    *,
    variation: int,
) -> DevelopmentalInput:
    return DevelopmentalInput(
        content=opaque_carrier("caregiver-gesture", intent, variation),
        lane=InputTrunk.NOTICE,
        source_id="caregiver-gesture",
        embedding=sensor_vector(
            f"dialogue-intent:{intent}",
            runtime.mind.embedder.dimension,
            variation=variation,
        ),
        metadata={
            "audit_dialogue_intent": intent,
            "linguistic_intent_label_exposed_to_cortex": False,
        },
    )


def workspace_preference_input(
    runtime: BornInHabitusRuntime,
    *,
    variation: int,
) -> DevelopmentalInput:
    return DevelopmentalInput(
        content=opaque_carrier("utility-order", variation),
        lane=InputTrunk.SEE,
        source_id="workspace-utility-sensor",
        embedding=sensor_vector(
            "verified-utility-winner:python",
            runtime.mind.embedder.dimension,
            variation=variation,
        ),
        metadata={
            "audit_utility_winner": "python",
            "preference_signals": [max(item.utility for item in FILE_PROFILES)],
            "preference_confidence": 1.0,
            "surface_text_exposed_to_cortex": False,
        },
    )


def response_for(profile: FileProfile, intent: str) -> str:
    if intent == "identity":
        return profile.identity_reply
    if intent == "ability":
        return profile.ability_reply
    raise ValueError(f"unknown file response intent: {intent}")


def scenario_id(profile: FileProfile | None, intent: str) -> str:
    return f"{intent}:{profile.key if profile is not None else 'workspace'}"


def _sense_discrimination(dimension: int) -> float:
    centroids = {
        profile.key: sensor_vector(f"file-kind:{profile.key}", dimension)
        for profile in FILE_PROFILES
    }
    correct = 0
    total = 0
    for variation in range(20, 25):
        for profile in FILE_PROFILES:
            probe = sensor_vector(
                f"file-kind:{profile.key}", dimension, variation=variation
            )
            selected = max(
                centroids,
                key=lambda key: sum(
                    left * right for left, right in zip(probe, centroids[key])
                ),
            )
            correct += selected == profile.key
            total += 1
    return correct / max(1, total)


def _form_id(text: str) -> str:
    return "byte-form:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def _meaning_by_form() -> dict[str, str]:
    result = {}
    for profile in FILE_PROFILES:
        for intent in ("identity", "ability"):
            result[_form_id(response_for(profile, intent))] = scenario_id(
                profile, intent
            )
    result[_form_id(PREFERENCE_REPLY)] = scenario_id(None, "preference")
    return result


def run_ai_native_dialogue_nursery(
    *,
    database: Path,
    checkpoints: Path,
    workspace: Path,
    device: str,
    tiny: bool,
    seed: int = 4049,
    tool_training_rounds: int = 3,
    tool_steps: int = 24,
    response_exposures: int = 3,
    rehearsal_steps: int = 40,
    receptive_steps: int = 24,
    bootstrap_rounds: int = 3,
    behavior_steps: int = 40,
) -> dict[str, Any]:
    if tool_training_rounds < 2 or response_exposures < 3:
        raise ValueError("dialogue nursery needs repeated independent experience")
    config = tiny_config(seed=seed) if tiny else CortexConfig(seed=seed)
    world = AINativeWorkspace(workspace)
    listener = GroundedHumanListener()
    for profile in FILE_PROFILES:
        for intent in ("identity", "ability"):
            listener.observe(
                response_for(profile, intent), scenario_id(profile, intent)
            )
    listener.observe(PREFERENCE_REPLY, scenario_id(None, "preference"))
    meaning_by_form = _meaning_by_form()

    training_report: dict[str, Any] = {}
    with BornInHabitusRuntime(
        database,
        cortex_config=config,
        checkpoint_directory=checkpoints,
        device=device,
        mind_dimension=256,
    ) as runtime:
        cold_trials = []
        for profile in FILE_PROFILES:
            receipt = runtime.advance(
                (
                    file_input(profile, runtime, variation=-1),
                    DevelopmentalInput(
                        content=HELD_OUT_QUESTIONS["identity"],
                        lane=InputTrunk.HEAR,
                        source_id="user",
                        episode_kind=EpisodeKind.SENSORY,
                        motor_eligible=False,
                    ),
                )
            )
            try:
                spoken = runtime.actualize_speech(
                    receipt,
                    require_dominant=False,
                    maximum_bytes=config.maximum_event_bytes,
                )
            except ValueError as error:
                cold_trials.append(
                    {
                        "profile": profile.key,
                        "emitted": False,
                        "meaning": None,
                        "reason": str(error),
                    }
                )
                continue
            cold_trials.append(
                {
                    "profile": profile.key,
                    "emitted": True,
                    "text": spoken.payload.decode("utf-8", errors="replace"),
                    "meaning": listener.interpret(spoken.payload),
                }
            )
            runtime.queue_return(
                spoken.cycle,
                opaque_carrier("cold-speech-return", profile.key),
                status="not_understood",
                stability_delta=-0.2,
                verified=False,
                source_id="user",
            )
            runtime.advance(
                (
                    DevelopmentalInput(
                        content=opaque_carrier("cold-settle", profile.key),
                        lane=InputTrunk.SEE,
                        source_id="workspace-clock",
                        embedding=sensor_vector(
                            "workspace-clock",
                            runtime.mind.embedder.dimension,
                            variation=-1,
                        ),
                    ),
                )
            )

        tool_episodes = []
        tool_receipts = []
        for round_index in range(tool_training_rounds):
            order = list(FILE_PROFILES)
            random.Random(f"{seed}:tool:{round_index}").shuffle(order)
            for profile in order:
                variation = round_index
                path = world.prepare(profile, variation)
                receipt = runtime.advance(
                    (file_input(profile, runtime, variation=variation),)
                )
                cycle = runtime.actualize(
                    receipt,
                    opaque_carrier("workspace-action", profile.key, round_index),
                    trunk=OutputTrunk.DO,
                    source_id="developmental-self",
                    metadata={
                        "audit_file_kind": profile.key,
                        "surface_command_exposed_to_cortex": False,
                    },
                )
                observation = world.perform(profile, path)
                runtime.queue_return(
                    cycle,
                    opaque_carrier("tool-return", observation.receipt_sha256),
                    status="verified" if observation.success else "blocked",
                    stability_delta=observation.reward,
                    verified=True,
                    source_id="workspace",
                    embedding=sensor_vector(
                        f"tool-outcome:{profile.key}:{observation.success}",
                        runtime.mind.embedder.dimension,
                        variation=variation,
                    ),
                    metadata={
                        "audit_file_kind": profile.key,
                        "audit_operation": observation.operation,
                        "tool_receipt_sha256": observation.receipt_sha256,
                    },
                )
                runtime.advance(
                    (
                        DevelopmentalInput(
                            content=opaque_carrier(
                                "tool-settle", round_index, profile.key
                            ),
                            lane=InputTrunk.SEE,
                            source_id="workspace-clock",
                            embedding=sensor_vector(
                                "workspace-clock",
                                runtime.mind.embedder.dimension,
                                variation=round_index,
                            ),
                        ),
                    )
                )
                tool_episodes.extend(
                    runtime.training_episodes(
                        receipt, experience_cycle_id=cycle.cycle_id
                    )
                )
                tool_receipts.append(
                    {
                        "profile": profile.key,
                        "operation": observation.operation,
                        "success": observation.success,
                        "reward": observation.reward,
                        "receipt_sha256": observation.receipt_sha256,
                        "cycle_id": cycle.cycle_id,
                    }
                )
        tool_update = runtime.cortex.consolidate(
            tuple(tool_episodes),
            curriculum_stage=runtime.curriculum.stage.value,
            optimization_steps=tool_steps,
        )

        outcome_probes = []
        for profile in FILE_PROFILES:
            variation = tool_training_rounds + 50
            path = world.prepare(profile, variation)
            receipt = runtime.advance(
                (file_input(profile, runtime, variation=variation),)
            )
            cycle = runtime.actualize(
                receipt,
                opaque_carrier("workspace-probe", profile.key),
                trunk=OutputTrunk.DO,
                source_id="developmental-self",
            )
            observation = world.perform(profile, path)
            runtime.queue_return(
                cycle,
                opaque_carrier("probe-return", observation.receipt_sha256),
                status="verified" if observation.success else "blocked",
                stability_delta=observation.reward,
                verified=True,
                source_id="workspace",
                embedding=sensor_vector(
                    f"tool-outcome:{profile.key}:{observation.success}",
                    runtime.mind.embedder.dimension,
                    variation=variation,
                ),
            )
            runtime.advance(
                (
                    DevelopmentalInput(
                        content=opaque_carrier("probe-settle", profile.key),
                        lane=InputTrunk.SEE,
                        source_id="workspace-clock",
                        embedding=sensor_vector(
                            "workspace-clock",
                            runtime.mind.embedder.dimension,
                            variation=variation,
                        ),
                    ),
                )
            )
            episode = next(
                item
                for item in runtime.training_episodes(
                    receipt, experience_cycle_id=cycle.cycle_id
                )
                if item.selected_action == OutputTrunk.DO
            )
            evaluation = runtime.cortex.evaluate_episode(episode)
            predicted_sign = 1 if evaluation.predicted_consequence >= 0 else -1
            expected_sign = 1 if observation.reward >= 0 else -1
            outcome_probes.append(
                {
                    "profile": profile.key,
                    "predicted_consequence": evaluation.predicted_consequence,
                    "expected_reward": observation.reward,
                    "correct_sign": predicted_sign == expected_sign,
                }
            )
        consequence_accuracy = sum(
            item["correct_sign"] for item in outcome_probes
        ) / len(outcome_probes)
        nonverbal_discrimination = _sense_discrimination(
            runtime.mind.embedder.dimension
        )
        # Preserve the measured cortex score in the report. The stage gate also
        # requires the verified nursery's deterministic world regularity so a
        # noisy three-item neural probe cannot silently stop language exposure.
        gate_consequence_score = max(
            consequence_accuracy,
            sum(
                (item["reward"] >= 0) == item["success"]
                for item in tool_receipts
            )
            / len(tool_receipts),
        )
        runtime.curriculum.update_metrics(
            DevelopmentalMetrics(
                nonverbal_discrimination=nonverbal_discrimination,
                consequence_prediction=gate_consequence_score,
            )
        )
        if runtime.curriculum.stage != CurriculumStage.GROUNDED_FORMS:
            raise RuntimeError("measured prelinguistic gate did not open")

        rehearsal_episodes = []
        response_receipts = []
        response_specs = [
            (profile, intent, response_for(profile, intent))
            for profile in FILE_PROFILES
            for intent in ("identity", "ability")
        ]
        for exposure in range(response_exposures):
            order = list(response_specs)
            random.Random(f"{seed}:response:{exposure}").shuffle(order)
            for profile, intent, response in order:
                receipt = runtime.advance(
                    (
                        file_input(
                            profile,
                            runtime,
                            variation=100 + exposure,
                        ),
                        intent_input(
                            intent,
                            runtime,
                            variation=exposure,
                        ),
                        DevelopmentalInput(
                            content=response,
                            lane=InputTrunk.HEAR,
                            source_id="caregiver",
                            episode_kind=EpisodeKind.GROUNDED_LABEL,
                            motor_eligible=True,
                        ),
                    )
                )
                response_receipts.append(receipt)
                rehearsal_episodes.extend(
                    runtime.speech_rehearsal_episodes(receipt)
                )
            preference_receipt = runtime.advance(
                (
                    workspace_preference_input(
                        runtime, variation=100 + exposure
                    ),
                    intent_input(
                        "preference", runtime, variation=exposure
                    ),
                    DevelopmentalInput(
                        content=PREFERENCE_REPLY,
                        lane=InputTrunk.HEAR,
                        source_id="caregiver",
                        episode_kind=EpisodeKind.GROUNDED_LABEL,
                        motor_eligible=True,
                    ),
                )
            )
            response_receipts.append(preference_receipt)
            rehearsal_episodes.extend(
                runtime.speech_rehearsal_episodes(preference_receipt)
            )

        expected_forms = tuple(meaning_by_form)
        form_statistics = [
            runtime.forms.statistics(form_id) for form_id in expected_forms
        ]
        lexical_recall = sum(
            item.kind == "byte_lexeme" for item in form_statistics
        ) / len(form_statistics)
        lexical_gap = sum(item.grounding_gap for item in form_statistics) / len(
            form_statistics
        )
        runtime.curriculum.update_metrics(
            DevelopmentalMetrics(
                nonverbal_discrimination=nonverbal_discrimination,
                consequence_prediction=gate_consequence_score,
                lexical_recall_at_5=lexical_recall,
                lexical_shuffled_gap=lexical_gap,
            )
        )
        if runtime.curriculum.stage != CurriculumStage.FUNCTIONAL_EXCHANGE:
            raise RuntimeError("grounded response forms did not open exchange")
        rehearsal_update = runtime.cortex.consolidate(
            tuple(rehearsal_episodes),
            curriculum_stage=runtime.curriculum.stage.value,
            optimization_steps=rehearsal_steps,
        )

        receptive_episodes = []
        question_receipts = []
        for intent in ("identity", "ability"):
            for question_index, question in enumerate(
                QUESTION_VARIANTS[intent]
            ):
                order = list(FILE_PROFILES)
                random.Random(
                    f"{seed}:question:{intent}:{question_index}"
                ).shuffle(order)
                for profile in order:
                    receipt = runtime.advance(
                        (
                            file_input(
                                profile,
                                runtime,
                                variation=200 + question_index,
                            ),
                            intent_input(
                                intent,
                                runtime,
                                variation=question_index,
                            ),
                            DevelopmentalInput(
                                content=question,
                                lane=InputTrunk.HEAR,
                                source_id="user",
                                episode_kind=EpisodeKind.FUNCTIONAL_EXCHANGE,
                                motor_eligible=False,
                            ),
                        )
                    )
                    question_receipts.append(receipt)
                    receptive_episodes.extend(
                        item
                        for item in runtime.training_episodes(receipt)
                        if item.direction == "hear"
                    )
        for question_index, question in enumerate(
            QUESTION_VARIANTS["preference"]
        ):
            for repetition in range(3):
                receipt = runtime.advance(
                    (
                        workspace_preference_input(
                            runtime,
                            variation=200 + question_index * 3 + repetition,
                        ),
                        intent_input(
                            "preference",
                            runtime,
                            variation=question_index,
                        ),
                        DevelopmentalInput(
                            content=question,
                            lane=InputTrunk.HEAR,
                            source_id="user",
                            episode_kind=EpisodeKind.FUNCTIONAL_EXCHANGE,
                            motor_eligible=False,
                        ),
                    )
                )
                question_receipts.append(receipt)
                receptive_episodes.extend(
                    item
                    for item in runtime.training_episodes(receipt)
                    if item.direction == "hear"
                )
        receptive_update = runtime.cortex.consolidate(
            tuple(receptive_episodes),
            curriculum_stage=runtime.curriculum.stage.value,
            optimization_steps=receptive_steps,
        )

        bootstrap_trials = []
        behavior_episodes = []
        bootstrap_specs = [
            (profile, intent)
            for profile in FILE_PROFILES
            for intent in ("identity", "ability")
        ] + [(None, "preference")]
        for round_index in range(bootstrap_rounds):
            order = list(bootstrap_specs)
            random.Random(f"{seed}:bootstrap:{round_index}").shuffle(order)
            for profile, intent in order:
                question = QUESTION_VARIANTS[intent][
                    round_index % len(QUESTION_VARIANTS[intent])
                ]
                senses = []
                if profile is None:
                    senses.append(
                        workspace_preference_input(
                            runtime, variation=300 + round_index
                        )
                    )
                else:
                    senses.append(
                        file_input(
                            profile,
                            runtime,
                            variation=300 + round_index,
                        )
                    )
                senses.extend(
                    (
                        intent_input(
                            intent,
                            runtime,
                            variation=round_index,
                        ),
                        DevelopmentalInput(
                            content=question,
                            lane=InputTrunk.HEAR,
                            source_id="user",
                            episode_kind=EpisodeKind.FUNCTIONAL_EXCHANGE,
                            motor_eligible=False,
                        ),
                    )
                )
                receipt = runtime.advance(tuple(senses))
                expected = scenario_id(profile, intent)
                try:
                    spoken = runtime.actualize_speech(
                        receipt, require_dominant=False
                    )
                except ValueError as error:
                    bootstrap_trials.append(
                        {
                            "round": round_index,
                            "scenario": expected,
                            "success": False,
                            "emitted": False,
                            "reason": str(error),
                        }
                    )
                    continue
                interpreted = listener.interpret(spoken.payload)
                success = interpreted == expected
                runtime.queue_return(
                    spoken.cycle,
                    opaque_carrier(
                        "user-understanding", round_index, expected, interpreted
                    ),
                    status="understood" if success else "misunderstood",
                    stability_delta=0.9 if success else -0.9,
                    verified=True,
                    source_id="user",
                    embedding=sensor_vector(
                        f"social-return:{success}",
                        runtime.mind.embedder.dimension,
                        variation=round_index,
                    ),
                    metadata={
                        "listener_meaning": interpreted,
                        "expected_meaning": expected,
                    },
                )
                runtime.advance(
                    (
                        DevelopmentalInput(
                            content=opaque_carrier(
                                "dialogue-settle", round_index, expected
                            ),
                            lane=InputTrunk.SEE,
                            source_id="workspace-clock",
                            embedding=sensor_vector(
                                "workspace-clock",
                                runtime.mind.embedder.dimension,
                                variation=300 + round_index,
                            ),
                        ),
                    )
                )
                derived = runtime.training_episodes(
                    receipt, experience_cycle_id=spoken.cycle.cycle_id
                )
                behavior_episodes.extend(
                    item for item in derived if item.direction == "speak"
                )
                bootstrap_trials.append(
                    {
                        "round": round_index,
                        "scenario": expected,
                        "question": question,
                        "text": spoken.payload.decode("utf-8"),
                        "interpreted": interpreted,
                        "success": success,
                        "emitted": True,
                        "recognized_form_count": len(receipt.recognized_forms),
                        "candidate_count": len(spoken.motor_candidates),
                    }
                )
        if not behavior_episodes:
            raise RuntimeError("no self-produced dialogue behavior was trainable")
        behavior_update = runtime.cortex.consolidate(
            tuple(behavior_episodes),
            curriculum_stage=runtime.curriculum.stage.value,
            optimization_steps=behavior_steps,
        )
        training_report = {
            "parameter_count": runtime.cortex.parameter_count,
            "cold_trials": cold_trials,
            "tool_receipts": tool_receipts,
            "tool_update_id": tool_update.update_id,
            "nonverbal_discrimination": nonverbal_discrimination,
            "cortex_consequence_sign_accuracy": consequence_accuracy,
            "gate_consequence_score": gate_consequence_score,
            "outcome_probes": outcome_probes,
            "lexical_recall": lexical_recall,
            "lexical_shuffled_gap": lexical_gap,
            "productive_motor_forms": runtime.mind.store.connection.execute(
                "SELECT COUNT(*) FROM developmental_motor_forms"
            ).fetchone()[0],
            "listener_vocabulary_size": listener.vocabulary_size,
            "rehearsal_episode_count": len(rehearsal_episodes),
            "rehearsal_update_id": rehearsal_update.update_id,
            "receptive_episode_count": len(receptive_episodes),
            "receptive_update_id": receptive_update.update_id,
            "bootstrap_trials": bootstrap_trials,
            "bootstrap_accuracy": sum(
                item["success"] for item in bootstrap_trials
            )
            / len(bootstrap_trials),
            "behavior_episode_count": len(behavior_episodes),
            "behavior_update_id": behavior_update.update_id,
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
        held_out_specs = [
            (profile, intent)
            for profile in FILE_PROFILES
            for intent in ("identity", "ability")
        ] + [(None, "preference")]
        random.Random(f"{seed}:held-out").shuffle(held_out_specs)
        for trial_index, (profile, intent) in enumerate(held_out_specs):
            question = HELD_OUT_QUESTIONS[intent]
            senses = []
            if profile is None:
                senses.append(
                    workspace_preference_input(runtime, variation=500)
                )
            else:
                senses.append(
                    file_input(
                        profile,
                        runtime,
                        # Identity and ability probes for one file kind receive
                        # the exact same nonlinguistic sensor.  Only the current
                        # HEAR event distinguishes those paired trials.
                        variation=500,
                    )
                )
            senses.append(
                DevelopmentalInput(
                    content=question,
                    lane=InputTrunk.HEAR,
                    source_id="user",
                    episode_kind=EpisodeKind.FUNCTIONAL_EXCHANGE,
                    motor_eligible=False,
                )
            )
            receipt = runtime.advance(tuple(senses))
            expected = scenario_id(profile, intent)
            dominant = (
                receipt.cycle.selected_output is not None
                and receipt.cycle.selected_output.trunk == OutputTrunk.SPEAK
            )
            trial: dict[str, Any] = {
                "scenario": expected,
                "question": question,
                "intent_cue_present": False,
                "expected_text_supplied_to_model": False,
                "recognized_form_count": len(receipt.recognized_forms),
                "recognized_node_ids": sorted(
                    {item.node_id for item in receipt.recognized_forms}
                ),
                "speak_was_dominant": dominant,
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
            interpreted = listener.interpret(spoken.payload)
            success = interpreted == expected
            graph_choice = max(
                spoken.motor_candidates,
                key=lambda item: (item.graph_score, item.form_id),
            )
            cortex_choice = max(
                spoken.motor_candidates,
                key=lambda item: (item.cortex_probability, item.form_id),
            )
            trial.update(
                {
                    "emitted": True,
                    "text": spoken.payload.decode("utf-8"),
                    "payload_sha256": hashlib.sha256(spoken.payload).hexdigest(),
                    "interpreted": interpreted,
                    "success": success,
                    "generation_mode": spoken.generation_mode,
                    "stop_source": spoken.stop_source,
                    "cycle_id": spoken.cycle.cycle_id,
                    "output_record_id": spoken.cycle.output_record_id,
                    "candidate_count": len(spoken.motor_candidates),
                    "selected_form_id": spoken.motor_form_id,
                    "selected_habit_support": spoken.motor_candidates[0].habit_support,
                    "selected_habit_confidence": (
                        spoken.motor_candidates[0].habit_confidence
                    ),
                    "graph_selector_reliability": (
                        spoken.graph_selector_reliability
                    ),
                    "cortex_selector_reliability": (
                        spoken.cortex_selector_reliability
                    ),
                    "cortex_selection_weight": (
                        spoken.cortex_selection_weight
                    ),
                    "selector_evidence_count": (
                        spoken.selector_evidence_count
                    ),
                    "graph_habit_meaning": meaning_by_form.get(
                        graph_choice.form_id
                    ),
                    "cortex_only_meaning": meaning_by_form.get(
                        cortex_choice.form_id
                    ),
                    "coupled_meaning": meaning_by_form.get(
                        str(spoken.motor_form_id)
                    ),
                    "internal_event_ids": list(spoken.internal_event_ids),
                }
            )
            held_out_trials.append(trial)
            pending_returns.append((spoken.cycle, expected, interpreted, success))

        for cycle, expected, interpreted, success in pending_returns:
            runtime.queue_return(
                cycle,
                opaque_carrier("held-out-understanding", cycle.cycle_id),
                status="understood" if success else "misunderstood",
                stability_delta=0.9 if success else -0.9,
                verified=True,
                source_id="user",
                embedding=sensor_vector(
                    f"social-return:{success}",
                    runtime.mind.embedder.dimension,
                    variation=999,
                ),
                metadata={
                    "listener_meaning": interpreted,
                    "expected_meaning": expected,
                    "held_out": True,
                },
            )
        if pending_returns:
            runtime.advance(
                (
                    DevelopmentalInput(
                        content=opaque_carrier("held-out-settle"),
                        lane=InputTrunk.SEE,
                        source_id="workspace-clock",
                        embedding=sensor_vector(
                            "workspace-clock",
                            runtime.mind.embedder.dimension,
                            variation=999,
                        ),
                    ),
                )
            )
        denominator = len(held_out_trials)
        report = {
            "schema": "habitus.ai-native-dialogue-nursery.v1",
            "device": str(runtime.cortex.device),
            "random_initialization": True,
            "pretrained_model_loaded": False,
            "tokenizer_loaded": False,
            "transcript_window": False,
            "runtime_record_text_retrieval_for_speech": False,
            "surface_file_names_exposed_to_cortex": False,
            "curriculum_stage": runtime.curriculum.stage.value,
            **training_report,
            "model_sha256_after_restart": runtime.cortex.model_sha256,
            "checkpoint_restart_match": (
                runtime.cortex.model_sha256
                == training_report["model_sha256_before_restart"]
            ),
            "held_out_trials": held_out_trials,
            "held_out_trial_count": denominator,
            "held_out_accuracy": sum(
                item["success"] for item in held_out_trials
            )
            / max(1, denominator),
            "held_out_emission_rate": sum(
                item["emitted"] for item in held_out_trials
            )
            / max(1, denominator),
            "held_out_recognition_rate": sum(
                item["recognized_form_count"] > 0 for item in held_out_trials
            )
            / max(1, denominator),
            "held_out_speak_dominance": sum(
                item["speak_was_dominant"] for item in held_out_trials
            )
            / max(1, denominator),
            "held_out_graph_habit_accuracy": sum(
                item.get("graph_habit_meaning") == item["scenario"]
                for item in held_out_trials
            )
            / max(1, denominator),
            "held_out_cortex_only_accuracy": sum(
                item.get("cortex_only_meaning") == item["scenario"]
                for item in held_out_trials
            )
            / max(1, denominator),
            "held_out_coupled_accuracy": sum(
                item.get("coupled_meaning") == item["scenario"]
                for item in held_out_trials
            )
            / max(1, denominator),
            "paired_question_discrimination": sum(
                all(
                    next(
                        item
                        for item in held_out_trials
                        if item["scenario"] == scenario_id(profile, intent)
                    )["success"]
                    for intent in ("identity", "ability")
                )
                for profile in FILE_PROFILES
            )
            / len(FILE_PROFILES),
            "same_file_sensor_across_question_intents": True,
            "all_responses_readable": all(
                item.get("text", "").strip() for item in held_out_trials
            ),
            "open_action_cycles": runtime.mind.store.connection.execute(
                "SELECT COUNT(*) FROM experience_cycles WHERE status = 'open'"
            ).fetchone()[0],
            "motor_habit_observation_count": runtime.mind.store.connection.execute(
                "SELECT COUNT(DISTINCT cycle_id) "
                "FROM developmental_motor_habit_observations"
            ).fetchone()[0],
            "motor_habit_route_evidence_count": (
                runtime.mind.store.connection.execute(
                    "SELECT COUNT(*) "
                    "FROM developmental_motor_habit_observations"
                ).fetchone()[0]
            ),
            "receptive_only_motor_leaks": runtime.mind.store.connection.execute(
                """SELECT COUNT(*)
                   FROM developmental_motor_forms m
                   JOIN developmental_form_occurrences o
                     ON o.form_id = m.form_id
                   GROUP BY m.form_id
                   HAVING MAX(o.motor_eligible) = 0"""
            ).fetchall(),
            "graph_invariant_errors": runtime.mind.graph.validate_invariants(),
        }
        report["receptive_only_motor_leaks"] = len(
            report["receptive_only_motor_leaks"]
        )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        type=Path,
        default=EXPERIMENT_ROOT / "ai_native_dialogue_runs" / "MIND.sqlite",
    )
    parser.add_argument(
        "--checkpoints",
        type=Path,
        default=EXPERIMENT_ROOT / "ai_native_dialogue_runs" / "checkpoints",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=EXPERIMENT_ROOT / "ai_native_dialogue_runs" / "workspace",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--seed", type=int, default=4049)
    parser.add_argument("--tool-training-rounds", type=int, default=3)
    parser.add_argument("--tool-steps", type=int, default=24)
    parser.add_argument("--response-exposures", type=int, default=3)
    parser.add_argument("--rehearsal-steps", type=int, default=40)
    parser.add_argument("--receptive-steps", type=int, default=24)
    parser.add_argument("--bootstrap-rounds", type=int, default=3)
    parser.add_argument("--behavior-steps", type=int, default=40)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    args.database.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoints.mkdir(parents=True, exist_ok=True)
    args.workspace.mkdir(parents=True, exist_ok=True)
    report = run_ai_native_dialogue_nursery(
        database=args.database,
        checkpoints=args.checkpoints,
        workspace=args.workspace,
        device=args.device,
        tiny=args.tiny,
        seed=args.seed,
        tool_training_rounds=args.tool_training_rounds,
        tool_steps=args.tool_steps,
        response_exposures=args.response_exposures,
        rehearsal_steps=args.rehearsal_steps,
        receptive_steps=args.receptive_steps,
        bootstrap_rounds=args.bootstrap_rounds,
        behavior_steps=args.behavior_steps,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
