#!/usr/bin/env python3
"""Run the local recurrent Habitus without conversation or retrieval context.

No GGUF context is retained between turns. The current utterance is admitted
only through the HEAR membrane, persistent graph dynamics choose an output
route, and a fixed numeric state frame reaches the local language surface.
Each current utterance first grows opaque trunk-rooted experiential routes, then
can grow lexical and social fibers onto that lived structure. Stored words
remain immutable evidence and are never retrieved or serialized back into the
model to choose a response.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
EXPERIMENT_ROOT = Path(__file__).resolve().parent
for import_root in (SOURCE_ROOT, EXPERIMENT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from habitus_ai.open_weight import (  # noqa: E402
    DevelopmentalLanguage,
    LexicalSensation,
    NoContextPlanner,
    OpenWeightTurnPlan,
    RecurrentSpeech,
    write_native_state_packet,
)
from habitus_ai.pipeline import BaseAgenticMemoryRAG  # noqa: E402
from habitus_ai.types import (  # noqa: E402
    DevelopmentalGrowth,
    InputTrunk,
    OutputTrunk,
    RecordType,
)
import accelerated_gestation  # noqa: E402
import desire_nursery  # noqa: E402
import nursery  # noqa: E402
from opaque_skeleton import run_native  # noqa: E402
import reverse_nursery  # noqa: E402


DEFAULT_RUN_DIRECTORY = EXPERIMENT_ROOT / "open_weight_runs"
DEFAULT_DATABASE = DEFAULT_RUN_DIRECTORY / "unified-mind-v2.sqlite"
DEFAULT_RUNNER = EXPERIMENT_ROOT / "native" / "graph_soft_generator"


def bootstrap_database(args: argparse.Namespace) -> dict[str, object] | None:
    if args.database.exists():
        return None
    args.database.parent.mkdir(parents=True, exist_ok=True)
    return accelerated_gestation.compile_mind(
        args.database,
        args.model,
        args.codec,
        human_name=args.human_name,
        agent_name=args.agent_name,
        taste_schema=args.taste,
        replay_cycles=args.replay_cycles,
    )


def _audit_names(manifest: dict[str, object]) -> dict[str, str]:
    nodes = manifest.get("nodes", {})
    if not isinstance(nodes, dict):
        return {}
    return {str(node_id): str(name) for name, node_id in nodes.items()}


def _desire_rows(
    plan: OpenWeightTurnPlan,
    audit_names: dict[str, str],
) -> list[dict[str, object]]:
    return [
        {
            **asdict(item),
            "audit_name": audit_names.get(item.node_id),
        }
        for item in plan.recurrent.desires
    ]


def _growth_row(item: DevelopmentalGrowth) -> dict[str, object]:
    """Render a developmental receipt without leaking transport text."""
    return {
        "record_ids": list(item.record_ids),
        "input_trunks": [trunk.value for trunk in item.input_trunks],
        "feature_node_ids": list(item.feature_node_ids),
        "candidate_pattern_node_ids": list(item.candidate_pattern_node_ids),
        "promoted_pattern_node_ids": list(item.promoted_pattern_node_ids),
        "active_context_node_ids": list(item.active_context_node_ids),
        "edge_ids": list(item.edge_ids),
        "cross_trunk": item.cross_trunk,
    }


def decode_productive_surface(
    speech: RecurrentSpeech,
    *,
    model: Path,
    codec: Path,
) -> tuple[str, dict[str, object]]:
    """Decode only the individual geometry winners selected at each pulse."""
    rows = list(speech.lexical_rows)
    if not rows:
        return "", {"token_ids": [], "projection_tensor": None}
    projection = reverse_nursery.nearest_vocabulary(
        model, codec, rows, top_k=1
    )
    token_ids = [
        int(item["candidates"][0]["token_id"])
        for item in projection["items"]
        if item["candidates"]
    ]
    surface = (
        nursery.render_token_ids(model, codec, token_ids) if token_ids else ""
    )
    return surface.strip(), {
        "token_ids": token_ids,
        "projection_tensor": projection.get("tensor"),
        "rows": len(rows),
        "read_node_labels": False,
        "read_exposure_text": False,
        "word_competitions": len(speech.steps),
    }


def _novel_continuation(raw: str, spoken_surface: str) -> str:
    """Drop decoder echo while retaining any genuinely new continuation."""
    remaining = str(raw).strip()
    prefix = spoken_surface.strip()
    if not prefix:
        return remaining
    while remaining.casefold().startswith(prefix.casefold()):
        remaining = remaining[len(prefix) :].lstrip()
    return remaining


def close_pending_from_message(
    language: DevelopmentalLanguage,
    planner: NoContextPlanner,
    sensation: LexicalSensation,
    *,
    source_id: str,
) -> tuple[tuple[dict[str, object], ...], tuple[str, ...]]:
    """Queue a social return to settle beside the message that caused it."""
    queued = language.queue_social_cycles(
        sensation,
        source_id=source_id,
        pulse_kernel=planner.pulse_kernel,
    )
    return (
        tuple(asdict(assessment) for assessment, _ in queued),
        tuple(bucket.item_id for _, bucket in queued),
    )


def apply_explicit_reward(
    mind: BaseAgenticMemoryRAG,
    value: float,
    *,
    planner: NoContextPlanner | None = None,
    language: DevelopmentalLanguage | None = None,
    source_id: str | None = None,
) -> dict[str, object]:
    open_cycles = list(mind.open_experience_cycles(OutputTrunk.SPEAK))
    if language is not None and source_id is not None:
        source_node = language.ensure_source(source_id)
        open_cycles = [
            cycle
            for cycle in open_cycles
            if not cycle.metadata.get("source_node_id")
            or str(cycle.metadata.get("source_node_id")) == source_node
        ]
    if not open_cycles:
        return {
            "applied": False,
            "reason": "no open speech cycle for this source",
        }
    cycle = max(open_cycles, key=lambda item: (item.opened_pulse, item.cycle_id))
    bounded = max(-1.0, min(1.0, float(value)))
    pulse_receipt = None
    if planner is None:
        result = mind.record_cycle_return(
            cycle.cycle_id,
            "An explicit user evaluation was received.",
            input_trunk=InputTrunk.HEAR,
            status="accepted" if bounded >= 0.0 else "rejected",
            stability_delta=bounded,
            verified=True,
            terminal=True,
            source_id="communication-channel",
            record_type=RecordType.RECEIPT,
            allow_growth=False,
            embedding=[0.0] * mind.embedder.dimension,
            metadata={"no_context_runtime": True, "explicit_reward": bounded},
        )
        outcome_id = result.outcome.outcome_id
    else:
        bucket = planner.pulse_kernel.enqueue_cycle_return(
            cycle.cycle_id,
            "An explicit user evaluation was received.",
            lane=InputTrunk.HEAR,
            status="accepted" if bounded >= 0.0 else "rejected",
            stability_delta=bounded,
            verified=True,
            terminal=True,
            source_id="communication-channel",
            embedding=[0.0] * mind.embedder.dimension,
            metadata={"no_context_runtime": True, "explicit_reward": bounded},
        )
        pulse_receipt = planner.pulse_kernel.advance_cycle(
            cycle_id=f"explicit-reward:{cycle.cycle_id}",
            item_ids=(bucket.item_id,),
        )
        if pulse_receipt is None:
            raise RuntimeError("explicit reward did not produce a self pulse")
        return_record_id = pulse_receipt.sensory.receipts[0].record_id
        outcome_id = None
        for row in mind.store.connection.execute(
            "SELECT outcome_id, payload_json FROM outcomes ORDER BY created_at DESC"
        ).fetchall():
            payload = json.loads(row["payload_json"])
            if payload.get("receipt_id") == return_record_id:
                outcome_id = row["outcome_id"]
                break
        if outcome_id is None:
            raise RuntimeError("explicit reward has no persisted outcome receipt")
    source_state = None
    source_node_id = cycle.metadata.get("source_node_id")
    if language is not None and source_node_id:
        source_state = language.learn_source_preference(
            str(source_node_id), bounded, confidence=1.0
        )
    return {
        "applied": True,
        "cycle_id": cycle.cycle_id,
        "value": bounded,
        "outcome_id": outcome_id,
        "self_pulse_state_sha256": (
            pulse_receipt.self_state.state_sha256 if pulse_receipt else None
        ),
        "source_preference": asdict(source_state) if source_state else None,
    }


def run_plan(
    mind: BaseAgenticMemoryRAG,
    planner: NoContextPlanner,
    plan: OpenWeightTurnPlan,
    *,
    args: argparse.Namespace,
    audit_names: dict[str, str],
    social_returns: Sequence[dict[str, object]] = (),
) -> dict[str, Any]:
    selected = plan.focus.selected
    pulse_cycle = plan.self_pulse
    selected_affordance = pulse_cycle.selected_output if pulse_cycle else None
    receipt: dict[str, Any] = {
        "schema": "habitus.open-weight-turn.v4",
        "pulse_id": plan.pulse_id,
        "state_sha256": plan.recurrent.state_sha256,
        "input_sha256": plan.frame.input_sha256 if plan.frame else None,
        "input_record_id": plan.input_record.record_id if plan.input_record else None,
        "input_concept_ids": list(plan.input_concept_ids),
        "transcript_records_used": 0,
        "recalled_records_used": 0,
        "rendered_prompt_used": False,
        "lexical_acquisition": (
            asdict(plan.acquisition) if plan.acquisition is not None else None
        ),
        "developmental_growth": [
            _growth_row(item) for item in plan.developmental_growth
        ],
        "social_returns": list(social_returns),
        "self_pulse": (
            {
                "pulse": pulse_cycle.self_state.pulse,
                "cycle_id": pulse_cycle.cycle_id,
                "state_sha256": pulse_cycle.self_state.state_sha256,
                "recurrent_state_sha256": (
                    pulse_cycle.self_state.recurrent_state_sha256
                ),
                "settled_order": [
                    lane.value for lane in pulse_cycle.self_state.settled_order
                ],
                "input_item_ids": list(
                    pulse_cycle.self_state.input_item_ids
                ),
                "input_record_ids": list(
                    pulse_cycle.self_state.input_record_ids
                ),
                "inward_trace_ids": list(
                    pulse_cycle.self_state.inward_trace_ids
                ),
                "perceived_stability": (
                    pulse_cycle.self_state.perceived_stability
                ),
                "free_energy": pulse_cycle.self_state.free_energy,
                "output_opportunities": [
                    {
                        "trunk": item.trunk.value,
                        "node_id": (
                            item.affordance.node_id if item.affordance else None
                        ),
                        "authorization_id": (
                            item.affordance.authorization_id
                            if item.affordance
                            else None
                        ),
                    }
                    for item in pulse_cycle.output_opportunities
                ],
                "internal_output_events": [],
            }
            if pulse_cycle is not None
            else None
        ),
        "desires": _desire_rows(plan, audit_names),
        "dominant_desire": (
            {
                "node_id": plan.recurrent.dominant_desire_id,
                "audit_name": audit_names.get(plan.recurrent.dominant_desire_id or ""),
            }
            if plan.recurrent.dominant_desire_id
            else None
        ),
        "selected_output": (
            {
                "trunk": selected.trunk.value,
                "terminal_node_id": selected.terminal_node_id,
                "audit_name": audit_names.get(selected.terminal_node_id),
                "path_node_ids": list(selected.path_node_ids),
                "path_edge_ids": list(selected.path_edge_ids),
                "effective_probability": selected.effective_probability,
                "kernel_score": (
                    selected_affordance.score if selected_affordance else None
                ),
                "expected_stability_delta": (
                    selected_affordance.expected_stability_delta
                    if selected_affordance
                    else None
                ),
                "uncertainty": (
                    selected_affordance.uncertainty
                    if selected_affordance
                    else None
                ),
                "authorization_id": plan.output_authorization_id,
            }
            if selected
            else None
        ),
        "native": None,
        "cycle_id": None,
    }
    if plan.frame is None:
        receipt["withheld"] = True
        return receipt

    stamp = time.time_ns()
    packet_path = args.run_directory / f"state-{stamp}.packet"
    receipt_path = args.run_directory / f"turn-{stamp}.json"
    speech = planner.compose_speech(
        plan,
        minimum_words=args.minimum_words,
        maximum_words=args.maximum_words,
    )
    if pulse_cycle is not None and receipt["self_pulse"] is not None:
        internal_rows = mind.store.connection.execute(
            """SELECT event_id, sequence, kind, state_before_sha256,
                      state_after_sha256, metadata_json
               FROM self_internal_output_events
               WHERE parent_pulse = ? ORDER BY sequence""",
            (pulse_cycle.self_state.pulse,),
        ).fetchall()
        receipt["self_pulse"]["internal_output_events"] = [
            {
                "event_id": row["event_id"],
                "sequence": int(row["sequence"]),
                "kind": row["kind"],
                "state_before_sha256": row["state_before_sha256"],
                "state_after_sha256": row["state_after_sha256"],
                "metadata": json.loads(row["metadata_json"]),
            }
            for row in internal_rows
        ]
    write_native_state_packet(packet_path, speech.frame)
    spoken_surface, lexical_decode = decode_productive_surface(
        speech, model=args.model, codec=args.codec
    )
    native = None
    continuation = ""
    if args.with_native_continuation:
        native = run_native(
            args.model,
            args.runner,
            packet_path,
            maximum_tokens=args.maximum_tokens,
            seed=args.seed,
            skip_think=True,
        )
        continuation = _novel_continuation(
            str(native.get("response", "")), spoken_surface
        )
    response = " ".join(
        part for part in (spoken_surface, continuation) if part
    ).strip()
    receipt.update(
        {
            "packet_path": str(packet_path),
            "packet_rows": len(speech.frame.rows),
            "packet_row_kinds": list(speech.frame.row_kinds),
            "native": native,
            "productive_surface": spoken_surface,
            "productive_surface_decode": lexical_decode,
            "lexical_competition": {
                "decision_unit": "one_word_per_recurrent_pulse",
                "stopped": speech.stopped,
                "lexical_node_ids": list(speech.lexical_node_ids),
                "steps": [
                    {
                        "index": step.index,
                        "pulse_id": step.pulse_id,
                        "cursor_node_id": step.cursor_node_id,
                        "selected_node_id": step.selected.node_id,
                        "selected_probability": step.selected.probability,
                        "selected_semantic_support": step.selected.semantic_support,
                        "selected_convergence_count": step.selected.convergence_count,
                        "state_before_sha256": step.state_before_sha256,
                        "state_after_sha256": step.state_after_sha256,
                        "candidates": [
                            {
                                "node_id": candidate.node_id,
                                "probability": candidate.probability,
                                "semantic_support": candidate.semantic_support,
                                "convergence_count": candidate.convergence_count,
                                "transition_probability": candidate.transition_probability,
                                "boundary_pull": candidate.boundary_pull,
                                "stop": candidate.stop,
                            }
                            for candidate in step.candidates
                        ],
                    }
                    for step in speech.steps
                ],
            },
            "native_continuation": continuation,
            "response": response,
            "withheld": False,
        }
    )
    if response:
        cycle = planner.begin_speech_cycle(plan, response, speech=speech)
        receipt["cycle_id"] = cycle.cycle_id
        receipt["post_output_state_sha256"] = mind.recurrent.snapshot(
            pulse=mind.pulse
        ).state_sha256
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    receipt["receipt_path"] = str(receipt_path)
    return receipt


def state_report(
    mind: BaseAgenticMemoryRAG,
    audit_names: dict[str, str],
) -> dict[str, object]:
    state = mind.recurrent.snapshot(pulse=mind.pulse)
    return {
        "pulse": state.pulse,
        "state_sha256": state.state_sha256,
        "should_express": state.should_express,
        "dominant_desire": audit_names.get(state.dominant_desire_id or ""),
        "desires": [
            {
                "node_id": item.node_id,
                "audit_name": audit_names.get(item.node_id),
                "pressure": round(item.pressure, 6),
                "activation": round(item.activation, 6),
                "valence": round(item.valence, 6),
                "urgency": round(item.urgency, 6),
                "ready": item.ready,
            }
            for item in state.desires
        ],
        "language": {
            "schema": mind.store.get_metadata("online_language_schema"),
            "promotion_exposures": int(
                mind.store.get_metadata("online_language_promotion_exposures", "0")
                or 0
            ),
            "transition_promotion_exposures": int(
                mind.store.get_metadata(
                    "online_language_transition_promotion_exposures", "0"
                )
                or 0
            ),
            "productive_lexemes": len(mind.store.list_concepts(kind="lexeme")),
            "lexeme_candidates": len(
                mind.store.list_concepts(kind="lexeme_candidate")
            ),
            "social_sources": len(mind.store.list_concepts(kind="social_source")),
            "source_preferences": [
                asdict(item) for item in mind.store.list_social_source_states()
            ],
        },
        "developmental_breadth": {
            "status": mind.store.get_metadata("trunk_breadth_growth", "disabled"),
            "sensory_features": len(
                mind.store.list_concepts(kind="sensory_feature")
            ),
            "candidate_patterns": len(
                mind.store.list_concepts(kind="experiential_candidate")
            ),
            "promoted_patterns": len(
                mind.store.list_concepts(kind="experiential_pattern")
            ),
        },
    }


def _print_turn(receipt: dict[str, Any], *, show_receipt: bool) -> None:
    response = str(receipt.get("response", ""))
    if response:
        print(f"Habitus> {response}")
    elif receipt.get("withheld"):
        selected = receipt.get("selected_output") or {}
        print(f"Habitus> [no speech; selected {selected.get('trunk', 'nothing')}]" )
    else:
        print("Habitus> [the decoder produced no external speech]")
    if show_receipt:
        print(json.dumps(receipt, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--model", type=Path, default=nursery.MODEL)
    parser.add_argument("--codec", type=Path, default=nursery.CODEC)
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    parser.add_argument("--run-directory", type=Path, default=DEFAULT_RUN_DIRECTORY)
    parser.add_argument("--human-name", default="Human")
    parser.add_argument(
        "--source-id",
        help="transport-local speaker identity mapped to an opaque graph node",
    )
    parser.add_argument("--agent-name", default="Habitus")
    parser.add_argument("--taste", default="curious")
    parser.add_argument("--replay-cycles", type=int, default=1)
    parser.add_argument("--maximum-tokens", type=int, default=64)
    parser.add_argument("--minimum-words", type=int, default=3)
    parser.add_argument("--maximum-words", type=int, default=12)
    parser.add_argument("--lexeme-promotion-exposures", type=int, default=3)
    parser.add_argument("--transition-promotion-exposures", type=int, default=3)
    parser.add_argument("--maximum-heard-units", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--with-native-continuation",
        action="store_true",
        help="let the frozen transformer elaborate after graph-native speech",
    )
    parser.add_argument("--once")
    parser.add_argument("--ticks", type=int, default=0)
    parser.add_argument("--initialize-only", action="store_true")
    parser.add_argument("--show-receipt", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    required_assets = [args.model, args.codec]
    if args.with_native_continuation:
        required_assets.append(args.runner)
    for required in required_assets:
        if not required.is_file():
            raise SystemExit(f"required local asset is missing: {required}")
    args.run_directory.mkdir(parents=True, exist_ok=True)
    bootstrap_database(args)

    embedder = accelerated_gestation.NativeMassEmbedder(args.model, args.codec)
    with BaseAgenticMemoryRAG(args.database, embedder=embedder) as mind:
        embedder.bootstrap = False
        manifest = desire_nursery.install_desire_nursery(
            mind, args.model, args.codec
        )
        if not manifest["ready"]:
            raise RuntimeError("the drive nursery failed its graph invariants")
        audit_names = _audit_names(manifest)
        language = DevelopmentalLanguage(
            mind,
            promotion_exposures=args.lexeme_promotion_exposures,
            transition_promotion_exposures=args.transition_promotion_exposures,
            maximum_units=args.maximum_heard_units,
        )
        planner = NoContextPlanner(
            mind,
            lexical_sensor=embedder.lexical_sequence,
            language_learner=language,
        )
        active_source = args.source_id or args.human_name

        if args.initialize_only:
            print(
                json.dumps(
                    {
                        "database": str(args.database),
                        "nursery_ready": manifest["ready"],
                        "graph_invariants": mind.graph.validate_invariants(),
                        "state": state_report(mind, audit_names),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0

        if args.once:
            sensation = planner.sense(args.once)
            social_returns, social_item_ids = close_pending_from_message(
                language, planner, sensation, source_id=active_source
            )
            _print_turn(
                run_plan(
                    mind,
                    planner,
                    planner.hear(
                        args.once,
                        source_id=active_source,
                        sensation=sensation,
                        additional_input_item_ids=social_item_ids,
                    ),
                    args=args,
                    audit_names=audit_names,
                    social_returns=social_returns,
                ),
                show_receipt=args.show_receipt,
            )
            return 0

        for _ in range(max(0, args.ticks)):
            receipt = run_plan(
                mind,
                planner,
                planner.tick(),
                args=args,
                audit_names=audit_names,
            )
            _print_turn(receipt, show_receipt=args.show_receipt)
        if args.ticks:
            return 0

        print(
            "Local open-weight Habitus is awake. Commands: /state, /tick, "
            "/reward VALUE, /source ID, /quit"
        )
        while True:
            try:
                text = input("You> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if not text:
                continue
            if text in {"/quit", "/exit"}:
                return 0
            if text == "/state":
                print(json.dumps(state_report(mind, audit_names), indent=2))
                continue
            if text.startswith("/reward"):
                pieces = text.split(maxsplit=1)
                if len(pieces) != 2:
                    print("Use /reward with a number from -1 to 1.")
                    continue
                try:
                    result = apply_explicit_reward(
                        mind,
                        float(pieces[1]),
                        planner=planner,
                        language=language,
                        source_id=active_source,
                    )
                except ValueError:
                    print("Use /reward with a number from -1 to 1.")
                    continue
                print(json.dumps(result, indent=2))
                continue
            if text.startswith("/source"):
                pieces = text.split(maxsplit=1)
                if len(pieces) != 2 or not pieces[1].strip():
                    print("Use /source with a transport-local speaker ID.")
                    continue
                active_source = pieces[1].strip()
                print("Active source changed; its durable graph identity is opaque.")
                continue
            if text == "/tick":
                receipt = run_plan(
                    mind,
                    planner,
                    planner.tick(),
                    args=args,
                    audit_names=audit_names,
                )
                _print_turn(receipt, show_receipt=args.show_receipt)
                continue

            sensation = planner.sense(text)
            social_returns, social_item_ids = close_pending_from_message(
                language, planner, sensation, source_id=active_source
            )
            receipt = run_plan(
                mind,
                planner,
                planner.hear(
                    text,
                    source_id=active_source,
                    sensation=sensation,
                    additional_input_item_ids=social_item_ids,
                ),
                args=args,
                audit_names=audit_names,
                social_returns=social_returns,
            )
            _print_turn(receipt, show_receipt=args.show_receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
