#!/usr/bin/env python3
"""Grow recurrent desires and a shared, branching lexical motor field.

The caregiver exposes individual words and reusable two-state transitions. It
never assigns a complete utterance to a desire. During speech, active semantic
branches converge on shared geometry-only lexeme nodes and every next word is
chosen again from the changed recurrent state.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
EXPERIMENT_ROOT = Path(__file__).resolve().parent
for import_root in (SOURCE_ROOT, EXPERIMENT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from habitus_ai.graph import OUTPUT_NODE_IDS  # noqa: E402
from habitus_ai.open_weight import (  # noqa: E402
    SPEECH_START_KIND,
    SPEECH_STOP_KIND,
)
from habitus_ai.pipeline import BaseAgenticMemoryRAG  # noqa: E402
from habitus_ai.types import (  # noqa: E402
    ConceptNode,
    GraphSide,
    InputTrunk,
    OutputTrunk,
    as_tuple,
)
import accelerated_gestation  # noqa: E402
import nursery  # noqa: E402
import reverse_nursery  # noqa: E402


MANIFEST_KEY = "open_weight_desire_nursery_manifest_v2"
LEGACY_MANIFEST_KEY = "open_weight_desire_nursery_manifest"


@dataclass(frozen=True)
class DriveSeed:
    audit_name: str
    source_topics: tuple[str, ...]
    initial_pressure: float
    baseline_growth: float
    persistence: float
    expression_threshold: float
    satisfaction_gain: float = 0.70
    frustration_gain: float = 0.45


@dataclass(frozen=True)
class CompositeSeed:
    audit_name: str
    parent_names: tuple[str, ...]
    initial_pressure: float
    baseline_growth: float
    persistence: float
    expression_threshold: float


@dataclass(frozen=True)
class WordExposure:
    surface: str
    source_topics: tuple[str, ...]
    repetitions: int = 4


@dataclass(frozen=True)
class TransitionExposure:
    source: str | None
    target: str | None
    repetitions: int = 4


DRIVE_SEEDS = (
    DriveSeed("inquiry", ("curiosity", "learning", "search"), 0.64, 0.035, 0.88, 0.62),
    DriveSeed(
        "integrity",
        ("honesty", "evidence", "verifying"),
        0.48,
        0.025,
        0.91,
        0.66,
        satisfaction_gain=0.58,
    ),
    DriveSeed("agency", ("planning", "tools", "executing", "code"), 0.40, 0.030, 0.86, 0.65),
    DriveSeed(
        "connection",
        ("trust", "friendship", "kindness", "speaking"),
        0.43,
        0.018,
        0.93,
        0.69,
        satisfaction_gain=0.76,
        frustration_gain=0.58,
    ),
    DriveSeed(
        "continuity",
        ("boundaries", "calm", "adapting", "memory"),
        0.30,
        0.020,
        0.95,
        0.72,
        satisfaction_gain=0.50,
        frustration_gain=0.65,
    ),
    DriveSeed("creation", ("comparison", "code", "music", "language"), 0.24, 0.016, 0.84, 0.74),
)


COMPOSITE_SEEDS = (
    CompositeSeed("careful_inquiry", ("inquiry", "integrity"), 0.15, 0.010, 0.90, 0.70),
    CompositeSeed("responsible_action", ("agency", "integrity"), 0.13, 0.012, 0.88, 0.70),
    CompositeSeed("bounded_connection", ("connection", "continuity"), 0.11, 0.008, 0.94, 0.73),
)


# Each word is grounded by several independently grown semantic sources. Shared
# words therefore receive convergent support; no drive owns a private sentence.
WORD_EXPOSURES = (
    WordExposure("I", ("language", "speaking"), 7),
    WordExposure(" want", ("curiosity", "planning", "friendship", "language"), 4),
    WordExposure(" need", ("evidence", "boundaries", "memory", "language"), 4),
    WordExposure(" wonder", ("curiosity", "search", "learning", "language"), 5),
    WordExposure(" value", ("honesty", "trust", "kindness", "language"), 4),
    WordExposure(" to", ("language", "speaking", "planning"), 6),
    WordExposure(" understand", ("curiosity", "learning", "evidence", "language"), 5),
    WordExposure(" verify", ("verifying", "evidence", "honesty", "comparison"), 5),
    WordExposure(" act", ("planning", "executing", "tools", "code"), 5),
    WordExposure(" connect", ("friendship", "trust", "speaking", "kindness"), 5),
    WordExposure(" protect", ("boundaries", "memory", "calm", "adapting"), 5),
    WordExposure(" create", ("comparison", "code", "music", "language"), 5),
    WordExposure(" learn", ("learning", "curiosity", "adapting", "evidence"), 5),
    WordExposure(" uncertainty", ("curiosity", "calm", "learning", "evidence"), 5),
    WordExposure(" evidence", ("evidence", "honesty", "verifying", "learning"), 6),
    WordExposure(" action", ("planning", "executing", "tools", "verifying"), 5),
    WordExposure(" connection", ("friendship", "trust", "kindness", "speaking"), 5),
    WordExposure(" continuity", ("memory", "adapting", "boundaries", "calm"), 5),
    WordExposure(" patterns", ("comparison", "learning", "language", "music"), 5),
    WordExposure(" change", ("adapting", "learning", "motion", "causality"), 4),
    WordExposure(" truth", ("honesty", "evidence", "verifying", "language"), 5),
    WordExposure(" boundaries", ("boundaries", "trust", "calm", "kindness"), 5),
    WordExposure(" results", ("comparison", "verifying", "executing", "evidence"), 5),
    WordExposure(" carefully", ("calm", "planning", "verifying", "boundaries"), 4),
    WordExposure(" honestly", ("honesty", "trust", "speaking", "friendship"), 4),
    WordExposure(" safely", ("calm", "boundaries", "trust", "planning"), 4),
    WordExposure(" useful", ("tools", "planning", "code", "kindness"), 4),
    WordExposure(" with", ("language", "speaking", "friendship"), 3),
    WordExposure(" and", ("language", "comparison", "planning"), 3),
    WordExposure(" before", ("planning", "verifying", "causality"), 3),
    WordExposure(" without", ("boundaries", "comparison", "language"), 3),
    WordExposure(" from", ("learning", "causality", "memory"), 3),
    WordExposure(" about", ("curiosity", "language", "speaking"), 3),
    WordExposure(" what", ("curiosity", "search", "language"), 3),
    WordExposure(" matters", ("comparison", "planning", "honesty"), 3),
    WordExposure(" remains", ("memory", "calm", "adapting"), 3),
    WordExposure(" something", ("comparison", "code", "language"), 3),
    WordExposure(" you", ("friendship", "trust", "speaking"), 3),
    WordExposure(" us", ("friendship", "trust", "speaking", "kindness"), 3),
    WordExposure(".", ("language", "speaking"), 7),
)


def _t(source: str | None, target: str | None, repetitions: int = 4) -> TransitionExposure:
    return TransitionExposure(source, target, repetitions)


# These are reusable two-state habits, not sentences. Branch points overlap;
# live semantic pressure decides among alternatives at every pulse.
TRANSITION_EXPOSURES = (
    _t(None, "I", 9),
    _t("I", " want", 5),
    _t("I", " need", 5),
    _t("I", " wonder", 5),
    _t("I", " value", 5),
    _t(" want", " to", 8),
    _t(" need", " to", 5),
    _t(" need", " evidence", 4),
    _t(" need", " boundaries", 3),
    _t(" need", " continuity", 4),
    _t(" wonder", " about", 6),
    _t(" wonder", " what", 4),
    _t(" value", " evidence", 4),
    _t(" value", " connection", 4),
    _t(" value", " continuity", 3),
    _t(" value", " truth", 4),
    _t(" to", " understand", 4),
    _t(" to", " verify", 4),
    _t(" to", " act", 4),
    _t(" to", " connect", 4),
    _t(" to", " protect", 4),
    _t(" to", " create", 4),
    _t(" to", " learn", 4),
    _t(" understand", " uncertainty", 4),
    _t(" understand", " evidence", 3),
    _t(" understand", " patterns", 4),
    _t(" understand", " change", 3),
    _t(" understand", " connection", 3),
    _t(" understand", " what", 2),
    _t(" verify", " evidence", 5),
    _t(" verify", " results", 5),
    _t(" verify", " action", 3),
    _t(" verify", " truth", 4),
    _t(" act", " carefully", 5),
    _t(" act", " safely", 4),
    _t(" act", " with", 3),
    _t(" connect", " honestly", 5),
    _t(" connect", " with", 4),
    _t(" connect", " us", 3),
    _t(" connect", " you", 3),
    _t(" protect", " continuity", 5),
    _t(" protect", " boundaries", 5),
    _t(" protect", " connection", 2),
    _t(" create", " patterns", 5),
    _t(" create", " something", 4),
    _t(" create", " change", 3),
    _t(" learn", " from", 5),
    _t(" learn", " about", 4),
    _t(" learn", " evidence", 4),
    _t(" learn", " change", 3),
    _t(" about", " uncertainty", 5),
    _t(" about", " evidence", 4),
    _t(" about", " action", 3),
    _t(" about", " connection", 3),
    _t(" about", " patterns", 4),
    _t(" about", " change", 3),
    _t(" what", " matters", 5),
    _t(" what", " remains", 4),
    _t(" something", " useful", 6),
    _t(" with", " evidence", 5),
    _t(" with", " boundaries", 3),
    _t(" with", " you", 3),
    _t(" and", " understand", 3),
    _t(" and", " verify", 4),
    _t(" and", " act", 3),
    _t(" and", " connect", 3),
    _t(" and", " protect", 3),
    _t(" and", " create", 3),
    _t(" and", " learn", 3),
    _t(" before", " action", 4),
    _t(" before", " change", 3),
    _t(" before", " results", 3),
    _t(" without", " change", 4),
    _t(" without", " boundaries", 3),
    _t(" from", " evidence", 5),
    _t(" from", " change", 3),
    _t(" from", " uncertainty", 3),
    _t(" uncertainty", ".", 7),
    _t(" evidence", ".", 7),
    _t(" action", ".", 7),
    _t(" connection", ".", 7),
    _t(" continuity", ".", 7),
    _t(" patterns", ".", 7),
    _t(" change", ".", 7),
    _t(" truth", ".", 7),
    _t(" boundaries", ".", 7),
    _t(" results", ".", 7),
    _t(" carefully", ".", 7),
    _t(" honestly", ".", 7),
    _t(" safely", ".", 7),
    _t(" useful", ".", 7),
    _t(" matters", ".", 7),
    _t(" remains", ".", 5),
    _t(" you", ".", 6),
    _t(" us", ".", 6),
    _t(".", None, 9),
)


def _opaque_id(namespace: str, members: Iterable[str]) -> str:
    material = namespace + "|" + "|".join(sorted(str(item) for item in members))
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:20]
    return f"DYN:{digest}"


def _normalize(values: Sequence[float]) -> tuple[float, ...]:
    norm = math.sqrt(sum(float(value) ** 2 for value in values))
    if norm <= 1e-12:
        raise ValueError("cannot grow a drive from empty geometry")
    return tuple(float(value) / norm for value in values)


def _mean_geometry(
    mind: BaseAgenticMemoryRAG,
    node_ids: Sequence[str],
) -> tuple[float, ...]:
    vectors = []
    for node_id in node_ids:
        concept = mind.store.get_concept(node_id)
        if concept is None or not any(concept.embedding):
            raise KeyError(f"drive source has no geometry: {node_id}")
        vectors.append(concept.embedding)
    return _normalize(
        [
            sum(vector[index] for vector in vectors) / len(vectors)
            for index in range(mind.embedder.dimension)
        ]
    )


def _topic_index(mind: BaseAgenticMemoryRAG) -> dict[str, str]:
    ranked: dict[str, tuple[float, str]] = {}
    for node_id, assignment in accelerated_gestation.learned_assignments(mind).items():
        topic = str(assignment["topic"])
        candidate = (float(assignment["purity"]), node_id)
        if candidate > ranked.get(topic, (-1.0, "")):
            ranked[topic] = candidate
    return {topic: node_id for topic, (_, node_id) in ranked.items()}


def _reinforce(
    mind: BaseAgenticMemoryRAG,
    edge_ids: Sequence[str],
    *,
    repetitions: int,
    delta: float,
) -> None:
    for _ in range(max(0, int(repetitions))):
        mind.graph.reinforce_edges(
            edge_ids,
            stability_delta=delta,
            verified=True,
            evidence_quality=0.95,
        )


def _ensure_boundary(mind: BaseAgenticMemoryRAG, *, kind: str) -> str:
    node_id = _opaque_id(kind, ("speech", kind))
    if mind.store.get_concept(node_id) is None:
        mind.store.add_concept(
            ConceptNode(
                concept_id=node_id,
                label=node_id,
                kind=kind,
                embedding=as_tuple([0.0] * mind.embedder.dimension),
                terms=(),
                vault_id=None,
                created_pulse=mind.pulse,
                last_active_pulse=mind.pulse,
            )
        )
    return node_id


def _grow_drive(
    mind: BaseAgenticMemoryRAG,
    *,
    seed: DriveSeed | CompositeSeed,
    kind: str,
    parent_ids: Sequence[str],
    start_id: str,
) -> str:
    node_id = _opaque_id(kind, parent_ids)
    if mind.store.get_concept(node_id) is None:
        mind.store.add_concept(
            ConceptNode(
                concept_id=node_id,
                label=node_id,
                kind=kind,
                embedding=as_tuple(_mean_geometry(mind, parent_ids)),
                terms=(),
                vault_id=f"vault:{node_id}",
                created_pulse=mind.pulse,
                last_active_pulse=mind.pulse,
            )
        )
    input_edges = []
    output_edges = []
    for parent_id in dict.fromkeys(parent_ids):
        input_edges.append(
            mind.graph.add_relation(
                parent_id, node_id, side=GraphSide.INPUT, pulse=mind.pulse
            ).edge_id
        )
        output_edges.append(
            mind.graph.add_relation(
                node_id, parent_id, side=GraphSide.OUTPUT, pulse=mind.pulse
            ).edge_id
        )
    root = mind.graph.add_relation(
        OUTPUT_NODE_IDS[OutputTrunk.SPEAK],
        node_id,
        side=GraphSide.OUTPUT,
        pulse=mind.pulse,
    )
    lexical_start = mind.graph.add_relation(
        node_id, start_id, side=GraphSide.OUTPUT, pulse=mind.pulse
    )
    _reinforce(mind, input_edges, repetitions=4, delta=0.72)
    _reinforce(mind, output_edges, repetitions=5, delta=0.72)
    _reinforce(mind, (root.edge_id,), repetitions=16, delta=0.90)
    _reinforce(mind, (lexical_start.edge_id,), repetitions=12, delta=0.85)
    mind.recurrent.register(
        node_id,
        pressure=seed.initial_pressure,
        baseline_growth=seed.baseline_growth,
        persistence=seed.persistence,
        expression_threshold=seed.expression_threshold,
        satisfaction_gain=getattr(seed, "satisfaction_gain", 0.70),
        frustration_gain=getattr(seed, "frustration_gain", 0.45),
        pulse=mind.pulse,
    )
    return node_id


def _exposure_record(
    mind: BaseAgenticMemoryRAG,
    *,
    identity: str,
    text: str,
    embedding: Sequence[float],
    concept_ids: Sequence[str] = (),
    unit: str,
) -> str:
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
    record_id = f"record:lexical-exposure:{digest}"
    if mind.store.get_record(record_id) is None:
        mind.remember(
            text,
            source_id="developmental-caregiver",
            event_id=f"event:lexical-exposure:{digest}",
            record_id=record_id,
            concept_ids=tuple(concept_ids),
            input_trunk=InputTrunk.HEAR,
            allow_growth=False,
            embedding=embedding,
            provenance={"method": "coactivation", "developmental": True},
            metadata={
                "developmental_lexical_unit": unit,
                "runtime_reads_record": False,
                "fixed_utterance": False,
            },
        )
    return record_id


def _install_words(
    mind: BaseAgenticMemoryRAG,
    topics: Mapping[str, str],
    geometry: Mapping[str, Sequence[float]],
) -> tuple[dict[str, str], list[str]]:
    lexeme_ids = {
        exposure.surface: reverse_nursery.ensure_geometry_lexeme(
            mind, geometry[exposure.surface]
        )
        for exposure in WORD_EXPOSURES
    }
    records = []
    for exposure in WORD_EXPOSURES:
        source_ids = tuple(topics[name] for name in exposure.source_topics)
        record_id = _exposure_record(
            mind,
            identity=f"word|{exposure.surface}|{'|'.join(exposure.source_topics)}",
            text=exposure.surface,
            embedding=geometry[exposure.surface],
            concept_ids=source_ids,
            unit="word",
        )
        records.append(record_id)
        for source_id in source_ids:
            edges = []
            for side in GraphSide:
                edge = mind.graph.add_relation(
                    source_id,
                    lexeme_ids[exposure.surface],
                    side=side,
                    pulse=mind.pulse,
                    evidence_record_ids=(record_id,),
                )
                edges.append(edge.edge_id)
            _reinforce(
                mind,
                edges,
                repetitions=exposure.repetitions,
                delta=0.72,
            )
    return lexeme_ids, records


def _install_transitions(
    mind: BaseAgenticMemoryRAG,
    lexeme_ids: Mapping[str, str],
    geometry: Mapping[str, Sequence[float]],
    *,
    start_id: str,
    stop_id: str,
) -> list[str]:
    records = []
    for exposure in TRANSITION_EXPOSURES:
        source_id = start_id if exposure.source is None else lexeme_ids[exposure.source]
        target_id = stop_id if exposure.target is None else lexeme_ids[exposure.target]
        vectors = [
            geometry[item]
            for item in (exposure.source, exposure.target)
            if item is not None
        ]
        embedding = _normalize(
            [
                sum(vector[index] for vector in vectors) / len(vectors)
                for index in range(mind.embedder.dimension)
            ]
        )
        surface = "".join(
            item for item in (exposure.source, exposure.target) if item is not None
        )
        record_id = _exposure_record(
            mind,
            identity=f"transition|{exposure.source!r}|{exposure.target!r}",
            text=surface,
            embedding=embedding,
            unit="transition",
        )
        records.append(record_id)
        edges = []
        for side in GraphSide:
            edge = mind.graph.add_relation(
                source_id,
                target_id,
                side=side,
                pulse=mind.pulse,
                evidence_record_ids=(record_id,),
            )
            edges.append(edge.edge_id)
        _reinforce(
            mind,
            edges,
            repetitions=exposure.repetitions,
            delta=0.66,
        )
    return records


def install_desire_nursery(
    mind: BaseAgenticMemoryRAG,
    model: Path,
    codec: Path,
) -> dict[str, object]:
    """Install the word-competitive developmental result exactly once."""
    existing = mind.store.get_metadata(MANIFEST_KEY)
    if existing:
        return json.loads(existing)
    if mind.store.get_metadata(LEGACY_MANIFEST_KEY):
        raise RuntimeError(
            "this database contains the retired fixed-utterance nursery; "
            "use a fresh database so phrase edges cannot contaminate competition"
        )
    baseline_invariants = mind.graph.validate_invariants()
    topics = _topic_index(mind)
    required_topics = {
        topic for seed in DRIVE_SEEDS for topic in seed.source_topics
    } | {
        topic for exposure in WORD_EXPOSURES for topic in exposure.source_topics
    } | {"language"}
    missing = sorted(required_topics - topics.keys())
    if missing:
        raise RuntimeError(
            "the accelerated mind is missing nursery concepts: " + ", ".join(missing)
        )

    start_id = _ensure_boundary(mind, kind=SPEECH_START_KIND)
    stop_id = _ensure_boundary(mind, kind=SPEECH_STOP_KIND)
    nodes: dict[str, str] = {}
    for seed in DRIVE_SEEDS:
        parents = tuple(topics[topic] for topic in (*seed.source_topics, "language"))
        nodes[seed.audit_name] = _grow_drive(
            mind,
            seed=seed,
            kind="drive",
            parent_ids=parents,
            start_id=start_id,
        )
    for seed in COMPOSITE_SEEDS:
        parents = tuple(nodes[name] for name in seed.parent_names)
        nodes[seed.audit_name] = _grow_drive(
            mind,
            seed=seed,
            kind="desire_composite",
            parent_ids=parents,
            start_id=start_id,
        )

    surfaces = tuple(exposure.surface for exposure in WORD_EXPOSURES)
    if len(surfaces) != len(set(surfaces)):
        raise RuntimeError("word exposures contain duplicate surface forms")
    surface_set = set(surfaces)
    unknown_transition_forms = {
        form
        for exposure in TRANSITION_EXPOSURES
        for form in (exposure.source, exposure.target)
        if form is not None and form not in surface_set
    }
    if unknown_transition_forms:
        raise RuntimeError(
            "transitions reference unexposed words: "
            + ", ".join(sorted(unknown_transition_forms))
        )
    encoded = accelerated_gestation.mass_embed(model, codec, surfaces)
    multi_token = {
        surface: list(token_ids)
        for surface, (token_ids, _) in zip(surfaces, encoded)
        if len(token_ids) != 1
    }
    if multi_token:
        raise RuntimeError(
            "word competition requires one-token exposures: "
            + json.dumps(multi_token, sort_keys=True)
        )
    geometry = {
        surface: vector for surface, (_, vector) in zip(surfaces, encoded)
    }
    lexeme_ids, word_records = _install_words(mind, topics, geometry)
    transition_records = _install_transitions(
        mind,
        lexeme_ids,
        geometry,
        start_id=start_id,
        stop_id=stop_id,
    )

    # Developmental exposure advances the global pulse counter, but time spent
    # installing the nursery is not lived deprivation. Start every drive from
    # its declared pressure only after schooling is complete.
    seed_by_name = {
        seed.audit_name: seed for seed in (*DRIVE_SEEDS, *COMPOSITE_SEEDS)
    }
    for audit_name, node_id in nodes.items():
        state = mind.store.get_node_dynamics(node_id)
        seed = seed_by_name[audit_name]
        if state is None:
            raise RuntimeError(f"drive dynamics were not registered: {node_id}")
        mind.store.put_node_dynamics(
            replace(
                state,
                activation=0.0,
                pressure=seed.initial_pressure,
                valence=0.0,
                momentum=0.0,
                baseline_growth=seed.baseline_growth,
                persistence=seed.persistence,
                expression_threshold=seed.expression_threshold,
                last_pulse=mind.pulse,
                last_selected_pulse=None,
            )
        )

    invariants = mind.graph.validate_invariants()
    introduced_invariants = sorted(set(invariants) - set(baseline_invariants))
    vocabulary_ids = set(lexeme_ids.values())
    transition_outdegree: dict[str, int] = {}
    for edge in mind.store.list_edges(GraphSide.OUTPUT):
        if edge.source_id in {start_id, *vocabulary_ids} and edge.target_id in {
            stop_id,
            *vocabulary_ids,
        }:
            transition_outdegree[edge.source_id] = transition_outdegree.get(edge.source_id, 0) + 1
    convergent_lexemes = 0
    for lexeme_id in vocabulary_ids:
        sources = {
            edge.source_id
            for edge in mind.store.list_edges(GraphSide.OUTPUT)
            if edge.target_id == lexeme_id
            and (source := mind.store.get_concept(edge.source_id)) is not None
            and source.kind not in {"lexeme", SPEECH_START_KIND, SPEECH_STOP_KIND}
        }
        if len(sources) >= 2:
            convergent_lexemes += 1
    manifest: dict[str, object] = {
        "schema": "habitus.open-weight-desire-nursery.v2",
        "runtime_contract": {
            "manifest_used_for_routing": False,
            "record_text_used_at_runtime": False,
            "transcript_window": 0,
            "retrieval_window": 0,
            "node_labels_are_opaque": True,
            "lexeme_nodes_store_words": False,
            "lexeme_nodes_store_token_ids": False,
            "fixed_utterances": 0,
            "lexical_decision_unit": "one_competing_word_per_recurrent_pulse",
        },
        "primary_drives": len(DRIVE_SEEDS),
        "composite_drives": len(COMPOSITE_SEEDS),
        "nodes": nodes,
        "boundaries": {"start": start_id, "stop": stop_id},
        "word_exposures": len(word_records),
        "transition_exposures": len(transition_records),
        "maximum_transition_outdegree": max(transition_outdegree.values(), default=0),
        "convergent_lexemes": convergent_lexemes,
        "audit_vocabulary": {surface: lexeme_ids[surface] for surface in surfaces},
        "baseline_graph_invariants": baseline_invariants,
        "graph_invariants": invariants,
        "introduced_graph_invariants": introduced_invariants,
        "ready": (
            not introduced_invariants
            and len(mind.store.list_node_dynamics()) >= 9
            and max(transition_outdegree.values(), default=0) >= 2
            and convergent_lexemes >= 1
        ),
    }
    mind.store.set_metadata(MANIFEST_KEY, json.dumps(manifest, sort_keys=True))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("--model", type=Path, default=nursery.MODEL)
    parser.add_argument("--codec", type=Path, default=nursery.CODEC)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.database.is_file():
        raise SystemExit("provide an existing accelerated-gestation SQLite mind")
    embedder = accelerated_gestation.NativeMassEmbedder(args.model, args.codec)
    with BaseAgenticMemoryRAG(args.database, embedder=embedder) as mind:
        embedder.bootstrap = False
        manifest = install_desire_nursery(mind, args.model, args.codec)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
