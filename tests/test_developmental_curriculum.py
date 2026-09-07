from __future__ import annotations

import hashlib
from pathlib import Path

from habitus_ai import BaseAgenticMemoryRAG, DeterministicHashEmbedder
from habitus_ai.developmental_curriculum import (
    BYTE_LEXEME_CANDIDATE_KIND,
    BYTE_LEXEME_KIND,
    ByteFormLearner,
    CurriculumStage,
    DevelopmentalCurriculum,
    DevelopmentalMetrics,
    EpisodeKind,
)
from habitus_ai.types import GraphSide, InputTrunk, OutputTrunk


def _mind(path: Path | str = ":memory:") -> BaseAgenticMemoryRAG:
    return BaseAgenticMemoryRAG(path, embedder=DeterministicHashEmbedder(32))


def _record(mind: BaseAgenticMemoryRAG, text: str, record_id: str):
    pulse, _ = mind.begin_pulse()
    return mind.remember(
        text,
        record_id=record_id,
        event_id=f"event:{record_id}",
        input_trunk=InputTrunk.HEAR,
        embedding=[0.0] * mind.embedder.dimension,
        allow_growth=False,
        pulse_number=pulse,
    )


def test_language_and_narratives_unlock_only_one_measured_stage_at_a_time() -> None:
    with _mind() as mind:
        mind.add_concept(
            "lived:shape",
            "opaque-shape",
            input_trunks=(InputTrunk.SEE,),
            output_trunks=(OutputTrunk.LOOK,),
            semantic_embedding=False,
        )
        curriculum = DevelopmentalCurriculum(mind)
        early = _record(mind, "a story before grounding", "early")
        rejected = curriculum.admit(
            early,
            episode_kind=EpisodeKind.EXTERNAL_NARRATIVE,
            route_ids=("lived:shape",),
        )
        assert rejected.accepted is False
        assert curriculum.stage == CurriculumStage.PRELINGUISTIC

        perfect = DevelopmentalMetrics(
            nonverbal_discrimination=1.0,
            consequence_prediction=1.0,
            lexical_recall_at_5=1.0,
            lexical_shuffled_gap=1.0,
            grounded_action_accuracy=1.0,
            grounding_coverage=1.0,
            narrative_route_recall=1.0,
        )
        assert curriculum.update_metrics(perfect) == CurriculumStage.GROUNDED_FORMS
        # A single metric report cannot jump over developmental stages.
        assert curriculum.stage == CurriculumStage.GROUNDED_FORMS
        grounded = _record(mind, "shape", "grounded")
        assert curriculum.admit(
            grounded,
            episode_kind=EpisodeKind.GROUNDED_LABEL,
            route_ids=("lived:shape",),
        ).accepted
        assert curriculum.admit(
            grounded,
            episode_kind=EpisodeKind.GROUNDED_LABEL,
            route_ids=(),
        ).accepted is False

        assert curriculum.update_metrics(perfect) == CurriculumStage.FUNCTIONAL_EXCHANGE
        assert curriculum.update_metrics(perfect) == CurriculumStage.EXPERIENCED_NARRATIVE
        missing = curriculum.admit(
            grounded,
            episode_kind=EpisodeKind.EXPERIENCED_NARRATIVE,
            route_ids=("absent:route",),
        )
        assert missing.accepted is False
        assert curriculum.update_metrics(perfect) == CurriculumStage.BROADER_NARRATIVE
        assert curriculum.admit(
            grounded,
            episode_kind=EpisodeKind.EXTERNAL_NARRATIVE,
            route_ids=("lived:shape",),
        ).accepted
        assert mind.store.connection.execute(
            "SELECT COUNT(*) FROM developmental_stage_transitions"
        ).fetchone()[0] == 4


def test_recurrent_byte_forms_promote_from_diverse_grounded_experience() -> None:
    with _mind() as mind:
        for route in ("lived:red", "lived:blue"):
            mind.add_concept(
                route,
                route.replace("lived:", "opaque-"),
                input_trunks=(InputTrunk.SEE,),
                output_trunks=(OutputTrunk.LOOK,),
                semantic_embedding=False,
            )
            # Independent geometry comes from lived sensor state, not spelling.
            mind.store.update_concept_embedding(
                route, mind.embedder.embed(f"numeric route {route[-1]}")
            )
        learner = ByteFormLearner(mind, maximum_candidates_per_record=512)
        receipts = []
        episodes = (
            ("see red now", "lived:red"),
            ("red moves", "lived:red"),
            ("touch red", "lived:red"),
            ("see blue now", "lived:blue"),
            ("blue moves", "lived:blue"),
            ("touch blue", "lived:blue"),
        )
        for index, (text, route) in enumerate(episodes):
            record = _record(mind, text, f"byte:{index}")
            receipts.append(
                learner.observe(record, route_ids=(route,), pulse=mind.pulse)
            )

        promoted_forms = {
            form_id
            for receipt in receipts
            for form_id in receipt.promoted_form_ids
        }
        red_id = "byte-form:" + hashlib.sha256(b"red").hexdigest()[:32]
        blue_id = "byte-form:" + hashlib.sha256(b"blue").hexdigest()[:32]
        assert red_id in promoted_forms
        assert blue_id in promoted_forms
        assert learner.audit_surface(red_id) == b"red"
        red = learner.statistics(red_id)
        assert red.kind == BYTE_LEXEME_KIND
        assert red.independent_records == 3
        assert red.grounding_score == 1.0
        assert red.grounding_gap >= 0.49
        node = mind.store.get_concept(red.node_id)
        assert node is not None
        assert node.kind == BYTE_LEXEME_KIND
        assert node.terms == ()
        assert "red" not in node.label
        assert any(
            edge.target_id == node.concept_id and not edge.archived
            for edge in mind.store.list_edges(GraphSide.INPUT, include_archived=True)
        )

        common_id = "byte-form:" + hashlib.sha256(b" moves").hexdigest()[:32]
        common = learner.statistics(common_id)
        assert common.kind == BYTE_LEXEME_CANDIDATE_KIND
        assert common.grounding_gap < 0.20


def test_one_exposure_never_creates_a_lexical_graph_node() -> None:
    with _mind() as mind:
        mind.add_concept(
            "lived:one",
            "opaque-one",
            input_trunks=(InputTrunk.SEE,),
            semantic_embedding=False,
        )
        learner = ByteFormLearner(mind)
        record = _record(mind, "novel carrier", "single")
        receipt = learner.observe(
            record, route_ids=("lived:one",), pulse=mind.pulse
        )
        assert receipt.candidate_node_ids == ()
        assert mind.store.list_concepts(kind=BYTE_LEXEME_CANDIDATE_KIND) == []


def test_only_promoted_whole_utterances_become_contextual_motor_forms() -> None:
    with _mind() as mind:
        for route in ("lived:a", "lived:b"):
            mind.add_concept(
                route,
                f"opaque-{route[-1]}",
                input_trunks=(InputTrunk.SEE,),
                output_trunks=(OutputTrunk.SPEAK,),
                semantic_embedding=False,
            )
            mind.store.update_concept_embedding(
                route, mind.embedder.embed(f"numeric-{route[-1]}")
            )
        learner = ByteFormLearner(mind, maximum_candidates_per_record=512)
        for index in range(3):
            learner.observe(
                _record(mind, "mava", f"whole-a:{index}"),
                route_ids=("lived:a",),
                pulse=mind.pulse,
            )
            learner.observe(
                _record(mind, "telu", f"whole-b:{index}"),
                route_ids=("lived:b",),
                pulse=mind.pulse,
            )

        options_a = learner.available_motor_forms(("lived:a",))
        options_b = learner.available_motor_forms(("lived:b",))
        assert options_a[0].payload == b"mava"
        assert options_a[0].route_alignment == 1.0
        assert options_b[0].payload == b"telu"
        assert options_b[0].route_alignment == 1.0
        assert {item.payload for item in options_a} == {b"mava", b"telu"}

        substring_id = "byte-form:" + hashlib.sha256(b"av").hexdigest()[:32]
        assert learner.statistics(substring_id).kind != BYTE_LEXEME_KIND
        assert mind.store.connection.execute(
            "SELECT 1 FROM developmental_motor_forms WHERE form_id = ?",
            (substring_id,),
        ).fetchone() is None


def test_receptive_only_language_is_recognized_but_never_becomes_a_motor_form(
    monkeypatch,
) -> None:
    with _mind() as mind:
        for route in ("lived:question", "lived:statement"):
            mind.add_concept(
                route,
                f"opaque-{route.rsplit(':', 1)[-1]}",
                input_trunks=(InputTrunk.NOTICE,),
                semantic_embedding=False,
            )
            mind.store.update_concept_embedding(
                route, mind.embedder.embed(f"numeric-{route}")
            )
        learner = ByteFormLearner(mind, maximum_candidates_per_record=512)
        for index in range(3):
            learner.observe(
                _record(mind, "what file is open", f"question:{index}"),
                route_ids=("lived:question",),
                pulse=mind.pulse,
                motor_eligible=False,
            )
            learner.observe(
                _record(mind, "the script is ready", f"statement:{index}"),
                route_ids=("lived:statement",),
                pulse=mind.pulse,
                motor_eligible=True,
            )

        question_id = "byte-form:" + hashlib.sha256(
            b"what file is open"
        ).hexdigest()[:32]
        assert learner.statistics(question_id).kind == BYTE_LEXEME_KIND
        assert mind.store.connection.execute(
            "SELECT 1 FROM developmental_motor_forms WHERE form_id = ?",
            (question_id,),
        ).fetchone() is None
        question_node_id = learner.statistics(question_id).node_id
        statement_id = "byte-form:" + hashlib.sha256(
            b"the script is ready"
        ).hexdigest()[:32]
        statement_node_id = learner.statistics(statement_id).node_id
        output_edges = mind.store.list_edges(
            GraphSide.OUTPUT, include_archived=True
        )
        assert not any(
            edge.target_id == question_node_id for edge in output_edges
        )
        assert any(
            edge.target_id == statement_node_id for edge in output_edges
        )

        def forbidden_record_read(*_args, **_kwargs):
            raise AssertionError("recognition must not recover canonical text")

        monkeypatch.setattr(mind.store, "get_record", forbidden_record_read)
        recognized = learner.recognize(b"please say what file is open now")
        assert recognized
        assert any(item.byte_length >= len(b"what file") for item in recognized)
        assert all(
            mind.store.get_concept(item.node_id) is not None
            for item in recognized
        )
