from __future__ import annotations

from habitus_ai import (
    BaseAgenticMemoryRAG,
    DeterministicHashEmbedder,
    DevelopmentalLanguage,
    EXPERIENTIAL_CANDIDATE_KIND,
    EXPERIENTIAL_PATTERN_KIND,
    GraphSide,
    InputTrunk,
    NoContextPlanner,
    SENSORY_FEATURE_KIND,
)
from habitus_ai.open_weight import (
    SPEECH_START_KIND,
    SPEECH_STOP_KIND,
    lexical_geometry_id,
)
from habitus_ai.types import ConceptNode, as_tuple


def _mind() -> BaseAgenticMemoryRAG:
    return BaseAgenticMemoryRAG(
        ":memory:", embedder=DeterministicHashEmbedder(32)
    )


def _install_boundaries(mind: BaseAgenticMemoryRAG) -> None:
    for node_id, kind in (
        ("speech-boundary:start", SPEECH_START_KIND),
        ("speech-boundary:stop", SPEECH_STOP_KIND),
    ):
        mind.store.add_concept(
            ConceptNode(
                concept_id=node_id,
                label=node_id,
                kind=kind,
                embedding=as_tuple([0.0] * mind.embedder.dimension),
                terms=(),
                vault_id=None,
                created_pulse=0,
                last_active_pulse=0,
            )
        )


def _planner(mind: BaseAgenticMemoryRAG) -> NoContextPlanner:
    _install_boundaries(mind)
    language = DevelopmentalLanguage(
        mind,
        promotion_exposures=2,
        transition_promotion_exposures=2,
        maximum_semantic_contexts=12,
    )
    return NoContextPlanner(
        mind,
        lexical_sensor=lambda text: tuple(
            mind.embedder.embed(f"heard-unit:{unit}")
            for unit in text.casefold().split()
        ),
        language_learner=language,
    )


def test_one_sensation_grows_only_bounded_opaque_routes_from_its_trunk() -> None:
    with _mind() as mind:
        record = mind.remember(
            "transport text must not become the concept label",
            input_trunk=InputTrunk.SEE,
            embedding=mind.embedder.embed("round bright left"),
            allow_growth=False,
        )
        growth = mind.graph.grow_trunk_breadth(
            record,
            input_trunk=InputTrunk.SEE,
            pulse=mind.pulse,
        )

        assert len(growth.feature_node_ids) == 6
        assert [
            node_id.split(":")[4] for node_id in growth.feature_node_ids
        ] == ["b03", "b04", "b05", "b06", "b07", "b08"]
        assert len(growth.candidate_pattern_node_ids) == 4
        assert growth.promoted_pattern_node_ids == ()
        assert growth.cross_trunk is False
        assert mind.store.list_concepts(kind="crown") == []
        for node_id in (
            *growth.feature_node_ids,
            *growth.candidate_pattern_node_ids,
        ):
            concept = mind.store.get_concept(node_id)
            assert concept is not None
            assert concept.terms == ()
            assert "transport" not in concept.label
        for node_id in growth.feature_node_ids:
            trace = mind.graph.traverse(
                pulse_id=f"root-proof:{node_id}",
                side=GraphSide.INPUT,
                target_id=node_id,
                endpoint_score=1.0,
                required_input_trunk=InputTrunk.SEE,
                mark_active=False,
            )
            assert trace is not None
            assert trace.path_node_ids[:2] == ("SELF", "IN:SEE")
        assert all(
            mind.store.get_edge(edge_id).archived
            for pattern_id in growth.candidate_pattern_node_ids
            for edge_id in growth.edge_ids
            if mind.store.get_edge(edge_id).target_id == pattern_id
        )
        assert mind.graph.validate_invariants() == []


def test_repeated_numeric_intersection_promotes_a_route_without_adding_words() -> None:
    with _mind() as mind:
        vector = mind.embedder.embed("stable numeric shape")
        receipts = []
        for index in range(2):
            record = mind.remember(
                f"opaque carrier {index}",
                record_id=f"seen:{index}",
                event_id=f"seen-event:{index}",
                input_trunk=InputTrunk.SEE,
                embedding=vector,
                allow_growth=False,
            )
            receipts.append(
                mind.graph.grow_trunk_breadth(
                    record,
                    input_trunk=InputTrunk.SEE,
                    pulse=mind.pulse,
                )
            )

        first, second = receipts
        assert set(first.candidate_pattern_node_ids) == set(
            second.promoted_pattern_node_ids
        )
        assert second.candidate_pattern_node_ids == ()
        for node_id in second.promoted_pattern_node_ids:
            concept = mind.store.get_concept(node_id)
            assert concept is not None
            assert concept.kind == EXPERIENTIAL_PATTERN_KIND
            assert concept.terms == ()
            trace = mind.graph.traverse(
                pulse_id=f"promoted:{node_id}",
                side=GraphSide.INPUT,
                target_id=node_id,
                endpoint_score=1.0,
                required_input_trunk=InputTrunk.SEE,
                mark_active=False,
            )
            assert trace is not None
        assert mind.store.list_concepts(kind=EXPERIENTIAL_CANDIDATE_KIND) == []


def test_repeated_co_settling_grows_one_pattern_reachable_from_both_senses() -> None:
    with _mind() as mind:
        cross_receipts = []
        for frame in range(2):
            pulse, _ = mind.begin_pulse()
            seen = mind.remember(
                f"seen carrier {frame}",
                record_id=f"frame:{frame}:see",
                event_id=f"frame:{frame}:see:event",
                input_trunk=InputTrunk.SEE,
                embedding=mind.embedder.embed("blue round object"),
                allow_growth=False,
                pulse_number=pulse,
            )
            heard = mind.remember(
                f"heard carrier {frame}",
                record_id=f"frame:{frame}:hear",
                event_id=f"frame:{frame}:hear:event",
                input_trunk=InputTrunk.HEAR,
                embedding=mind.embedder.embed("blue round object"),
                allow_growth=False,
                pulse_number=pulse,
            )
            sensory = (
                mind.graph.grow_trunk_breadth(
                    seen,
                    input_trunk=InputTrunk.SEE,
                    pulse=pulse,
                ),
                mind.graph.grow_trunk_breadth(
                    heard,
                    input_trunk=InputTrunk.HEAR,
                    pulse=pulse,
                ),
            )
            cross = mind.graph.grow_cross_trunk_breadth(
                sensory,
                anchor_record=heard,
                pulse=pulse,
            )
            assert cross is not None
            cross_receipts.append(cross)

        first, second = cross_receipts
        assert set(first.candidate_pattern_node_ids) == set(
            second.promoted_pattern_node_ids
        )
        for node_id in second.promoted_pattern_node_ids:
            for trunk in (InputTrunk.HEAR, InputTrunk.SEE):
                trace = mind.graph.traverse(
                    pulse_id=f"cross:{trunk.value}:{node_id}",
                    side=GraphSide.INPUT,
                    target_id=node_id,
                    endpoint_score=1.0,
                    required_input_trunk=trunk,
                    mark_active=False,
                )
                assert trace is not None
                assert trace.path_node_ids[1] == f"IN:{trunk.value}"
        assert mind.graph.validate_invariants() == []


def test_live_language_binds_to_self_grown_patterns_not_text_crowns() -> None:
    with _mind() as mind:
        planner = _planner(mind)
        latest = None
        for index in range(2):
            visual = planner.pulse_kernel.enqueue_input(
                "visual transport only",
                lane=InputTrunk.SEE,
                source_id="nursery-camera",
                embedding=mind.embedder.embed("small blue sphere"),
                item_id=f"visual:{index}",
            )
            latest = planner.hear(
                "blue sphere rolls",
                source_id="caregiver",
                additional_input_item_ids=(visual.item_id,),
                input_item_id=f"heard:{index}",
            )

        assert latest is not None and latest.acquisition is not None
        assert mind.store.list_concepts(kind="crown") == []
        cross_growth = [
            item for item in latest.developmental_growth if item.cross_trunk
        ]
        assert len(cross_growth) == 1
        assert cross_growth[0].promoted_pattern_node_ids
        lexical_ids = set(latest.acquisition.lexical_node_ids)
        contextual_edges = [
            edge
            for edge in mind.store.list_edges(GraphSide.OUTPUT)
            if edge.target_id in lexical_ids
            and (source := mind.store.get_concept(edge.source_id)) is not None
            and source.kind in {SENSORY_FEATURE_KIND, EXPERIENTIAL_PATTERN_KIND}
        ]
        assert contextual_edges
        assert any(
            edge.source_id in cross_growth[0].promoted_pattern_node_ids
            for edge in contextual_edges
        )
        assert all(
            concept.terms == ()
            for kind in (SENSORY_FEATURE_KIND, EXPERIENTIAL_PATTERN_KIND)
            for concept in mind.store.list_concepts(kind=kind)
        )
        assert mind.graph.validate_invariants() == []


def test_varied_experience_produces_broad_but_per_turn_bounded_word_web() -> None:
    with _mind() as mind:
        planner = _planner(mind)
        phrases = (
            "blue sphere rolls",
            "blue cube rests",
            "red sphere stops",
            "red cube rolls",
            "soft bell rings",
            "bright bell stops",
        )
        for repetition in range(2):
            for index, phrase in enumerate(phrases):
                plan = planner.hear(
                    phrase,
                    source_id="caregiver",
                    input_item_id=f"breadth:{repetition}:{index}",
                )
                local = [
                    item for item in plan.developmental_growth if not item.cross_trunk
                ]
                assert len(local) == 1
                assert len(local[0].feature_node_ids) <= 6
                assert (
                    len(local[0].candidate_pattern_node_ids)
                    + len(local[0].promoted_pattern_node_ids)
                    <= 4
                )
            if repetition == 0:
                # Related but nonidentical experiences already share coarse
                # receptor intersections; promotion is not sentence matching.
                assert mind.store.list_concepts(
                    kind=EXPERIENTIAL_PATTERN_KIND
                )

        features = mind.store.list_concepts(kind=SENSORY_FEATURE_KIND)
        patterns = mind.store.list_concepts(kind=EXPERIENTIAL_PATTERN_KIND)
        lexemes = mind.store.list_concepts(kind="lexeme")
        assert len(features) > 6
        assert len(patterns) > 4
        assert len(lexemes) >= 8
        lexical_ids = {concept.concept_id for concept in lexemes}
        associations = [
            edge
            for edge in mind.store.list_edges(GraphSide.OUTPUT)
            if edge.target_id in lexical_ids
            and (source := mind.store.get_concept(edge.source_id)) is not None
            and source.kind in {SENSORY_FEATURE_KIND, EXPERIENTIAL_PATTERN_KIND}
        ]
        supported_words = {edge.target_id for edge in associations}
        assert len(associations) > len(lexemes)
        assert len(supported_words) >= len(lexemes) // 2
        blue_id = lexical_geometry_id(mind.embedder.embed("heard-unit:blue"))
        blue_contexts = {
            edge.source_id for edge in associations if edge.target_id == blue_id
        }
        # The repeated word is not owned by one sentence or one concept.  It is
        # a shared membrane state reached by several self-grown lower routes.
        assert len(blue_contexts) >= 4
        assert mind.store.list_concepts(kind="crown") == []
        assert mind.graph.validate_invariants() == []
