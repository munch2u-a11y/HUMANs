from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from habitus_ai import BaseAgenticMemoryRAG, DeterministicHashEmbedder
from habitus_ai.graph import OUTPUT_NODE_IDS
from habitus_ai.open_weight import (
    DevelopmentalLanguage,
    LEXEME_CANDIDATE_KIND,
    LexicalSensation,
    SPEECH_START_KIND,
    SPEECH_STOP_KIND,
    NoContextPlanner,
    OpenWeightFocus,
    OpenWeightTrajectory,
    build_native_state_frame,
    compete_lexical_step,
    lexical_geometry_id,
    resolve_output_focus,
    write_native_state_packet,
)
from habitus_ai.types import (
    ConceptNode,
    ExperienceCycle,
    GraphSide,
    InputTrunk,
    OutputTrunk,
    as_tuple,
)


def _mind(path: Path) -> BaseAgenticMemoryRAG:
    mind = BaseAgenticMemoryRAG(path, embedder=DeterministicHashEmbedder(32))
    for index, (node_id, pressure) in enumerate(
        (("drive:a", 0.72), ("drive:b", 0.18))
    ):
        vector = mind.embedder.embed(f"desire geometry {index}")
        mind.graph.add_concept(
            node_id,
            f"opaque-{index}",
            embedding=vector,
            input_trunks=(InputTrunk.HEAR,),
            output_trunks=(OutputTrunk.SPEAK,),
            kind="drive",
        )
        lexeme_id = f"lexeme:{index}"
        mind.store.add_concept(
            ConceptNode(
                concept_id=lexeme_id,
                label=lexeme_id,
                kind="lexeme",
                embedding=as_tuple(mind.embedder.embed(f"surface {index}")),
                terms=(),
                vault_id=None,
                created_pulse=0,
                last_active_pulse=0,
            )
        )
        mind.graph.add_relation(node_id, lexeme_id, side=GraphSide.OUTPUT)
        mind.recurrent.register(
            node_id,
            pressure=pressure,
            baseline_growth=0.04,
            persistence=0.80,
            expression_threshold=0.55,
        )
    return mind


def _word_competition_mind(path: Path):
    mind = BaseAgenticMemoryRAG(path, embedder=DeterministicHashEmbedder(32))
    mind.graph.add_concept(
        "drive:speech",
        "opaque-drive",
        embedding=mind.embedder.embed("drive geometry"),
        input_trunks=(InputTrunk.HEAR,),
        output_trunks=(OutputTrunk.SPEAK,),
        kind="drive",
    )
    for node_id in ("semantic:a", "semantic:b"):
        mind.graph.add_concept(
            node_id,
            node_id,
            embedding=mind.embedder.embed(node_id),
            input_trunks=(InputTrunk.HEAR,),
            kind="crown",
        )
        mind.graph.add_relation("drive:speech", node_id, side=GraphSide.OUTPUT)
        mind.recurrent.register(node_id, persistence=0.90)
    for node_id, kind in (
        ("boundary:start", SPEECH_START_KIND),
        ("boundary:stop", SPEECH_STOP_KIND),
    ):
        mind.store.add_concept(
            ConceptNode(
                concept_id=node_id,
                label=node_id,
                kind=kind,
                embedding=as_tuple([0.0] * 32),
                terms=(),
                vault_id=None,
                created_pulse=0,
                last_active_pulse=0,
            )
        )
    mind.graph.add_relation(
        "drive:speech", "boundary:start", side=GraphSide.OUTPUT
    )
    lexemes = {}
    for name in ("shared", "solo", "alpha", "beta"):
        node_id = f"lexeme:{name}"
        mind.store.add_concept(
            ConceptNode(
                concept_id=node_id,
                label=node_id,
                kind="lexeme",
                embedding=as_tuple(mind.embedder.embed(f"lexical {name}")),
                terms=(),
                vault_id=None,
                created_pulse=0,
                last_active_pulse=0,
            )
        )
        mind.graph.add_relation("boundary:start", node_id, side=GraphSide.OUTPUT)
        lexemes[name] = node_id
    mind.store.add_concept(
        ConceptNode(
            concept_id="lexeme:cursor",
            label="lexeme:cursor",
            kind="lexeme",
            embedding=as_tuple(mind.embedder.embed("lexical cursor")),
            terms=(),
            vault_id=None,
            created_pulse=0,
            last_active_pulse=0,
        )
    )
    mind.graph.add_relation(
        "lexeme:cursor", lexemes["alpha"], side=GraphSide.OUTPUT
    )
    mind.graph.add_relation(
        "lexeme:cursor", lexemes["beta"], side=GraphSide.OUTPUT
    )
    mind.graph.add_relation("semantic:a", lexemes["shared"], side=GraphSide.OUTPUT)
    mind.graph.add_relation("semantic:b", lexemes["shared"], side=GraphSide.OUTPUT)
    mind.graph.add_relation("semantic:a", lexemes["solo"], side=GraphSide.OUTPUT)
    mind.graph.add_relation("semantic:a", lexemes["alpha"], side=GraphSide.OUTPUT)
    mind.graph.add_relation("semantic:b", lexemes["beta"], side=GraphSide.OUTPUT)
    mind.recurrent.register(
        "drive:speech", pressure=0.9, expression_threshold=0.5
    )
    root = mind.store.find_edge(
        GraphSide.OUTPUT,
        OUTPUT_NODE_IDS[OutputTrunk.SPEAK],
        "drive:speech",
    )
    assert root is not None
    focus = OpenWeightFocus(
        pulse_id="lexical-test",
        candidates=(),
        trunk_probabilities={OutputTrunk.SPEAK.value: 1.0},
        selected=OpenWeightTrajectory(
            trunk=OutputTrunk.SPEAK,
            terminal_node_id="drive:speech",
            path_node_ids=(
                "SELF",
                OUTPUT_NODE_IDS[OutputTrunk.SPEAK],
                "drive:speech",
            ),
            path_edge_ids=(
                mind.store.find_edge(
                    GraphSide.OUTPUT,
                    "SELF",
                    OUTPUT_NODE_IDS[OutputTrunk.SPEAK],
                ).edge_id,
                root.edge_id,
            ),
            path_score=1.0,
            transient_pull=1.0,
            within_trunk_probability=1.0,
            trunk_probability=1.0,
            effective_probability=1.0,
        ),
    )
    return mind, focus, lexemes


def test_desire_pressure_accumulates_and_survives_restart(tmp_path: Path) -> None:
    database = tmp_path / "mind.sqlite"
    with _mind(database) as mind:
        before = mind.recurrent.snapshot(pulse=mind.pulse)
        pulse, pulse_id = mind._next_pulse()
        after = mind.recurrent.advance(pulse=pulse, pulse_id=pulse_id)
        assert after.desires[0].pressure > before.desires[0].pressure
        assert after.should_express is True
        expected_hash = after.state_sha256

    with BaseAgenticMemoryRAG(
        database, embedder=DeterministicHashEmbedder(32)
    ) as reopened:
        observed = reopened.recurrent.snapshot(pulse=reopened.pulse)
        assert observed.state_sha256 == expected_hash
        assert observed.desires[0].node_id == "drive:a"


def test_target_free_output_follows_live_desire_not_supplied_endpoint(tmp_path: Path) -> None:
    with _mind(tmp_path / "focus.sqlite") as mind:
        pulse, pulse_id = mind._next_pulse()
        state = mind.recurrent.advance(pulse=pulse, pulse_id=pulse_id)
        focus = resolve_output_focus(mind, state, pulse_id=pulse_id)

        assert focus.selected is not None
        assert focus.selected.terminal_node_id == "drive:a"
        assert focus.selected.path_node_ids[:2] == (
            "SELF",
            OUTPUT_NODE_IDS[OutputTrunk.SPEAK],
        )
        assert focus.selected.path_node_ids[-1] == state.dominant_desire_id


def test_two_semantic_branches_converge_on_one_word_without_reading_records(
    tmp_path: Path,
) -> None:
    mind, focus, lexemes = _word_competition_mind(
        tmp_path / "convergence.sqlite"
    )
    try:
        neutral = mind.embedder.embed("same lexical comparison geometry")
        for node_id in (lexemes["shared"], lexemes["solo"]):
            mind.store.update_concept_embedding(node_id, neutral)
        for node_id in ("semantic:a", "semantic:b"):
            state = mind.store.get_node_dynamics(node_id)
            assert state is not None
            mind.store.put_node_dynamics(replace(state, activation=1.0))

        def forbidden(*_args, **_kwargs):
            raise AssertionError("lexical competition cannot inspect stored text")

        mind.store.list_records = forbidden  # type: ignore[method-assign]
        mind.store.get_records = forbidden  # type: ignore[method-assign]
        mind.store.get_record = forbidden  # type: ignore[method-assign]
        recurrent = mind.recurrent.snapshot(pulse=mind.pulse)
        winner, candidates = compete_lexical_step(
            mind,
            recurrent,
            focus,
            cursor_node_id="boundary:start",
            step_index=0,
            minimum_words=0,
        )

        by_id = {candidate.node_id: candidate for candidate in candidates}
        assert winner.node_id == lexemes["shared"]
        assert by_id[lexemes["shared"]].convergence_count == 2
        assert by_id[lexemes["shared"]].semantic_support > by_id[
            lexemes["solo"]
        ].semantic_support
    finally:
        mind.close()


def test_a_productive_dead_end_ends_short_speech_without_erasing_its_word(
    tmp_path: Path,
) -> None:
    mind, _, _ = _word_competition_mind(tmp_path / "dead-end-speech.sqlite")
    try:
        planner = NoContextPlanner(mind)
        plan = planner.tick()

        speech = planner.compose_speech(
            plan, minimum_words=1, maximum_words=3
        )

        assert len(speech.lexical_node_ids) == 1
        assert len(speech.steps) == 1
        assert speech.stopped is True
    finally:
        mind.close()


def test_changing_only_recurrent_semantics_changes_the_next_word(
    tmp_path: Path,
) -> None:
    mind, focus, lexemes = _word_competition_mind(
        tmp_path / "counterfactual.sqlite"
    )
    try:
        neutral = mind.embedder.embed("same lexical comparison geometry")
        for node_id in (lexemes["alpha"], lexemes["beta"]):
            mind.store.update_concept_embedding(node_id, neutral)

        def set_activation(a: float, b: float):
            for node_id, value in (("semantic:a", a), ("semantic:b", b)):
                state = mind.store.get_node_dynamics(node_id)
                assert state is not None
                mind.store.put_node_dynamics(
                    replace(
                        state,
                        activation=value,
                        momentum=0.0,
                        last_selected_pulse=None,
                    )
                )
            return mind.recurrent.snapshot(pulse=mind.pulse)

        alpha_state = set_activation(1.0, 0.0)
        alpha, _ = compete_lexical_step(
            mind,
            alpha_state,
            focus,
            cursor_node_id="lexeme:cursor",
            step_index=2,
            minimum_words=0,
        )
        beta_state = set_activation(0.0, 1.0)
        beta, _ = compete_lexical_step(
            mind,
            beta_state,
            focus,
            cursor_node_id="lexeme:cursor",
            step_index=2,
            minimum_words=0,
        )

        assert alpha.node_id == lexemes["alpha"]
        assert beta.node_id == lexemes["beta"]
        assert alpha.node_id != beta.node_id
    finally:
        mind.close()


def test_receipted_outcome_relaxes_only_desire_on_causal_path(tmp_path: Path) -> None:
    with _mind(tmp_path / "outcome.sqlite") as mind:
        planner = NoContextPlanner(mind)
        plan = planner.hear("desire geometry zero")
        assert plan.focus.selected is not None
        selected_id = plan.focus.selected.terminal_node_id
        other_id = "drive:b" if selected_id == "drive:a" else "drive:a"
        selected_before = mind.store.get_node_dynamics(selected_id)
        other_before = mind.store.get_node_dynamics(other_id)
        cycle = planner.begin_speech_cycle(plan, "numeric state became language")
        mind.record_cycle_return(
            cycle.cycle_id,
            "verified helpful response",
            input_trunk=InputTrunk.HEAR,
            status="accepted",
            stability_delta=0.8,
            verified=True,
            terminal=True,
            record_id="receipt:accepted",
        )
        selected_after = mind.store.get_node_dynamics(selected_id)
        other_after = mind.store.get_node_dynamics(other_id)

        assert selected_before is not None and selected_after is not None
        assert other_before is not None and other_after is not None
        assert selected_after.pressure < selected_before.pressure
        assert other_after.pressure == pytest.approx(other_before.pressure)


def test_no_context_planner_never_calls_recall_or_serializes_input(tmp_path: Path) -> None:
    with _mind(tmp_path / "contextless.sqlite") as mind:
        def forbidden(*_args, **_kwargs):
            raise AssertionError("stored language is outside the no-context runtime")

        mind.recall = forbidden  # type: ignore[method-assign]
        mind.store.list_records = forbidden  # type: ignore[method-assign]
        mind.store.get_records = forbidden  # type: ignore[method-assign]
        mind.store.records_for_vault = forbidden  # type: ignore[method-assign]
        planner = NoContextPlanner(mind)
        text = "desire geometry zero"
        plan = planner.hear(text)
        assert plan.frame is not None
        assert plan.recurrent.state_sha256 == mind.recurrent.snapshot(
            pulse=mind.pulse
        ).state_sha256
        assert plan.frame.transcript_records_used == 0
        assert plan.frame.recalled_records_used == 0
        packet = write_native_state_packet(tmp_path / "state.packet", plan.frame)
        payload = packet.read_text(encoding="ascii")
        assert payload.startswith("HABITUS_RECURRENT_PACKET_V1\n32 ")
        assert text not in payload
        assert "drive:a" not in payload


def test_refractory_desire_yields_to_another_ready_drive(tmp_path: Path) -> None:
    with _mind(tmp_path / "refractory.sqlite") as mind:
        second = mind.store.get_node_dynamics("drive:b")
        assert second is not None
        mind.store.put_node_dynamics(replace(second, pressure=0.70))
        pulse, pulse_id = mind._next_pulse()
        initial = mind.recurrent.advance(pulse=pulse, pulse_id=pulse_id)
        mind.recurrent.mark_selected(("drive:a",), pulse=pulse)
        after_selection = mind.recurrent.snapshot(pulse=pulse)
        focus = resolve_output_focus(
            mind, after_selection, pulse_id=f"{pulse_id}:next", mark_active=False
        )

        assert initial.dominant_desire_id == "drive:a"
        assert focus.selected is not None
        assert focus.selected.terminal_node_id == "drive:b"


def test_batched_input_traversal_matches_individual_routes(tmp_path: Path) -> None:
    with _mind(tmp_path / "batched.sqlite") as mind:
        targets = {"drive:a": 0.9, "drive:b": 0.7}
        batched = {
            trace.target_node_id: trace
            for trace in mind.graph.traverse_many(
                pulse_id="batch:1",
                side=GraphSide.INPUT,
                targets=targets,
                required_input_trunk=InputTrunk.HEAR,
                mark_active=False,
            )
        }
        for target_id, score in targets.items():
            individual = mind.graph.traverse(
                pulse_id=f"individual:{target_id}",
                side=GraphSide.INPUT,
                target_id=target_id,
                endpoint_score=score,
                required_input_trunk=InputTrunk.HEAR,
                mark_active=False,
            )
            assert individual is not None
            assert batched[target_id].path_node_ids == individual.path_node_ids
            assert batched[target_id].path_edge_ids == individual.path_edge_ids
            assert batched[target_id].total_travel_time == individual.total_travel_time


def test_internal_tick_withholds_language_until_a_desire_is_ready(tmp_path: Path) -> None:
    with _mind(tmp_path / "tick.sqlite") as mind:
        for node_id in ("drive:a", "drive:b"):
            state = mind.store.get_node_dynamics(node_id)
            assert state is not None
            mind.store.put_node_dynamics(
                replace(
                    state,
                    pressure=0.0,
                    baseline_growth=0.0,
                    expression_threshold=0.9,
                )
            )
        plan = NoContextPlanner(mind).tick()
        assert plan.recurrent.should_express is False
        assert plan.frame is None


def test_ready_desire_can_initiate_speech_without_an_input_message(
    tmp_path: Path,
) -> None:
    with _mind(tmp_path / "autonomous.sqlite") as mind:
        planner = NoContextPlanner(mind)
        plan = planner.tick()

        assert plan.input_record is None
        assert plan.input_concept_ids == ()
        assert plan.recurrent.should_express is True
        assert plan.focus.selected is not None
        assert plan.focus.selected.terminal_node_id == "drive:a"
        assert plan.frame is not None
        cycle = planner.begin_speech_cycle(plan, "self-initiated speech")
        assert cycle.status == "open"
        selected = mind.store.get_node_dynamics("drive:a")
        assert selected is not None
        assert selected.last_selected_pulse == mind.pulse


def test_native_frame_is_fixed_numeric_state_not_message_history(tmp_path: Path) -> None:
    with _mind(tmp_path / "frame.sqlite") as mind:
        pulse, pulse_id = mind._next_pulse()
        state = mind.recurrent.advance(pulse=pulse, pulse_id=pulse_id)
        focus = resolve_output_focus(mind, state, pulse_id=pulse_id)
        frame = build_native_state_frame(
            mind, state, focus, input_sha256=None, maximum_rows=12
        )
        assert frame is not None
        assert 4 <= len(frame.rows) <= 12
        assert all(len(row) == 32 for row in frame.rows)
        assert frame.row_kinds[:4] == (
            "whole_graph",
            "recurrent_activation",
            "desire_field",
            "selected_output",
        )
        assert mind.graph.validate_invariants() == []


def _developmental_planner(mind: BaseAgenticMemoryRAG):
    def lexical_sensor(text: str):
        return tuple(
            mind.embedder.embed(f"heard-unit:{unit}")
            for unit in text.casefold().split()
        )

    language = DevelopmentalLanguage(mind, promotion_exposures=3)
    planner = NoContextPlanner(
        mind,
        lexical_sensor=lexical_sensor,
        language_learner=language,
    )
    return planner, language


def test_heard_units_begin_weak_then_become_productive_vocabulary(
    tmp_path: Path,
) -> None:
    mind, _, _ = _word_competition_mind(tmp_path / "online-language.sqlite")
    try:
        planner, _ = _developmental_planner(mind)
        first = planner.hear("azure lattice", source_id="caregiver-a")
        assert first.acquisition is not None
        assert first.acquisition.transition_edges_staged
        assert set(first.acquisition.transition_edges_dormant) == set(
            first.acquisition.transition_edge_ids
        )
        azure_id, lattice_id = first.acquisition.lexical_node_ids
        assert mind.store.get_concept(azure_id).kind == LEXEME_CANDIDATE_KIND
        assert mind.store.get_concept(azure_id).terms == ()
        assert mind.store.get_concept(azure_id).label == azure_id

        start_id = mind.store.list_concepts(kind=SPEECH_START_KIND)[0].concept_id
        _, immature_candidates = compete_lexical_step(
            mind,
            first.recurrent,
            first.focus,
            cursor_node_id=start_id,
            step_index=0,
            minimum_words=0,
        )
        assert azure_id not in {item.node_id for item in immature_candidates}

        planner.hear("azure lattice", source_id="caregiver-a")
        third = planner.hear("azure lattice", source_id="caregiver-a")
        assert third.acquisition is not None
        assert set(third.acquisition.nodes_promoted) == {azure_id, lattice_id}
        assert third.acquisition.transition_edges_dormant == ()
        assert set(third.acquisition.transition_edges_promoted) == set(
            third.acquisition.transition_edge_ids
        )
        assert mind.store.get_concept(azure_id).kind == "lexeme"
        assert mind.store.get_concept(lattice_id).kind == "lexeme"
        assert mind.store.vault_record_count(
            mind.store.get_concept(azure_id).vault_id
        ) == 3

        _, mature_candidates = compete_lexical_step(
            mind,
            third.recurrent,
            third.focus,
            cursor_node_id=start_id,
            step_index=0,
            minimum_words=0,
        )
        assert azure_id in {item.node_id for item in mature_candidates}
        pair = mind.store.find_edge(GraphSide.OUTPUT, azure_id, lattice_id)
        stop_id = mind.store.list_concepts(kind=SPEECH_STOP_KIND)[0].concept_id
        assert pair is not None
        assert mind.store.find_edge(GraphSide.OUTPUT, lattice_id, stop_id) is not None
        before = pair.log_strength
        planner.hear("azure lattice", source_id="caregiver-a")
        after = mind.store.find_edge(GraphSide.OUTPUT, azure_id, lattice_id)
        assert after is not None and after.log_strength > before
        assert mind.graph.validate_invariants() == []
    finally:
        mind.close()


def test_lexical_candidate_exposure_and_promotion_survive_restart(
    tmp_path: Path,
) -> None:
    database = tmp_path / "persistent-language.sqlite"
    mind, _, _ = _word_competition_mind(database)
    planner, language = _developmental_planner(mind)
    source_node = language.ensure_source("caregiver-a")
    language.learn_source_preference(source_node, 0.8, confidence=0.9)
    first = planner.hear("cobalt", source_id="caregiver-a")
    planner.hear("cobalt", source_id="caregiver-a")
    assert first.acquisition is not None
    cobalt_id = first.acquisition.lexical_node_ids[0]
    assert mind.store.get_concept(cobalt_id).kind == LEXEME_CANDIDATE_KIND
    mind.close()

    reopened, _, _ = _word_competition_mind(database)
    try:
        planner, language = _developmental_planner(reopened)
        third = planner.hear("cobalt", source_id="caregiver-a")
        assert third.acquisition is not None
        assert third.acquisition.nodes_promoted == (cobalt_id,)
        assert reopened.store.get_concept(cobalt_id).kind == "lexeme"
        source_node = language.ensure_source("caregiver-a")
        assert reopened.store.get_concept(source_node).terms == ()
        source_state = reopened.store.get_social_source_state(source_node)
        assert source_state is not None
        assert source_state.preference_mean == pytest.approx(0.8)
        assert source_state.preference_weight == pytest.approx(0.9)
    finally:
        reopened.close()


def test_one_heard_adjacency_between_known_lexemes_stays_dormant(
    tmp_path: Path,
) -> None:
    mind, _, _ = _word_competition_mind(tmp_path / "transition-maturation.sqlite")
    try:
        rows = {
            name: as_tuple(mind.embedder.embed(f"online-known:{name}"))
            for name in ("alpha", "beta")
        }
        ids = {name: lexical_geometry_id(row) for name, row in rows.items()}
        for name, node_id in ids.items():
            mind.store.add_concept(
                ConceptNode(
                    concept_id=node_id,
                    label=node_id,
                    kind="lexeme",
                    embedding=rows[name],
                    terms=(),
                    vault_id=f"lexical-geometry:{node_id}",
                    created_pulse=mind.pulse,
                    last_active_pulse=mind.pulse,
                )
            )

        def known_sensor(text: str):
            return tuple(rows[unit] for unit in text.split())

        language = DevelopmentalLanguage(mind, promotion_exposures=3)
        planner = NoContextPlanner(
            mind,
            lexical_sensor=known_sensor,
            language_learner=language,
        )
        first = planner.hear("alpha beta", source_id="caregiver-a")
        assert first.acquisition is not None
        pair = mind.store.find_edge(
            GraphSide.OUTPUT, ids["alpha"], ids["beta"]
        )
        assert pair is not None and pair.archived is True
        assert pair.edge_id in first.acquisition.transition_edges_dormant
        assert pair not in mind.store.list_edges(GraphSide.OUTPUT)

        second = planner.hear("alpha beta", source_id="caregiver-a")
        assert second.acquisition is not None
        assert pair.edge_id in second.acquisition.transition_edges_dormant
        third = planner.hear("alpha beta", source_id="caregiver-a")
        assert third.acquisition is not None
        matured = mind.store.get_edge(pair.edge_id)
        assert matured is not None and matured.archived is False
        assert pair.edge_id in third.acquisition.transition_edges_promoted
        assert pair.edge_id not in third.acquisition.transition_edges_dormant
    finally:
        mind.close()


def test_restart_consolidation_promotes_only_patterns_with_enough_evidence(
    tmp_path: Path,
) -> None:
    mind, _, _ = _word_competition_mind(tmp_path / "restart-consolidation.sqlite")
    try:
        rows = {
            name: as_tuple(mind.embedder.embed(f"restart-word:{name}"))
            for name in ("rare", "repeated", "tail")
        }

        def sensor(text: str):
            return tuple(rows[unit] for unit in text.split())

        original = DevelopmentalLanguage(
            mind,
            promotion_exposures=3,
            transition_promotion_exposures=3,
        )
        planner = NoContextPlanner(
            mind,
            lexical_sensor=sensor,
            language_learner=original,
        )
        first = planner.hear("rare repeated", source_id="caregiver-a")
        planner.hear("rare repeated", source_id="caregiver-a")
        planner.hear("tail", source_id="caregiver-a")
        rare_id, repeated_id = first.acquisition.lexical_node_ids
        tail_id = lexical_geometry_id(rows["tail"])
        pair = mind.store.find_edge(GraphSide.OUTPUT, rare_id, repeated_id)
        assert pair is not None and pair.archived is True
        assert mind.store.get_concept(rare_id).kind == LEXEME_CANDIDATE_KIND

        lower_threshold = DevelopmentalLanguage(
            mind,
            promotion_exposures=2,
            transition_promotion_exposures=2,
        )
        consolidated = lower_threshold.consolidate_ready_patterns()

        assert {rare_id, repeated_id}.issubset(
            consolidated["nodes_promoted"]
        )
        assert tail_id not in consolidated["nodes_promoted"]
        assert pair.edge_id in consolidated["transition_edges_promoted"]
        assert mind.store.get_edge(pair.edge_id).archived is False
        assert mind.store.get_concept(tail_id).kind == LEXEME_CANDIDATE_KIND
    finally:
        mind.close()


def test_separate_heard_sequences_create_an_unpresented_recombinant_route(
    tmp_path: Path,
) -> None:
    mind, _, _ = _word_competition_mind(tmp_path / "recombination.sqlite")
    try:
        planner, _ = _developmental_planner(mind)
        first_ids = None
        second_ids = None
        for _ in range(3):
            first = planner.hear("amber bridge quiet", source_id="caregiver-a")
            second = planner.hear("cobalt bridge bright", source_id="caregiver-a")
            first_ids = first.acquisition.lexical_node_ids
            second_ids = second.acquisition.lexical_node_ids
        assert first_ids is not None and second_ids is not None
        amber, shared_bridge, quiet = first_ids
        cobalt, second_bridge, bright = second_ids
        assert shared_bridge == second_bridge

        start_id = mind.store.list_concepts(kind=SPEECH_START_KIND)[0].concept_id
        stop_id = mind.store.list_concepts(kind=SPEECH_STOP_KIND)[0].concept_id
        recombinant = (start_id, amber, shared_bridge, bright, stop_id)
        assert all(
            mind.store.find_edge(GraphSide.OUTPUT, source, target) is not None
            for source, target in zip(recombinant, recombinant[1:])
        )
        assert recombinant[1:-1] not in {first_ids, second_ids}
        assert mind.store.find_edge(GraphSide.OUTPUT, cobalt, shared_bridge) is not None
        assert mind.store.find_edge(GraphSide.OUTPUT, shared_bridge, quiet) is not None
    finally:
        mind.close()


def _sensation(vector, lexical_rows=()):
    return LexicalSensation(
        input_sha256="numeric-test-sensation",
        utterance_embedding=as_tuple(vector),
        lexical_rows=tuple(as_tuple(row) for row in lexical_rows),
    )


def test_social_return_depends_on_content_and_learned_source_preference(
    tmp_path: Path,
) -> None:
    mind, _, _ = _word_competition_mind(tmp_path / "social-return.sqlite")
    try:
        _, language = _developmental_planner(mind)
        welcomed = language.ensure_source("source-welcomed")
        resisted = language.ensure_source("source-resisted")
        neutral = language.ensure_source("source-neutral")
        for _ in range(4):
            language.learn_source_preference(welcomed, 1.0, confidence=1.0)
            language.learn_source_preference(resisted, -1.0, confidence=1.0)

        semantic = mind.store.get_concept("semantic:a").embedding
        opposite = tuple(-value for value in semantic)
        welcomed_cycle = ExperienceCycle(
            cycle_id="cycle:welcomed",
            output_record_id="unused:welcomed",
            output_pulse_id="pulse:welcomed",
            output_trunk=OutputTrunk.SPEAK,
            credited_edge_ids=(),
            opened_pulse=mind.pulse,
            metadata={
                "source_node_id": welcomed,
                "semantic_node_ids": ["semantic:a"],
                "lexical_node_ids": [],
            },
        )
        resisted_cycle = replace(
            welcomed_cycle,
            cycle_id="cycle:resisted",
            output_record_id="unused:resisted",
            metadata={
                **welcomed_cycle.metadata,
                "source_node_id": resisted,
            },
        )
        neutral_cycle = replace(
            welcomed_cycle,
            cycle_id="cycle:neutral",
            output_record_id="unused:neutral",
            metadata={
                **welcomed_cycle.metadata,
                "source_node_id": neutral,
            },
        )

        def forbidden_record_read(*_args, **_kwargs):
            raise AssertionError("social return cannot inspect stored message text")

        mind.store.get_record = forbidden_record_read  # type: ignore[method-assign]
        mind.store.list_records = forbidden_record_read  # type: ignore[method-assign]
        mind.store.records_for_vault = forbidden_record_read  # type: ignore[method-assign]
        related = language.assess_social_return(
            welcomed_cycle, _sensation(semantic), source_id="source-welcomed"
        )
        unrelated = language.assess_social_return(
            welcomed_cycle, _sensation(opposite), source_id="source-welcomed"
        )
        same_content_resisted = language.assess_social_return(
            resisted_cycle, _sensation(semantic), source_id="source-resisted"
        )
        unrelated_neutral = language.assess_social_return(
            neutral_cycle, _sensation(opposite), source_id="source-neutral"
        )

        assert related is not None
        assert unrelated is not None
        assert same_content_resisted is not None
        assert unrelated_neutral is not None
        assert related.semantic_alignment == pytest.approx(1.0)
        assert unrelated.semantic_alignment == pytest.approx(0.0)
        assert related.stability_delta > unrelated.stability_delta
        assert related.stability_delta > same_content_resisted.stability_delta
        assert unrelated_neutral.stability_delta == pytest.approx(0.0)
        assert language.assess_social_return(
            welcomed_cycle, _sensation(semantic), source_id="someone-else"
        ) is None
    finally:
        mind.close()


def test_social_reply_closes_only_the_cycle_for_its_source(tmp_path: Path) -> None:
    mind, _, _ = _word_competition_mind(tmp_path / "social-cycle.sqlite")
    try:
        planner, language = _developmental_planner(mind)
        plan = planner.hear("steady signal", source_id="alice")
        cycle = planner.begin_speech_cycle(plan, "numeric graph speech")
        semantic = mind.store.get_concept(plan.input_concept_ids[0]).embedding
        reply = _sensation(
            semantic,
            (mind.embedder.embed("heard-unit:steady"),),
        )

        assert language.close_social_cycles(reply, source_id="bob") == ()
        assert mind.experience_cycle(cycle.cycle_id).status == "open"
        closed = language.close_social_cycles(reply, source_id="alice")
        assert len(closed) == 1
        assert closed[0].cycle_id == cycle.cycle_id
        assert mind.experience_cycle(cycle.cycle_id).status == "closed"
        source_state = mind.store.get_social_source_state(plan.source_node_id)
        assert source_state is not None
        assert source_state.observation_count == 1
    finally:
        mind.close()


def test_social_reply_and_current_message_settle_in_one_self_pulse(
    tmp_path: Path,
) -> None:
    mind, _, _ = _word_competition_mind(tmp_path / "queued-social-cycle.sqlite")
    try:
        planner, language = _developmental_planner(mind)
        plan = planner.hear("steady signal", source_id="alice")
        output = planner.begin_speech_cycle(plan, "numeric graph speech")
        pulse_before_reply = mind.pulse
        reply_text = "steady reply"
        sensation = planner.sense(reply_text)

        queued = language.queue_social_cycles(
            sensation,
            source_id="alice",
            pulse_kernel=planner.pulse_kernel,
        )
        assert len(queued) == 1
        assert mind.experience_cycle(output.cycle_id).status == "open"

        reply_plan = planner.hear(
            reply_text,
            source_id="alice",
            sensation=sensation,
            additional_input_item_ids=(queued[0][1].item_id,),
        )
        assert mind.pulse == pulse_before_reply + 1
        assert reply_plan.self_pulse is not None
        assert [
            item.kind for item in reply_plan.self_pulse.sensory.receipts
        ] == ["cycle_return", "input"]
        assert all(
            item.pulse == reply_plan.self_pulse.self_state.pulse
            for item in reply_plan.self_pulse.sensory.receipts
        )
        assert mind.experience_cycle(output.cycle_id).status == "closed"
    finally:
        mind.close()
