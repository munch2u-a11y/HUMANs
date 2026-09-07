from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys

import pytest


EXPERIMENT = Path(__file__).resolve().parents[1] / "experiments" / "graph_native_live"
for import_root in (Path(__file__).resolve().parents[1] / "src", EXPERIMENT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

MODULE_PATH = EXPERIMENT / "desire_nursery.py"
SPEC = importlib.util.spec_from_file_location("graph_desire_nursery", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
DESIRES = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = DESIRES
SPEC.loader.exec_module(DESIRES)

from habitus_ai import BaseAgenticMemoryRAG, DeterministicHashEmbedder  # noqa: E402
from habitus_ai.open_weight import NoContextPlanner, resolve_output_focus  # noqa: E402
from habitus_ai.types import GraphSide, InputTrunk, OutputTrunk  # noqa: E402


def test_nursery_grows_shared_word_competition_without_fixed_utterances(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with BaseAgenticMemoryRAG(
        tmp_path / "desires.sqlite",
        embedder=DeterministicHashEmbedder(32),
    ) as mind:
        topic_names = sorted(
            {
                topic
                for seed in DESIRES.DRIVE_SEEDS
                for topic in seed.source_topics
            }
            | {
                topic
                for exposure in DESIRES.WORD_EXPOSURES
                for topic in exposure.source_topics
            }
            | {"language"}
        )
        topic_nodes = {}
        for index, topic in enumerate(topic_names):
            node_id = f"topic:{index:02d}"
            mind.graph.add_concept(
                node_id,
                node_id,
                terms=(),
                embedding=mind.embedder.embed(f"topic geometry {topic}"),
                input_trunks=(InputTrunk.HEAR,),
                output_trunks=(OutputTrunk.SPEAK,),
                kind="crown",
            )
            topic_nodes[topic] = node_id
        monkeypatch.setattr(DESIRES, "_topic_index", lambda _mind: topic_nodes)

        fragment_vectors = {}

        def fake_mass_embed(_model, _codec, fragments):
            result = []
            for index, fragment in enumerate(fragments):
                vector = tuple(
                    mind.embedder.embed(f"lexical-{fragment.encode().hex()}")
                )
                fragment_vectors[fragment] = vector
                result.append(((index + 100,), vector))
            return result

        monkeypatch.setattr(DESIRES.accelerated_gestation, "mass_embed", fake_mass_embed)
        before_records = len(mind.store.list_records())
        manifest = DESIRES.install_desire_nursery(
            mind, Path("unused.gguf"), Path("unused-codec")
        )

        assert manifest["ready"] is True
        assert manifest["introduced_graph_invariants"] == []
        assert manifest["primary_drives"] == 6
        assert manifest["composite_drives"] == 3
        assert manifest["runtime_contract"]["fixed_utterances"] == 0
        assert manifest["runtime_contract"]["lexical_decision_unit"] == (
            "one_competing_word_per_recurrent_pulse"
        )
        assert manifest["maximum_transition_outdegree"] >= 4
        assert manifest["convergent_lexemes"] >= 1
        assert len(mind.store.list_node_dynamics()) == 9
        inquiry_state = mind.store.get_node_dynamics(manifest["nodes"]["inquiry"])
        assert inquiry_state is not None
        assert inquiry_state.pressure == pytest.approx(0.64)
        assert inquiry_state.activation == pytest.approx(0.0)
        assert inquiry_state.last_pulse == mind.pulse
        assert len(mind.store.list_records()) == before_records + len(
            DESIRES.WORD_EXPOSURES
        ) + len(DESIRES.TRANSITION_EXPOSURES)
        assert mind.graph.validate_invariants() == []

        for node_id in manifest["nodes"].values():
            concept = mind.store.get_concept(node_id)
            assert concept is not None
            assert concept.label == node_id
            assert concept.terms == ()
            assert concept.kind in {"drive", "desire_composite"}
        assert all(
            node.label == node.concept_id and node.terms == ()
            for node in mind.store.list_concepts(kind="lexeme")
        )

        output_edges = mind.store.list_edges(GraphSide.OUTPUT)
        drive_ids = set(manifest["nodes"].values())
        assert not any(
            edge.source_id in drive_ids
            and mind.store.get_concept(edge.target_id).kind == "lexeme"
            for edge in output_edges
        )
        start_id = manifest["boundaries"]["start"]
        assert all(
            mind.store.find_edge(GraphSide.OUTPUT, node_id, start_id) is not None
            for node_id in drive_ids
        )

        developmental_records = [
            record
            for record in mind.store.list_records()
            if record.metadata.get("developmental_lexical_unit")
        ]
        assert developmental_records
        assert all(record.metadata["fixed_utterance"] is False for record in developmental_records)
        assert all(len(record.text.replace(".", " .").split()) <= 2 for record in developmental_records)

        planner = NoContextPlanner(mind)

        def forbidden_runtime_record_read(*_args, **_kwargs):
            raise AssertionError("word competition cannot read developmental text")

        with monkeypatch.context() as runtime_guard:
            runtime_guard.setattr(mind.store, "list_records", forbidden_runtime_record_read)
            runtime_guard.setattr(mind.store, "get_records", forbidden_runtime_record_read)
            runtime_guard.setattr(mind.store, "get_record", forbidden_runtime_record_read)
            runtime_guard.setattr(
                mind.store, "records_for_vault", forbidden_runtime_record_read
            )
            plan = planner.tick()
            speech = planner.compose_speech(plan, minimum_words=3, maximum_words=10)
        inquiry_id = manifest["nodes"]["inquiry"]
        assert plan.recurrent.dominant_desire_id == inquiry_id
        assert plan.focus.selected is not None
        assert plan.focus.selected.terminal_node_id in drive_ids
        assert len(speech.lexical_node_ids) >= 3
        assert speech.stopped is True
        assert speech.steps[0].state_before_sha256 != speech.steps[0].state_after_sha256
        assert any(len(step.candidates) > 1 for step in speech.steps)
        assert all(
            mind.store.find_edge(
                GraphSide.OUTPUT, source_id, target_id
            ) is not None
            for source_id, target_id in zip(
                speech.path_node_ids, speech.path_node_ids[1:]
            )
        )

        vocabulary = {
            node_id: surface
            for surface, node_id in manifest["audit_vocabulary"].items()
        }
        generated = tuple(vocabulary[node_id] for node_id in speech.lexical_node_ids)
        taught_pairs = {
            tuple(item for item in (exposure.source, exposure.target) if item is not None)
            for exposure in DESIRES.TRANSITION_EXPOSURES
        }
        assert len(generated) > 2
        assert generated not in taught_pairs
        cycle = planner.begin_speech_cycle(
            plan, "decoded geometry output", speech=speech
        )
        assert set(speech.path_edge_ids).issubset(cycle.credited_edge_ids)
        assert set(speech.causal_edge_ids) == set(cycle.credited_edge_ids)

        composite_id = manifest["nodes"]["responsible_action"]
        for state in mind.store.list_node_dynamics():
            if mind.store.get_concept(state.node_id).kind not in {
                "drive",
                "desire_composite",
            }:
                continue
            mind.store.put_node_dynamics(
                replace(
                    state,
                    activation=0.0,
                    pressure=0.95 if state.node_id == composite_id else 0.02,
                    baseline_growth=0.0,
                    last_selected_pulse=None,
                )
            )
        composite_state = mind.recurrent.snapshot(pulse=mind.pulse)
        composite_focus = resolve_output_focus(
            mind,
            composite_state,
            pulse_id="composite-probe",
            mark_active=False,
        )
        assert composite_state.dominant_desire_id == composite_id
        assert composite_focus.selected is not None
        assert composite_focus.selected.terminal_node_id == composite_id

        records_after_first_install = len(mind.store.list_records())
        repeated = DESIRES.install_desire_nursery(
            mind, Path("unused.gguf"), Path("unused-codec")
        )
        assert repeated == manifest
        assert len(mind.store.list_records()) == records_after_first_install
        stored = json.loads(mind.store.get_metadata(DESIRES.MANIFEST_KEY))
        assert stored["runtime_contract"]["transcript_window"] == 0
