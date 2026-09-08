from __future__ import annotations

import json

from habitus_ai import BaseAgenticMemoryRAG, gestate
from habitus_ai.functional_agent import FunctionalAgent, main
from habitus_ai.models import OllamaChatModel
from habitus_ai.types import RecordType


class ScriptedModel:
    def __init__(self, *responses: str) -> None:
        self.responses = list(responses)
        self.calls: list[list[dict[str, str]]] = []

    def generate(self, messages):
        self.calls.append([dict(message) for message in messages])
        return self.responses.pop(0)


def _gestated_mind(database):
    mind = BaseAgenticMemoryRAG(database)
    gestate(
        mind,
        human_name="Nemo",
        agent_name="Moss",
        taste_schema="builder",
        model_name="scripted",
    )
    return mind


def test_conversation_and_explicit_memory_survive_restart(tmp_path) -> None:
    database = tmp_path / "mind.sqlite"
    model = ScriptedModel(
        "Hello Nemo. I am here.",
        "We are still in the same conversation.",
    )
    mind = _gestated_mind(database)
    agent = FunctionalAgent(mind, model, workspace=tmp_path)

    first = agent.handle("Hello, are you awake?")
    second = agent.handle("What did I just ask you?")
    remembered = agent.handle("remember that my launch color is ultraviolet")

    assert first.kind == "conversation"
    assert second.kind == "conversation"
    assert "Hello, are you awake?" in "\n".join(
        message["content"] for message in model.calls[1]
    )
    assert remembered.kind == "memory_write"
    assert "ultraviolet" in remembered.response
    assert agent.state()["durable_facts"] == ["my launch color is ultraviolet"]
    mind.close()

    resumed_model = ScriptedModel("Your launch color is ultraviolet.")
    with BaseAgenticMemoryRAG(database) as resumed_mind:
        resumed = FunctionalAgent(resumed_mind, resumed_model, workspace=tmp_path)
        answer = resumed.handle("What is my launch color?")
        supplied = "\n".join(
            message["content"] for message in resumed_model.calls[0]
        )
        assert answer.response == "Your launch color is ultraviolet."
        assert "my launch color is ultraviolet" in supplied
        assert "Treat memories as quiet background" in supplied
        assert "follow requested brevity" in supplied
        assert resumed.state()["graph_invariant_errors"] == []


def test_workspace_open_and_run_have_verified_receipts(tmp_path) -> None:
    database = tmp_path / "mind.sqlite"
    note = tmp_path / "note.txt"
    note.write_text("the orchard signal is violet\n", encoding="utf-8")
    script = tmp_path / "probe.py"
    script.write_text("print('FUNCTIONAL_RUN_OK')\n", encoding="utf-8")
    folder = tmp_path / "folder"
    folder.mkdir()
    (folder / "nested.txt").write_text("nested\n", encoding="utf-8")
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("outside", encoding="utf-8")

    mind = _gestated_mind(database)
    agent = FunctionalAgent(mind, ScriptedModel(), workspace=tmp_path)

    listed = agent.handle("/open folder")
    opened = agent.handle("/open note.txt")
    ran = agent.handle("/run probe.py")
    blocked = agent.handle("/open ../outside.txt")

    assert listed.tool_receipt is not None
    assert listed.tool_receipt.verified is True
    assert listed.tool_receipt.output["kind"] == "directory"
    assert listed.tool_receipt.output["entries"][0]["name"] == "nested.txt"
    assert opened.kind == "tool"
    assert opened.tool_receipt is not None
    assert opened.tool_receipt.verified is True
    assert opened.tool_receipt.status == "success"
    assert "the orchard signal is violet" in opened.response
    assert ran.tool_receipt is not None
    assert ran.tool_receipt.verified is True
    assert ran.tool_receipt.status == "success"
    assert "FUNCTIONAL_RUN_OK" in ran.response
    assert blocked.tool_receipt is not None
    assert blocked.tool_receipt.verified is True
    assert blocked.tool_receipt.status == "error"
    assert "outside the authorized workspace" in blocked.response

    tool_calls = [
        record
        for record in mind.store.list_records()
        if record.record_type == RecordType.TOOL_CALL
    ]
    tool_returns = [
        record
        for record in mind.store.list_records()
        if record.record_type == RecordType.TOOL_RESULT
    ]
    assert len(tool_calls) == 4
    assert len(tool_returns) == 4
    assert all(mind.experience_cycle(record.metadata["experience_id"]) for record in tool_calls)
    assert agent.state()["verified_tool_returns"] == 4
    mind.close()


def test_state_and_help_work_without_calling_the_language_model(tmp_path) -> None:
    mind = _gestated_mind(tmp_path / "mind.sqlite")
    model = ScriptedModel()
    agent = FunctionalAgent(mind, model, workspace=tmp_path)

    state = agent.handle("/state")
    help_turn = agent.handle("/help")

    assert state.kind == "state"
    assert json.loads(state.response)["workspace"] == str(tmp_path)
    assert help_turn.kind == "help"
    assert "remember that" in help_turn.response
    assert model.calls == []
    mind.close()


def test_one_shot_state_has_a_scriptable_json_surface(tmp_path, capsys) -> None:
    result = main(
        [
            "--database",
            str(tmp_path / "mind.sqlite"),
            "--workspace",
            str(tmp_path),
            "--once",
            "/state",
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert result == 0
    assert payload["kind"] == "state"
    assert json.loads(payload["response"])["workspace"] == str(tmp_path)


def test_ollama_cpu_mode_is_sent_to_local_model(monkeypatch) -> None:
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return json.dumps({"message": {"content": "awake"}}).encode()

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    model = OllamaChatModel("local-test", num_gpu=0, timeout_seconds=12)
    assert model.generate([{"role": "user", "content": "hello"}]) == "awake"
    assert captured["payload"]["options"]["num_gpu"] == 0
    assert captured["payload"]["options"]["num_predict"] == 512
    assert captured["payload"]["think"] is False
    assert captured["timeout"] == 12
