import pytest
from habitus_ai import (
    GraphSide,
    HabitusAI,
    InputTrunk,
    OutputTrunk,
    RecordType,
    SelfPulseKernel,
)
from habitus_ai.graph import OUTPUT_NODE_IDS
from habitus_ai.tools import (
    ToolDefinition,
    ToolReceipt,
    ToolRegistry,
    BUILTIN_OPERATIONAL_TOOLS,
)

def test_tool_registry_binding(tmp_path):
    mind = HabitusAI(tmp_path / "test_tools_mind.sqlite")
    registry = ToolRegistry(mind)

    # Register built-in operational tools
    for tool in BUILTIN_OPERATIONAL_TOOLS:
        registry.register_tool(tool)

    # Verify tool concepts exist in store under correct output trunks
    read_concept = mind.store.get_concept("tool:read_file")
    assert read_concept is not None

    read_edge = mind.store.find_edge(GraphSide.OUTPUT, OUTPUT_NODE_IDS[OutputTrunk.LOOK], "tool:read_file")
    assert read_edge is not None

    write_edge = mind.store.find_edge(GraphSide.OUTPUT, OUTPUT_NODE_IDS[OutputTrunk.DO], "tool:write_file")
    assert write_edge is not None

    # Execute tool and verify receipt
    target_file = tmp_path / "sample.txt"
    write_receipt = registry.execute("tool:write_file", {"filepath": str(target_file), "content": "hello habitus"})
    assert write_receipt.verified is True
    assert write_receipt.output["bytes_written"] > 0
    assert write_receipt.cycle_id is not None
    assert write_receipt.output_record_id is not None
    assert write_receipt.return_record_id == write_receipt.receipt_id
    cycle = mind.experience_cycle(write_receipt.cycle_id)
    assert cycle.status == "closed"
    assert cycle.output_record_id == write_receipt.output_record_id
    assert cycle.terminal_return_record_id == write_receipt.return_record_id
    assert mind.store.get_record(cycle.output_record_id).record_type == RecordType.TOOL_CALL
    assert mind.store.get_record(write_receipt.return_record_id).record_type == RecordType.TOOL_RESULT
    assert mind.store.get_record(cycle.output_record_id).metadata["membrane_words"] is False
    assert mind.store.get_record(write_receipt.return_record_id).metadata["membrane_words"] is False
    returned = mind.store.returns_for_experience_cycle(cycle.cycle_id)
    assert [(item.status, item.terminal, item.verified) for item in returned] == [
        ("success", True, True)
    ]
    link = mind.store.connection.execute(
        """SELECT relation FROM record_links
           WHERE source_record_id = ? AND target_record_id = ?""",
        (write_receipt.return_record_id, write_receipt.output_record_id),
    ).fetchone()
    assert link["relation"] == "returns_to"
    state = mind.experience_state(cycle.cycle_id)
    assert state.preference_mean == pytest.approx(0.20)
    assert any(
        projection.side == GraphSide.OUTPUT
        for projection in mind.experience_projections(cycle.cycle_id)
    )
    assert any(
        projection.side == GraphSide.INPUT
        for projection in mind.experience_projections(cycle.cycle_id)
    )

    read_receipt = registry.execute("tool:read_file", {"filepath": str(target_file)})
    assert read_receipt.verified is True
    assert read_receipt.output["content"] == "hello habitus"

    error_receipt = registry.execute(
        "tool:read_file",
        {"filepath": str(tmp_path / "missing.txt")},
    )
    assert error_receipt.status == "error"
    assert error_receipt.verified is True
    assert mind.experience_cycle(error_receipt.cycle_id).status == "closed"
    assert mind.experience_state(error_receipt.cycle_id).preference_mean == pytest.approx(-0.20)

    mind.close()


def test_kernel_backed_tool_requires_selected_affordance_and_settles_return(
    tmp_path,
):
    mind = HabitusAI(tmp_path / "kernel_tools_mind.sqlite")
    kernel = SelfPulseKernel(mind)
    calls = []
    registry = ToolRegistry(mind, pulse_kernel=kernel)
    registry.register_tool(
        ToolDefinition(
            tool_id="ability:touch",
            trunk=OutputTrunk.DO,
            label="Touch",
            description="Mutate one controlled environment state.",
            terms=("touch",),
            parameters={"value": {"type": "integer"}},
            handler=lambda arguments: calls.append(arguments["value"]) or {"ok": True},
        )
    )
    kernel.enqueue_input(
        "touch opportunity",
        lane=InputTrunk.NOTICE,
        concept_ids=("ability:touch",),
        token_cost=1,
    )
    proposal = kernel.advance_cycle()
    assert proposal is not None
    affordance = next(
        item
        for item in proposal.selected_outputs
        if item.node_id == "ability:touch"
    )

    with pytest.raises(ValueError, match="selected self-pulse affordance"):
        registry.execute("ability:touch", {"value": 1})
    assert calls == []

    receipt = registry.execute(
        "ability:touch",
        {"value": 7},
        affordance=affordance,
    )
    assert calls == [7]
    assert receipt.verified is True
    assert receipt.status == "success"
    assert mind.pulse == 2
    assert mind.experience_cycle(receipt.cycle_id).status == "closed"
    assert mind.store.get_record(receipt.return_record_id).record_type == (
        RecordType.TOOL_RESULT
    )
    payload = kernel.latest_state_payload()
    assert payload["inputs"][0]["kind"] == "cycle_return"
    assert payload["inputs"][0]["lane"] == "NOTICE"
    authorization = mind.store.connection.execute(
        """SELECT status FROM self_output_authorizations
           WHERE authorization_id = ?""",
        (affordance.authorization_id,),
    ).fetchone()
    assert authorization["status"] == "consumed"
    mind.close()


def test_kernel_backed_tool_can_delegate_return_to_full_runtime_advancer(tmp_path):
    mind = HabitusAI(tmp_path / "delegated_return.sqlite")
    kernel = SelfPulseKernel(mind)
    advanced = []

    def advance(item_ids):
        advanced.append(tuple(item_ids))
        return kernel.advance_cycle(item_ids=item_ids)

    registry = ToolRegistry(
        mind,
        pulse_kernel=kernel,
        pulse_advancer=advance,
    )
    registry.register_tool(
        ToolDefinition(
            tool_id="ability:delegated-touch",
            trunk=OutputTrunk.DO,
            label="Delegated touch",
            description="Exercise the full-runtime return callback.",
            terms=(),
            parameters={},
            handler=lambda _arguments: {"ok": True},
            opaque=True,
        )
    )
    kernel.enqueue_input(
        "opaque opportunity",
        lane=InputTrunk.NOTICE,
        concept_ids=("ability:delegated-touch",),
    )
    pulse = kernel.advance_cycle()
    affordance = next(
        item
        for item in pulse.selected_outputs
        if item.node_id == "ability:delegated-touch"
    )
    receipt = registry.execute(
        "ability:delegated-touch",
        {},
        affordance=affordance,
    )

    assert receipt.verified is True
    assert advanced == [(f"tool-return:{receipt.receipt_id}",)]
    assert kernel.pending() == ()
    mind.close()
