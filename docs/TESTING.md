# Testing and verification

The standard checks are CPU-only. They do not contact Ollama, download model
weights, start a nursery, or write persistent state outside disposable test
directories.

## One command before shipping

```bash
make verify
```

That target runs these checks in order:

| Command | What it checks | External model |
| --- | --- | --- |
| `make portability` | no developer home paths or machine-specific absolute code defaults | no |
| `make docs` | every repository-relative Markdown link resolves | no |
| `make playground` | disposable integrated conversation, memory, recall, read, run, receipts, and state | no |
| `make test` | complete deterministic behavioral and structural pytest suite | no |

Run a focused test during development with the environment interpreter:

```bash
.venv/bin/python -m pytest tests/test_integrated_agent.py -q
.venv/bin/python -m pytest tests/test_runtime_paths.py -q
```

Three native GGUF integration tests may report `SKIPPED` on a normal checkout.
They require optional locally compiled artifacts and model weights; a skip does
not hide a failure in the shippable Ollama-backed runtime.

## Live acceptance check

After `make doctor` succeeds, run:

```bash
make smoke
```

This creates or resumes the selected mind, sends one current event through the
real Ollama speech motor, and prints JSON. A successful process plus a receipt
demonstrates that the local endpoint, model, cortex, SELF pulse, and speech path
are connected. It does not test long-term recall or file execution by itself.

For a live action check, put a trusted Python file in `workspace/` and run:

```bash
.venv/bin/habitus-mind \
  --database state/acceptance.sqlite \
  --workspace workspace \
  --once "/run hello.py" \
  --json
```

Inspect `tool_receipt.status`, `tool_receipt.verified`, the selected output,
return record ID, and process output. Generated prose alone is not proof that a
file ran.

## What the suite covers

The test suite checks, among other things:

- canonical record immutability, supersession, and restart persistence;
- conserved graph weights, directional routing, growth, and exact invariants;
- recurrent drives and persisted neural hidden state;
- absence of transcript and implicit recall text in the current-event renderer;
- explicit memory commit and evidence lookup;
- one-use output authorization and rejection of mismatched actions;
- root confinement, file hashes, execution limits, and sensory returns;
- deterministic developmental curricula and controlled research ablations.

These are separate evidence classes. Deterministic tests establish implemented
behavior, the offline playground establishes integration with a fake renderer,
and a live smoke establishes local-model connectivity. None alone establishes
open-ended competence or validates every historical research result.

## Packaging check

To inspect the source and wheel artifacts without publishing them:

```bash
.venv/bin/python -m pip wheel --no-deps --wheel-dir dist .
```

Generated `dist/`, databases, checkpoints, workspaces, model weights, and
experiment run directories are ignored by Git.
