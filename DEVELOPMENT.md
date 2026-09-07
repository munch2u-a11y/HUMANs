# Developer guide

This document describes the code that ships as Habitus Mind 0.1.x. It is an
implementation map, not an architecture wishlist. For installation and command
examples, begin with [Getting started](docs/GETTING_STARTED.md).

## Development setup

From the repository root:

```bash
make setup
make playground
make test
```

Before shipping, run `make verify`. It adds machine-path and documentation-link
checks to the offline playground and complete CPU suite. Standard tests do not
need Ollama or a GPU.

## The product composition

The `habitus-mind` entry point joins four persistent mechanisms and two adapter
boundaries:

```text
HEAR / SEE / NOTICE
        |
        v
BornInHabitusRuntime.advance()
        |
        +-- canonical SQLite records and graph projections
        +-- recurrent drives, pressure, valence, and stability
        +-- persisted cortex hidden state and cortex proposal
        `-- SelfPulseKernel ranking and one-use affordance
                              |
                 +------------+-------------+
                 |                          |
              SPEAK                     LOOK / DO
                 |                          |
       current-event renderer        ToolRegistry.execute()
                 |                          |
       delivered speech cycle       receipt + SEE / NOTICE
                                            |
                                      next full pulse
```

`BornInHabitusRuntime.advance()` is the authoritative state transition. The
language model is a downstream speech renderer. A tool handler is a downstream
actuator. Neither may invent its own authorization or bypass the returned
sensory cycle.

The default `CortexConfig` has 18,015,141 parameters. Its weights are born from
the lineage seed; its hidden state is advanced and stored on each pulse. Graph
structure and recurrent desire state also evolve during ordinary use. Gradient
updates are evidence-gated operations exposed by the cortex training API and
developmental nurseries; the interactive CLI does not silently train cortex
weights on every conversation.

## Module ownership

| Module | Responsibility |
| --- | --- |
| `integrated_agent.py` | shippable CLI, current-event speech, explicit memory, local abilities, inspectable turn envelope |
| `developmental_runtime.py` | one pulse composition, byte-native admission, graph field, cortex proposal, growth, and persistence |
| `developmental_cortex.py` | random-born recurrent network, hidden-state ledger, evidence validation, plasticity receipts, checkpoints |
| `self_pulse.py` | sensory buckets, recurrent update, candidate ranking, one-use output authorization, consequence settlement |
| `recurrent.py` | primitive drives, composite desires, pressure, urgency, satisfaction, and frustration |
| `graph.py` | conserved directional routing between SELF, six trunks, learned nodes, and crown concepts |
| `store.py` | canonical SQLite records and transactional ledgers |
| `pipeline.py` | graph/evidence facade, ingestion, retrieval, growth, and invariant validation |
| `tools.py` | ability registration, action records, execution receipts, return records, and cycle closure |
| `functional_agent.py` | workspace policy and older `habitus-rag` comparison runtime |
| `models.py` | replaceable chat protocol and Ollama adapter |
| `gestation.py` | one-time identity, relationship, taste seed, and backend profile |

The specialized programs under `experiments/graph_native_live/` are research
lineages and ablations. They are importable and tested where practical, but are
not transitively required by `habitus-mind`.

## Persistent state and lineage

The database is the primary authority for a mind. It contains canonical records,
graph nodes and edges, projections, recurrent snapshots, action cycles, outcome
receipts, cortex lineage metadata, and immutable cortex pulse states. When
plasticity is explicitly run, checkpoint files live beside the database by
default and their hashes are recorded in SQLite.

The CLI creates the database parent directory. The authorized workspace must be
a directory. The default Make targets use these ignored, repository-relative
locations:

```text
state/habitus.sqlite
state/developmental_cortex_checkpoints/
workspace/
```

The name, familiar human, taste seed, cortex architecture, cortex seed, and
embedding space belong to the lineage. Reopening the database validates and
resumes them. It does not re-gestate identity from new flags or silently switch
the cortex configuration.

Never mutate or delete a user's SQLite database or checkpoints during a test.
Use pytest's `tmp_path` or `TemporaryDirectory` for new tests and examples.

## Conversation path

An ordinary message is stored once as the current `HEAR` event. The integrated
runtime admits it, advances graph/recurrent/cortex state, and ranks outputs. If
`SPEAK` is selected, `CurrentEventSpeechRenderer` sends exactly two messages to
the configured `ChatModel`:

1. a fixed motor contract with configured names and a compact numeric-state
   transduction;
2. the current human event.

There is no transcript window, automatic retrieval, summary, identity document,
or skill document in that call. The output record carries counters and flags so
tests can audit that boundary. The next actual human event settles the prior
speech cycle as its observed consequence.

`ChatModel` is intentionally small:

```python
class MyMotor:
    def generate(self, messages) -> str:
        ...
```

Pass an instance to `IntegratedMind`. The complete disposable example is
[`examples/api_playground.py`](examples/api_playground.py).

## Memory path

All admitted events remain canonical records, but stored text is not
automatically placed in the speech prompt.

- `remember that ...` and `/remember TEXT` expose the exact opaque
  `ability:memory-commit` opportunity.
- SELF must select its `DO` affordance before the fact is committed.
- `/recall QUERY` exposes the exact `ability:memory-recall` opportunity.
- SELF must select its `LOOK` affordance before canonical language records are
  inspected.
- Returned evidence crosses the pulse as an opaque `SEE` embedding and is then
  rendered deterministically with its record IDs.

This keeps verbatim evidence retrieval without making retrieval the hidden
bloodstream of every turn. If you add a memory strategy, preserve canonical
record IDs, provenance, the explicit output cycle, and the inspectable count of
records supplied to any language model.

## Ability and receipt lifecycle

The current CLI exposes four abilities:

| ID | Surface | Trunk | Terminal return |
| --- | --- | --- | --- |
| `ability:memory-commit` | `/remember` | `DO` | `NOTICE` |
| `ability:memory-recall` | `/recall` | `LOOK` | `SEE` |
| `ability:workspace-read` | `/open` | `LOOK` | `SEE` |
| `ability:workspace-run-python` | `/run` | `DO` | `NOTICE` |

The command parser supplies an exact, one-turn opportunity rather than asking a
language model to format a tool call. That sensed opportunity constrains its
motor trunk to the exact currently available ability, so a route strengthened
by an earlier command cannot hijack a later command. SELF still ranks that
candidate and must issue its one-use authorization. `ToolRegistry` then:

1. validates and consumes that affordance;
2. writes the output/action record before external work;
3. invokes the handler;
4. records status, output or error, duration, and receipt ID;
5. queues an opaque sensory return;
6. advances the full runtime again and closes the action cycle.

Do not report an action as complete before steps 4–6. For mutations, add an
independent read-back when the environment permits it and assert that artifact
in tests.

### Adding a bounded ability

Add an ability only when there is a concrete user surface and authority policy:

1. define a stable opaque ability ID;
2. choose `LOOK` for observation or `DO` for mutation;
3. implement a narrow handler with explicit limits;
4. register its `ToolDefinition` in `_register_abilities()`;
5. place it below the motives it can satisfy or frustrate;
6. encode returns with `_sensory_encoder()`;
7. require its exact affordance in the execution call;
8. test denial, success, error, receipt, sensory return, and restart behavior.

Avoid exposing a general shell. The included runner accepts one root-confined
`.py` file, launches the current interpreter in isolated mode, clears most of
the environment, disables stdin, and applies POSIX resource limits. It is still
not a security sandbox.

## Portability rules

No code or documentation may depend on a contributor's home directory. Use:

- `Path(__file__).resolve()` for shipped assets relative to a module;
- CLI arguments or documented environment variables for user-selected state,
  workspace, model, and native-library locations;
- `tempfile` for disposable artifacts;
- `os.pathsep`, `sys.platform`, and `os.defpath` for platform-specific runtime
  details.

Do not guess package-manager prefixes or GPU library locations. The optional
native experiments honor `OLLAMA_LIB_DIR` only when a user sets it. Run
`python3 scripts/check_portability.py` after editing source, config, examples,
Makefiles, shell scripts, or documentation.

## Test design

Prefer behavioral assertions over response-only smoke tests. Depending on the
change, inspect:

- canonical record text, type, provenance, and ID;
- selected output trunk and opaque ability ID;
- action receipt, verified return, terminal status, and outcome ID;
- file hash, process return code, or reread final artifact;
- cortex state hash and hidden-state continuity after restart;
- recurrent pressure/valence changes after a verified consequence;
- graph conservation and structural invariants.

The default full suite is `make test`; the full release gate is `make verify`.
Live Ollama tests are intentionally separate under `make smoke`, because model
availability and generated wording are not deterministic unit-test evidence.

## Documentation authority

Use this order when documents differ:

1. current source and passing behavioral tests;
2. [Integrated mind runtime](docs/INTEGRATED_MIND.md) and
   [Architecture contract](ARCHITECTURE.md);
3. subsystem technical briefs;
4. historical white paper and experiment snapshots.

Update docs in the same change as a behavior or command. Keep measured research
results tied to their exact harness and artifact status; do not generalize a
controlled nursery result into a product capability.

## License

Apache License 2.0. See [LICENSE](LICENSE).
