# Habitus Mind

**A persistent local agent whose memory, drives, action selection, and observed
consequences live in one continuing runtime—not in a repeatedly reconstructed
prompt.**

Habitus Mind is a functional alpha you can talk to, restart, teach explicit
facts, and authorize to inspect or run files. Its 18,015,141-parameter
random-born cortex, recurrent desire field, structural graph, and canonical
SQLite record store persist together. A local open-weight model supplies fluent
speech only after the mind has selected a `SPEAK` action.

The result is not a conventional chatbot with extra RAG. Ordinary conversation
does not receive a transcript, summary, retrieved passages, `identity.md`, or
`SKILL.md` body. Old language becomes current experience only when the mind
selects the explicit `LOOK` recall ability. File actions likewise require an
exact one-use affordance, produce a receipt, and return through `SEE` or
`NOTICE` before they can be reported.

This repository ships the complete runnable composition, an offline integrated
playground, behavioral tests, and the earlier RAG path as a control. It does not
ship anyone else's memory database, model weights, or trained checkpoint; a
clean checkout creates a new local lineage owned by its operator.

## What works now

| Capability | Current implementation |
| --- | --- |
| Conversation | Local Ollama model renders the current `HEAR` event after `SPEAK` authorization |
| Continuity | Persisted recurrent field, neural hidden state, graph dynamics, and immutable event records |
| Long-term recall | `/remember` commits canonical evidence; `/recall` performs an explicit receipt-backed `LOOK` |
| File sensing | `/open` reads one UTF-8 file inside the authorized workspace and returns content plus SHA-256 |
| File execution | `/run` executes one named Python file with time, memory, descriptor, and output limits |
| Inspectability | `/state` and `--json` expose pulse IDs, selected outputs, evidence IDs, receipts, hashes, and graph invariants |

## Five-minute offline start

Requirements:

- Linux, macOS, or another POSIX system with Python 3.11+
- enough disk and RAM to install PyTorch

```bash
git clone https://github.com/munch2u-a11y/HUMANs.git
cd HUMANs
make setup
make playground
make test
```

The playground runs the integrated conversation, memory, recall, file-read,
file-run, receipt, and state paths with a tiny cortex and an editable fake
speech motor. It is disposable, CPU-only, and needs no Ollama service, model
download, network request, or existing database.

## Run your own persistent mind

Install and start [Ollama](https://ollama.com/), then:

```bash
ollama pull qwen3.5:0.8b
make doctor
make smoke HUMAN_NAME="Your name" AGENT_NAME=Mira
make run HUMAN_NAME="Your name" AGENT_NAME=Mira
```

By default this uses `state/habitus.sqlite` and authorizes only `workspace/`.
Override any location explicitly; no developer-specific home path is assumed:

```bash
make run \
  MIND_DATABASE=state/mira.sqlite \
  WORKSPACE=workspace \
  HUMAN_NAME="Your name" \
  AGENT_NAME=Mira
```

Names and the initial taste are gestated only when a new database is created.
Reopening that database resumes the same lineage and state. Use another database
path to create another mind.

CPU is the default for both the cortex and Ollama. Add `--allow-gpu` only when
you intentionally want both components to use an available GPU. AMD users can
install the project-specific ROCm environment described in
[Developmental Cortex](docs/DEVELOPMENTAL_CORTEX.md). The standard setup,
playground, and tests do not enable a GPU.

### Try the complete loop

Create a file inside the authorized workspace:

```bash
cp examples/workspace/hello.py workspace/hello.py
```

Then enter:

```text
Hello. What should I call you?
remember that my launch color is ultraviolet
/recall launch color
/open hello.py
/run hello.py
/state
```

For scripts and acceptance checks, process one event and emit a machine-readable
receipt:

```bash
.venv/bin/habitus-mind \
  --database state/mira.sqlite \
  --workspace workspace \
  --once '/run hello.py' \
  --json
```

See [Getting started](docs/GETTING_STARTED.md) for setup variants,
troubleshooting, a copy-paste prompt for another coding agent, and the smallest
editable API integration.

## The execution boundary

```text
current HEAR bytes ----+
numeric SEE / NOTICE --+--> one SELF pulse --> selected SPEAK / LOOK / DO
persistent cortex -----+                              |
graph + desires -------+                              +--> one-use authorization
                                                             |
                                  Ollama speech <--- SPEAK    |
                                  local ability <--- LOOK/DO -+
                                                             |
                                             verified sensory return
                                                             |
                                                       next SELF pulse
```

One call to `BornInHabitusRuntime.advance()` admits the current sensory frame,
updates graph projections and recurrent drives, advances and persists the
cortex, ranks opportunities, and issues one-use output authorization. The
Ollama model is downstream of that transition. It sees only a fixed motor
contract, the current human event, the configured names, and a small numeric
state transduction.

Long-term text retrieval still exists because exact evidence lookup is useful.
Its placement is the distinction: retrieval is a visible action with a motor
cycle, sensory return, and receipt—not invisible text injected before every
thought or sentence.

Read [Integrated Mind Runtime](docs/INTEGRATED_MIND.md) for the precise memory,
language, identity, sensing, and tool boundaries. Read the
[born-in cortex technical brief](docs/HABITUS_BORN_IN_CORTEX_TECHNICAL_BRIEF.md)
for the developmental design and [Architecture](ARCHITECTURE.md) for the
underlying conserved graph substrate. The complete map is in
[docs/README.md](docs/README.md).

## Components

```text
src/habitus_ai/integrated_agent.py       deployable conversation/action surface
src/habitus_ai/developmental_runtime.py  authoritative full-pulse composition
src/habitus_ai/developmental_cortex.py   persistent 18M-parameter born-in cortex
src/habitus_ai/self_pulse.py             sensory settling and output authorization
src/habitus_ai/recurrent.py              drives, pressures, valence, and continuity
src/habitus_ai/graph.py                  conserved structural learning substrate
src/habitus_ai/store.py                  canonical SQLite authority
src/habitus_ai/tools.py                  ability execution and verified returns
experiments/graph_native_live/           reproducible nurseries and ablations
tests/                                   behavioral and structural verification
```

The earlier recent-turn/RAG agent remains available as `habitus-rag`. It is
kept as a comparison and lightweight fallback; it is not the architecture
described above.

## Verification

Run every local release check:

```bash
make verify
```

The tests cover restart continuity, absence of transcript/retrieval injection,
exact action authorization, receipt-backed sensory return, explicit memory,
root-confined file access, bounded execution, graph invariants, developmental
learning, and controlled nursery experiments. A test pass demonstrates those
specific behaviors; it does not stand in for a claim of consciousness or
general intelligence.

The verification target also rejects machine-specific paths, checks local
documentation links, and runs the disposable integrated playground. See
[Testing](docs/TESTING.md) for the exact evidence boundary of each check.

## Security

`/open` resolves the selected file beneath the configured workspace. `/run`
adds useful process limits, but it is **not a hostile-code sandbox**: Python
code you authorize can still use the permissions of your local account. Point
`--workspace` at a directory containing code you trust. See
[Security Policy](.github/SECURITY.md).

## For coding agents

Read [AGENTS.md](AGENTS.md) before changing the runtime. It identifies the
authoritative product path, safe first commands, persistent-state rules,
portability requirements, and the receipts that must back action claims.

## License

Apache License 2.0. See [LICENSE](LICENSE).
