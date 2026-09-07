# Integrated Mind Runtime

`habitus-mind` is the deployable composition of the born-in developmental
cortex, persistent Habitus graph, recurrent desires, SELF pulse, open-weight
speech, explicit long-term recall, and receipt-backed local abilities.

This document is authoritative for the 0.1.x product path. The historical
white paper and native GGUF notes describe narrower research lineages. For a
clean installation, use [Getting started](GETTING_STARTED.md).

## The runtime boundary

```text
current HEAR bytes ----------------------------+
                                                |
numeric SEE / NOTICE ability opportunity -------+--> SELF pulse
                                                |      |
persistent graph + recurrent desires + cortex --+      +--> SPEAK authorization
                                                       |      |
                                                       |      +--> current-event LLM motor
                                                       |
                                                       +--> exact LOOK / DO affordance
                                                              |
                                                              +--> execution receipt
                                                                      |
                                                        numeric SEE / NOTICE return
                                                                      |
                                                              next full SELF pulse
```

There is one authoritative transition. `BornInHabitusRuntime.advance()` admits
the current sensory frame, grows its sparse graph projections, advances the
recurrent field, advances and persists the cortex hidden state, ranks output
opportunities, and issues one-use authorizations in the same pulse transaction.

The surface LLM is not that transition. It is invoked only after a selected
`SPEAK` affordance exists. The default Ollama adapter gets:

1. a fixed motor contract;
2. the voice and partner names;
3. a compact transduction of current drive urgency, stability, and free energy;
4. the current HEAR event.

It receives no transcript, summary, retrieved record, `identity.md`, or
`SKILL.md` body. Output records state `transcript_records_used=0`,
`recalled_records_used=0`, and `automatic_text_retrieval=false` so this boundary
is inspectable rather than a marketing claim.

The default cortex contains 18,015,141 parameters. Its random-born weights and
lineage are stable, and its hidden state advances and persists with each pulse.
The graph and recurrent desire field evolve during ordinary use. Cortex gradient
updates remain explicit, evidence-gated training operations; the interactive
CLI does not silently train weights on conversational text.

## Long-term memory is not the bloodstream

Every current HEAR event remains an immutable canonical record. Continuity also
lives in a bounded numeric recurrent field and a fixed-size persisted neural
hidden state, so it does not disappear when a text context window rolls over.

Exact old language enters behavior only through a selected inspection:

- `remember that ...` senses the opaque `ability:memory-commit` opportunity;
- that current opportunity constrains `DO` to the exact ability, and SELF must
  still select and authorize it;
- the statement is committed as canonical language memory;
- `/recall QUERY` senses `ability:memory-recall`;
- that current opportunity constrains `LOOK` to the exact ability, and SELF
  must still select and authorize it;
- matching canonical records return as a current opaque `SEE` result;
- the verified evidence is reported deterministically, without putting it in an
  LLM history prompt.

Retrieval still exists because exact evidence lookup is useful. The architectural
difference is where it exists: it is a voluntary sensor action with an output
cycle, a return lane, and a receipt. It is not invisible context injected before
every thought or sentence.

## Identity, drives, tools, and sensing

- Identity and the initial taste are gestated once into the database. The
  renderer receives only the configured voice name, not a repeated personality
  document.
- Six opaque drives and three opaque composite desires are registered in the
  recurrent field. Their human names exist only in the audit manifest and state
  view. Ability paths grow beneath relevant motives, so successful or failed
  receipts change the pressure and valence of the motives on the path.
- Tool descriptions remain adapter-side manifests. Executable competence is an
  opaque graph ability bound to one motor trunk; narration cannot invoke it.
- File contents, paths, result JSON, and return statuses stay canonical for
  inspection but cross SEE/NOTICE as numeric opaque embeddings with
  `membrane_words=false`.
- HEAR remains byte-native. The developmental cortex has no tokenizer or
  pretrained weights and persists its hidden state across restarts.

## Current shippable abilities

| User surface | SELF motor | Sensory return | Result |
| --- | --- | --- | --- |
| ordinary conversation | `SPEAK` | next human `HEAR` | current-event open-weight rendering |
| `/remember TEXT` | `DO` | `NOTICE` | durable explicit fact |
| `/recall QUERY` | `LOOK` | `SEE` | exact canonical evidence |
| `/open PATH` | `LOOK` | `SEE` | root-confined UTF-8 read plus hash |
| `/run PATH` | `DO` | `NOTICE` | isolated, timed, resource-bounded Python run |

The prior recent-turn/RAG chat path remains available as `habitus-rag` so the
architectural difference can be tested directly.

## Run, inspect, and extend it

```bash
make setup
make playground
ollama pull qwen3.5:0.8b
make doctor
make run HUMAN_NAME="Your name" AGENT_NAME=Mira
```

The Make targets use ignored repository-relative `state/` and `workspace/`
locations. The CLI accepts explicit `--database`, `--checkpoint-directory`, and
`--workspace` paths; source code contains no developer-home defaults. A mind's
database and checkpoint directory form one lineage and should be backed up
together after explicit cortex plasticity.

For a different agent or model, implement the small `ChatModel.generate()`
protocol and pass it to `IntegratedMind`. The offline example in
[`examples/api_playground.py`](../examples/api_playground.py) is the minimal
working integration. It uses a tiny cortex and fake renderer but exercises the
real SELF, memory, action, receipt, and sensory-return paths.

Use `make verify` for the CPU-only release gate. Use `make smoke` separately for
one real Ollama-backed turn. [Testing](TESTING.md) explains what each result does
and does not establish.

## Honest boundary

The 18,015,141-parameter cortex owns persistent numeric state and participates
in action ranking, consequence prediction, valence, and future plasticity. A
fresh random-born cortex does not yet produce generally coherent byte speech.
The Ollama motor supplies fluency while receiving only current-event language.
Replacing that text-only motor with a model exposing a numeric prefix or hidden
state hook would let the same graph field couple below the token boundary; it is
an adapter change, not a replacement of the mind, memory, or action runtime.

`/run` limits a selected Python process but does not isolate hostile code from
the operating-system account. The current four abilities are deliberately
explicit; automatic tool discovery and a general shell are not part of this
release.
