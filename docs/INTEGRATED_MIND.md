# Integrated Mind Runtime

`habitus-mind` is the deployable composition of the born-in developmental
cortex, persistent Habitus graph, recurrent desires, SELF pulse, open-weight
speech, explicit long-term recall, and receipt-backed local abilities.

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

## Long-term memory is not the bloodstream

Every current HEAR event remains an immutable canonical record. Continuity also
lives in a bounded numeric recurrent field and a fixed-size persisted neural
hidden state, so it does not disappear when a text context window rolls over.

Exact old language enters behavior only through a selected inspection:

- `remember that ...` senses the opaque `ability:memory-commit` opportunity;
- SELF must select that exact `DO` affordance;
- the statement is committed as canonical language memory;
- `/recall QUERY` senses `ability:memory-recall`;
- SELF must select that exact `LOOK` affordance;
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

## Honest boundary

The 18,015,141-parameter cortex owns persistent numeric state and participates
in action ranking, consequence prediction, valence, and future plasticity. A
fresh random-born cortex does not yet produce generally coherent byte speech.
The Ollama motor supplies fluency while receiving only current-event language.
Replacing that text-only motor with a model exposing a numeric prefix or hidden
state hook would let the same graph field couple below the token boundary; it is
an adapter change, not a replacement of the mind, memory, or action runtime.
