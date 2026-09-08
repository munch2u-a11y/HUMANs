# Architecture Contract

## Purpose

This package is a compact, inspectable persistent-agent runtime. The current
`habitus-mind` composition puts the conserved bicone graph, immutable evidence,
recurrent desires, random-born cortex, SELF action selection, open-weight speech,
and four receipt-backed abilities behind one CLI. The graph is not treated as an
LLM or a source of factual truth.

It combines six things:

1. a shared semantic surface for language-level endpoint nomination;
2. distinct input and output graph traversal from one `SELF` origin;
3. immutable canonical evidence with direct and graph-local retrieval;
4. verified outcome learning under conserved relative weights;
5. persistent recurrent drives and a persistent born-in neural state; and
6. one-use output authorization with observed action receipts.

The graph is learned routing structure. SQLite records are evidence authority.
The open-weight language model is a replaceable current-event speech motor, and
local abilities are replaceable actuators around that core. See
[Integrated Mind Runtime](docs/INTEGRATED_MIND.md) for the exact product boundary.

## Topology

There is exactly one seed origin:

```text
                              shared semantic crown
                        concepts, vectors, and vaults
                         /                       \
             HEAR -- SEE -- NOTICE       SPEAK -- LOOK -- DO
                         \                       /
                                   SELF
```

The drawing is flattened. Operationally, every crown concept may have distinct
directional ports and paths on the two sides while sharing its meaning, vector,
and vault:

```text
input path -> concept identity + vault <- output path
```

The input trunks are based on causal event metadata, not guessed prose:

- `HEAR`: an immediate message or conversational input;
- `SEE`: an immediate, correlated observation or tool return;
- `NOTICE`: a delayed result, notification, or uncorrelated observation.

The output trunks are basal classes of external effect:

- `SPEAK`: communicate externally;
- `LOOK`: acquire information without changing external state;
- `DO`: execute or mutate external state.

Private model output can remain internal and activate no output trunk.

## Semantic surface and Y traversal

The surface uses vector and lexical overlap to nominate crown endpoints. It does
not decide the internal route. For each admitted endpoint, the Y cipher searches
from `SELF` through the relevant directional graph.

For edge `e` leaving node `v`:

```text
local_probability(e | v) = weight(e) / sum(weight(outgoing(v)))

travel_time(e) = delta_y(e) / (epsilon + local_probability(e | v))
                 + conflict_penalty(e)

path_time(path) = sum(travel_time(e) for e in path)
```

The winning path therefore depends on structural depth and learned familiarity,
not only hop count or surface cosine. Semantic score admits and disambiguates
endpoints; it never rewrites Y-edge travel time.

Associative expansion begins at nodes that the winning Y paths actually visited.
This makes graph retrieval dependent on learned routes instead of merely taking
more nearest neighbors from the language surface.

## Conserved fluid weights

Edges store log-strength rather than an accumulating authority score:

```text
effective_logit(e, t) = log_strength(e)
                        + fast_recency(e, t)
                        - conflict_penalty(e)

global_weight(e, t) = softmax(effective_logit(all live edges) / temperature)
```

All live global weights sum to `1.0`. Each outgoing frontier is normalized again
to `1.0` for the local decision available at that node. Increasing one route
therefore reduces competitors rather than creating unlimited confidence.

Recency is deliberately fast and temporary. Durable strength changes only after
a verified outcome. A claimed action cannot reinforce itself, and any verified
external outcome must carry a receipt identifier.

Graph familiarity expresses subjective routing relevance. It is never used as a
truth score for a memory record.

## Canonical memory and retrieval

SQLite is the single authority. Canonical record text, timestamp, source,
provenance, embedding, type, and metadata are immutable. A correction creates a
new record that supersedes the old record; it does not overwrite history.

The conserved substrate exposes a two-lane `BaseAgenticMemoryRAG.recall()` path:

```text
query
  +-- global direct dense top 3       factual safety rail
  `-- semantic endpoints
        -> weighted Y paths
        -> visited-path expansion
        -> selected vaults
        -> dense + BM25 retrieval
```

The lanes meet only by canonical record ID. Graph candidates cannot evict the
direct safety rail. This path remains available to `habitus-rag` and controlled
memory experiments.

The current `habitus-mind` renderer does not call that retrieval path
automatically. `/recall QUERY` must first be selected as an exact `LOOK` ability;
it then searches active language records and returns canonical text plus record
IDs through a receipt-backed `SEE` cycle. Retrieved text is rendered
deterministically and is not inserted into the Ollama prompt. The older RAG
surface alone retains recently injected record IDs under a bounded context
budget; it does not duplicate them into a second authority.

## Multi-resolution experience memory

Every canonical record has an `experience_id`. For ordinary ingestion this is
the event ID; a conversational turn deliberately shares one ID across its inbound
message, outbound response, and delivery receipt.

The exact text and configured-space embedding remain stored once in the
canonical record.
Lower vaults contain record references plus compact projections:

```text
layer 0  SELF             activation, preference, confidence, pulse
layer 1  stimulus trunk   activation, preference, confidence, pulse
layer 2  preference band  activation, preference, confidence, pulse
layer 3  emergent child   activation, preference, confidence, pulse
layer 4  semantic port    canonical record IDs, language, vector projection
```

The lower `experience_projections` table has no natural-language field. Each
projection stores the shared experience ID, canonical record ID, node, layer,
side, activation, preference, confidence, pulse, and structural metadata.

An experience can receive multiple stability or preference observations over
time. The store keeps their confidence-weighted mean and immediately updates all
lower projections carrying that experience ID. Thus later verified outcomes can
change how the same turn is remembered without rewriting its immutable language.

`SELF`, every basal trunk, every preference node, and every emergent child owns a
lower vault. Vault frequency and preference statistics are derived from the
projection ledger, so they survive restarts without storing duplicate prose.

## Evidence-backed growth

Every input experience is first deposited through `SELF`, its causal stimulus
trunk, and one `STABLE`, `NEUTRAL`, or `UNSTABLE` lower preference vault. A novel
experience starts an overlap cluster inside that parent vault. Later experiences
join only when their canonical embeddings exceed the overlap threshold and their
remembered preferences remain compatible.

Promotion requires distinct experience IDs. Its evidence threshold grows
logarithmically with the size of the parent vault, with a configurable minimum:

```text
required_support = max(base_support, ceil(log2(parent_experiences + 1)))
```

This allows early learning without letting a mature, busy vault turn every pair
of coincidences into a permanent branch.

Promotion creates two linked nodes rather than jumping directly into language:

```text
lower preference parent
  -> unlabeled child with zero semantic vector and a numeric lower vault
      -> crown semantic port with the evidence centroid and language vault
```

The shared supporting record IDs are the bridge. The child is justified by
overlap within the lower vault; its surface location and provisional terms are
derived afterward from those exact records. Semantic similarity therefore cannot
manufacture lower ancestry. Once promoted, later matching traversals continue to
grow the child vault and overlap cluster.

Potential duplicate branches should eventually be joined by reversible bridges
before any destructive merge. Destructive merging is not implemented in the
current release; canonical evidence must never be merged merely because two
vectors are close.

## Runtime flow

```text
current HEAR + pending SEE / NOTICE
  -> immutable canonical records and numeric projections
  -> graph field + recurrent update + persisted cortex advance
  -> cortex proposal participates in SELF candidate ranking
  -> selected one-use SPEAK / LOOK / DO affordance
       +-- SPEAK -> current-event Ollama renderer -> delivered output cycle
       `-- LOOK/DO -> registered local ability -> execution receipt
                                              -> opaque SEE / NOTICE return
                                              -> next full SELF pulse
```

`BornInHabitusRuntime.advance()` owns the transition through state update and
authorization. `IntegratedMind` supplies the current user surface and registers
four exact abilities. `ToolRegistry` writes the action before execution, records
the observed result, routes it through the appropriate sensory lane, and closes
the cycle. Generated prose is never treated as execution.

## Gestation and hatching

Gestation begins from the same minimal seed topology rather than installing a
manufactured biography or skill catalog. It adds:

- one self-identity concept and immutable name record;
- one familiar-human concept and immutable relationship record;
- one small taste branch;
- gentle relative output priors associated with that taste;
- a persistent model/backend profile.

The identity records are the only pinned core memories. A taste statement lives
in an ordinary vault and the taste's edge priors remain subject to the same global
and local normalization as every later edge. The preset can therefore influence
early exploration without becoming a permanent personality command.

The integrated shell uses the replaceable `ChatModel` protocol. Its Ollama
adapter sees a fixed first-person voice directive, the configured voice and
partner names, a qualitative transduction of the current dominant drive and
stability, and only the current HEAR event. It sees no
transcript, retrieved text, `identity.md`, or skill body. Incoming and outgoing
messages become immutable records, and the next human event settles the prior
speech cycle. That return verifies an observed continuation, not the truth or
quality of the reply.

The older `habitus-rag` hatch shell deliberately keeps bounded dialogue and
retrieval context as a comparison surface. Its prompt behavior must not be
confused with the integrated product path.

## Persisted state

The database contains:

- immutable canonical records and supersession links;
- shared crown concepts and their vectors;
- directional input and output edges;
- edge-to-record evidence links;
- vault membership by canonical ID;
- confidence-weighted experience preference state;
- language-free projections in lower node vaults;
- persistent overlap clusters and their child/semantic ports;
- traversal traces and outcome packets;
- embedding-space identity and the pulse counter.

The `habitus-mind` CLI defaults to `habitus-mind.sqlite`; the repository Make
targets select `state/habitus.sqlite` and `workspace/`. The database parent is
created automatically, while the authorized workspace must exist. Passing
`:memory:` remains an explicit opt-in for tests and disposable experiments; it
is never the durable CLI default.

The included deterministic hash embedder is an offline test adapter. A production
embedder must implement `Embedder`, preserve a stable `space_id`, and use the same
dimension for both records and concepts. Opening an existing mind with a different
space or dimension fails rather than silently corrupting retrieval.

## Required invariants

1. One `SELF` origin exists.
2. Its input frontier is exactly `HEAR`, `SEE`, and `NOTICE`.
3. Its output frontier is exactly `SPEAK`, `LOOK`, and `DO`.
4. Input and output paths are directional but share crown concepts and vaults.
5. Global live edge mass sums to `1.0`.
6. Every non-empty local outgoing frontier sums to `1.0`.
7. Endpoint semantic score cannot alter Y travel time.
8. Multi-hop expansion starts from visited Y-path nodes.
9. Direct dense evidence cannot be evicted by graph retrieval.
10. Canonical records are immutable and corrections are explicit supersessions.
11. Unverified output cannot durably reinforce a path.
12. Persisted embedding identity cannot change silently.
13. Lower projections contain no natural-language payload.
14. A promoted child retains every canonical experience that justified it.
15. Opposing preference bands cannot collapse into the same overlap cluster.

`GraphRuntime.validate_invariants()` checks the structural and conservation
invariants at runtime. The behavioral suite separately checks routing, evidence
preservation, supersession, persistence, growth, output classification, and the
receipt gate.

## Honest boundaries

The current integrated release provides conversation, persistent state, explicit
recall, root-confined folder listing and file reading, and bounded Python-file
execution. Its present limits are:

- the random-born cortex participates in state and action ranking but does not
  yet generate generally coherent open-ended speech by itself;
- the interactive CLI does not silently perform gradient training; plasticity
  is an explicit evidence-gated API and research workflow;
- the standard semantic surface is a deterministic developmental byte embedder,
  not a production vector index;
- abilities are explicitly registered rather than discovered autonomously;
- `/run` applies useful POSIX limits but is not a hostile-code sandbox;
- the shipped speech adapter targets Ollama, while custom motors implement the
  small `ChatModel` protocol;
- destructive branch merging, reversible alias bridges, and upper-transformer
  activation control are not implemented; and
- persistent pressure, valence, and neural state are engineering variables, not
  evidence of consciousness or simulated qualia.

Future layers must not hide direct evidence, mutate canonical history, bypass
one-use authorization and receipt verification, or turn graph familiarity into
a fact.
