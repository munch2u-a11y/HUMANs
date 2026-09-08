# Habitus Born-In Cortex

## Current architecture brief and technical audit

**Status:** integrated functional alpha
**Audit date:** 2026-09-07
**Release:** Habitus Mind 0.1.0
**Primary implementation:** `BornInHabitusRuntime` and `DevelopmentalCortex`

This brief audits the born-in cortex subsystem. The deployable runtime now
attaches a current-event Ollama speech motor downstream of SELF authorization;
see [`INTEGRATED_MIND.md`](INTEGRATED_MIND.md) for that exact boundary.

## Abstract

The Habitus cortex is a local, inspectable developmental system in
which persistent graph state, recurrent activation, sensory growth, motor
authorization, observed consequences, and a randomly initialized neural cortex
share one pulse protocol. The cortex itself does not contain pretrained
language weights or sit behind a retrieval prompt. It begins from a recorded
random seed, consumes current UTF-8 bytes plus a numeric projection of the live
Habitus graph, and changes weights only through explicitly admitted local
training episodes.

The architecture has crossed an important implementation threshold. The
audited research environment ran the full 18,016,166-parameter cortex on a
Radeon 780M, persisted neural
state and hashed checkpoints, grow opaque concepts from three sensory trunks,
authorize all three motor trunks, close output-first experience cycles through
verified returns, and learn simple held-out sensory/consequence regularities in
a procedural nursery without a pretrained tokenizer or rolling text context.

The cortex-only speech path has not yet crossed the open-ended conversational
threshold. The graph
and recurrent kernel still own eligibility and one-use motor authorization, but
the cortex now contributes bounded learned action/consequence proposals to that
live competition. SPEAK can select a self-grown whole-form motor, emit it byte
by byte, carry its hidden state forward, receive a listener's verified return,
and train from that exact closed cycle. Across three tiny seeded conventions,
the coupled path caused 18/18 held-out listener actions without prompt text or
runtime memory retrieval. This remains referential signaling, not composed
conversation: nursery tools are externally scaffolded, HEAR recognition
does not yet drive responses, and the graph and neural cortex remain explicitly
coupled subsystems rather than one differentiable weight field. The most
accurate description of the cortex subsystem is therefore **a causal
developmental hybrid with a working born-in signaling loop**. The shippable
`IntegratedMind` adds coherent current-event speech and four concrete abilities
without representing those adapter capabilities as cortex-only learning.

## 1. Scope and terminology

In this paper, **the current model** means the path assembled by:

- [`developmental_runtime.py`](../src/habitus_ai/developmental_runtime.py);
- [`self_pulse.py`](../src/habitus_ai/self_pulse.py);
- [`developmental_cortex.py`](../src/habitus_ai/developmental_cortex.py);
- [`developmental_curriculum.py`](../src/habitus_ai/developmental_curriculum.py);
- the graph, recurrent field, and SQLite store beneath them.

The product composition additionally includes
[`integrated_agent.py`](../src/habitus_ai/integrated_agent.py), which seeds
persistent drive nodes, installs four exact memory/workspace abilities, and
attaches a current-event Ollama speech motor downstream of SELF.

The older `NoContextPlanner` / `DevelopmentalLanguage` /
[`unified_open_weight_agent.py`](../experiments/graph_native_live/unified_open_weight_agent.py)
path remains in the same repository. It uses
Qwen/GGUF-derived token geometry and a native vocabulary projection, optionally
followed by a transformer continuation. It is a useful ablation and bootstrap
system, but it does **not** instantiate the born-in cortex and should not be
used as evidence for what the current model can do.

Likewise, this paper uses **selection-like** in a precise sense. Branches
compete through local softmax weights; repeated routes deepen; weak
intersections remain dormant; sufficiently repeated intersections promote;
and verified consequences reinforce or penalize credited paths. This resembles
selection pressure, but it is not Darwinian natural selection: there is no
population of reproducing agents, inherited mutation, or differential
reproduction. The implemented mechanisms are sparse feature growth, threshold
promotion, habituation, reward-modulated path learning, recurrent dynamics, and
gradient descent.

## 2. System overview

```text
                    current ENVIRONMENT signals
                               |
                    NOTICE + SEE + HEAR bytes
                               |
            canonical records + trunk-rooted graph growth
                               |
              +----------------+----------------+
              |                                 |
      graph and recurrent field        random-born GRU cortex
              |                        persistent hidden state
              |                        bounded causal proposal
              +----------------+----------------+
                               |
                      SELF candidate ranking
                               |
                  one-use DO / LOOK / SPEAK
                               |
                  output-first experience cycle
                               |
                    observed terminal return
                               |
                  NOTICE / SEE / HEAR next pulse
                               |
          exact graph credit + admissible training evidence
```

The causal control path runs through `SelfPulseKernel`. A proposal bound to the
current model, graph field, and sensory payload contributes bounded action
support, consequence prediction, and confidence to ordinary candidate ranking.
The graph still owns eligibility and one-use authorization, so the cortex cannot
invent or directly execute an unavailable action.

## 3. Main subsystems

### 3.1 Canonical evidence store

`MindStore` is a local SQLite database using foreign keys and WAL mode. Its
main durable categories are:

- immutable event records and links between records;
- concepts, graph edges, edge evidence, and node vault membership;
- route traces and outcome packets;
- output-first experience cycles and their later returns;
- scalar experience preference state;
- recurrent node dynamics and recurrent pulse receipts;
- SELF pulse inbox, pulse states, output authorizations, and internal events;
- developmental admissions, stage transitions, and byte-form occurrences;
- cortex lineage, hidden states, checkpoints, and plasticity receipts.

Canonical record text cannot be updated or deleted. Corrections must be new
records that supersede older ones. Several newer tables, including SELF states,
developmental admissions/transitions, cortex pulse states, and plasticity
receipts, are also protected by immutable-update/delete triggers.

This is not merely logging. Training episode validation resolves record IDs,
route IDs, committed graph-field hashes, action-cycle IDs, and terminal return
IDs back against these authoritative tables before an optimizer step is
allowed.

### 3.2 Dual-cipher graph and six trunks

The graph is rooted at `SELF` and has two directed sides:

- input: `NOTICE`, `SEE`, and `HEAR`;
- output: `DO`, `LOOK`, and `SPEAK`.

The initial topology is not literally seven empty points. It contains one SELF
node, six trunk nodes, and nine fixed input-preference nodes (`STABLE`,
`NEUTRAL`, and `UNSTABLE` beneath each input trunk), for 16 seeded nodes. The
trunk nodes also carry English labels/terms for compatibility with the older
semantic graph. The born-in cortex never reads those labels or terms, but this
is still an engineered structural and representational prior that should be
acknowledged when describing the system as blank.

Sibling edges compete through a local softmax over effective logits:

```text
effective logit = learned strength + decaying recency - conflict penalty
```

A bounded flow propagates conserved mass from a selected graph root. Route
search converts low local probability into greater travel cost, so repeatedly
supported branches become faster/easier paths. Input topology is stored
outward from SELF for consistent growth, while an actual sensation persists the
causal trace in reverse, ending at SELF.

Graph learning has two distinct updates:

- **habituation:** repeated exposure increases edge strength without claiming
  that the event was rewarding;
- **reinforcement:** a verified positive or negative terminal return changes
  only the credited output path, scaled by evidence quality and divided across
  that path.

### 3.3 Trunk-rooted sensory growth

`DevelopmentalByteEmbedder` is the current HEAR receptor. It is fixed, not
pretrained: UTF-8 byte and byte-pair features are mapped into a deterministic
signed hash space. SEE and NOTICE require numeric vectors from an external
sensor or environment adapter. `IntegratedMind` provides deterministic opaque
encoders for its four memory/workspace returns; there is not yet a learned
camera, filesystem, terminal, or body encoder.

For every newly admitted record, the graph grows from the record's actual input
trunk and current preference band. Deterministic sparse hyperplanes create six
coarse-to-fine locality-sensitive sensory fibers. Adjacent co-active fibers can
create up to four within-trunk intersection nodes. A first occurrence is an
archived `experiential_candidate`; recurrence across independent records
promotes it to a traversable `experiential_pattern`.

When at least two sensory trunks settle in the same SELF pulse, a second
bounded pass stages cross-trunk intersections. These are the current mechanism
for concepts such as a repeatedly co-occurring visual and heard pattern. Their
runtime nodes have opaque IDs, empty term lists, numeric centroids, and vaults
containing the records that selected them. They are not named noun or memory
nodes.

### 3.4 Persistent recurrent field

`RecurrentField` is the graph's changing working state. Each registered or
excited node may carry:

- activation;
- pressure;
- valence;
- momentum;
- persistence and baseline pressure growth;
- satisfaction/frustration gains;
- an expression threshold and refractory selection state.

Activation decays by persistence, receives current excitation, and propagates
for two bounded graph steps. Nodes explicitly typed as `drive`, `desire`, or
`desire_composite` receive urgency and readiness calculations. Verified
positive returns relieve pressure on desire nodes present on the credited path;
negative returns frustrate them. Unverified outcomes cannot do either.

`BornInHabitusRuntime` alone does not seed desire nodes, which keeps cortex
nurseries minimal. The shippable `IntegratedMind` calls
`ensure_integrated_foundations()` and persistently registers six primitive
drives plus three opaque composites. Their pressures and valences participate
in live ability paths and change after verified consequences. These are
engineered initial drives whose numeric state evolves; self-grown new drive
categories remain a research goal.

### 3.5 SELF pulse authority

`SelfPulseKernel` provides one reconciliation boundary for a complete current
frame.

1. In its automatic dequeue mode, pending inputs are bounded independently per
   lane by item count and a conservative byte-derived cost.
2. Regardless of arrival order, they settle as NOTICE, then SEE, then HEAR.
3. Each becomes a canonical record and an inward graph trace.
4. All excitation, pressure changes, and deferred outcome effects advance the
   recurrent field once.
5. Candidate output routes are valued using route probability, recurrent pull,
   effort, uncertainty, and context-overlapping verified outcome history.
6. At most one positive route per output trunk receives a one-use
   authorization.
7. The recurrent snapshot, SELF state, cortex extension state, inbox
   completion, and output authorizations commit together.

An authorization is not an action. It must be claimed, matched against its
exact trunk/nodes/edges, used to persist an output-first experience cycle, and
then consumed. Stale, altered, duplicated, or superseded authorizations are
rejected. Interrupted claims are reconciled from cycle metadata at restart.

Returns are physically paired with their action channels:

| Output | Expected return receptor |
|---|---|
| DO | NOTICE |
| LOOK | SEE |
| SPEAK | HEAR |

The generic `ToolRegistry` has a tested kernel-backed mode that persists a tool
cycle before calling its handler and settles success/error through the paired
receptor. `BornInHabitusRuntime` remains reusable without a registry.
`IntegratedMind` constructs one and installs memory commit, memory recall,
workspace folder/file inspection, and bounded Python-file execution. Its explicit command adapter
supplies validated arguments; autonomous language-derived argument formation is
not part of this release.

### 3.6 Developmental curriculum

The curriculum is a persistent admission gate with five ordered stages:

1. `prelinguistic`;
2. `grounded_forms`;
3. `functional_exchange`;
4. `experienced_narrative`;
5. `broader_narrative`.

Each stage allows only the episode classes appropriate to that level. Language
ahead of the current gate remains canonical evidence but cannot become a
training episode. A metric update can advance only one stage at a time.

This is a scaffold, not an autonomous developmental judge. `update_metrics()`
accepts caller-supplied scalar values and does not require the IDs or hashes of
the held-out evaluations that produced them. Its immutable transition log
records the values, but not their provenance. A production nursery must bind
every gate transition to a signed evaluation manifest.

### 3.7 Opaque byte-form learner

Once language episodes are admitted, `ByteFormLearner` considers recurring
2–16 byte spans. It does not use an alphabet, whitespace tokenization, target
vocabulary, or semantic word list. A span becomes a candidate only after two
independent records. Promotion currently requires at least three records,
positive compression gain, at least 0.67 route grounding, and at least 0.20
advantage over the route's global prevalence.

The graph node contains a hashed form ID, no term string, and a numeric route
centroid. Candidate relations remain archived until promotion. Surface bytes
remain recoverable for audit by slicing immutable source records. If a promoted
span exactly covered an entire heard utterance, a hash-checked copy may also
enter the immutable motor-form table. That bounded engram makes the learned act
executable without reading record text at speech time; arbitrary substrings do
not become motors.

The database column and documentation call the prevalence baseline a
`shuffled_control`, but the implementation does not perform a shuffle. It uses
the global frequency of each route across language records. That is a useful
population baseline, but the terminology should be corrected or a real
permutation control should be implemented.

### 3.8 Random-born recurrent cortex

The default cortex contains 18,016,166 trainable parameters:

- 256 byte embeddings plus a boundary embedding;
- three direction embeddings (`hear`, `speak`, and `quiet`);
- a concatenated graph field;
- a three-layer, 1024-wide GRU;
- byte, learned-stop, route-reconstruction, consequence, valence, and three-way
  motor heads.

The default graph field is 256-dimensional and is built by feature hashing the
IDs and scalar states of currently active graph nodes, context nodes, and
selected output routes. Labels, terms, and record text are not placed in this
field. Hash collisions are possible and intentionally force the neural model
to learn a distributed interpretation rather than receive a semantic lookup.

On each SELF pulse, HEAR records advance the GRU with at most 256 current event
bytes. SEE and NOTICE affect it indirectly through the graph field. If there is
no HEAR record, a quiet boundary step still advances the hidden state. The
hidden tensor and exact graph field are serialized into the same SELF
transaction and tied to the current model hash.

Neural consolidation is explicit rather than automatic. Each optimization pass
starts from zero, sorts admitted evidence by pulse, and carries detached hidden
state from one episode to the next. It uses teacher-forced next-byte and stop
losses for HEAR payloads; admitted HEAR may also supply low-weight, explicitly
unverified SPEAK motor rehearsal. Only the agent's exact self-generated message
plus a verified terminal return can fully train SPEAK, consequence, valence, and
motor choice. Graph reconstruction and optional negative-field contrast remain
active. Training therefore experiences life order, but gradients are truncated
at every episode boundary and do not span a complete delayed trajectory.

Successful consolidation writes:

- pre- and post-update model hashes;
- exact record, route, graph-field, payload, action-cycle, and return evidence;
- aggregate losses and optimization-step count;
- a hashed model/optimizer checkpoint;
- an immutable SQLite plasticity receipt.

Only random-from-seed lineages are accepted. A checkpoint from a different
architecture, a pretrained source, a changed artifact hash, a changed model
hash, or non-finite parameters is rejected. Persistent learning uses FP32 on
CPU and ROCm. The measured FP16 path produced a finite first loss but
non-finite parameters after its first Adam update.

### 3.9 Context and memory semantics

The born-in path has no LLM prompt, RAG packet, transcript summary, tokenizer,
or rolling conversation window. It nevertheless has several forms of state:

- the current event byte sequence, clipped to 256 bytes per HEAR record at the
  default cortex boundary;
- the persistent GRU hidden tensor between compatible-weight pulses;
- the persistent recurrent graph field;
- canonical records retained for evidence, audit, form extraction, and
  training validation;
- verified outcome history used numerically to value graph routes.

Thus “no context window” means **no replayed text context supplied to a language
model**, not “no memory” or “no bounded current sequence.” Raw generation and
whole-form motor selection can operate after record-reading methods are
disabled, which is directly tested. Every selected motor byte is an immutable
internal-output event, and the final hidden tensor is persisted against the
accepted stop event before the outbound cycle is opened.

There is no explicit proposition or belief object in the new path. What might
eventually function as beliefs are presently distributed dispositions: graph
route strengths, dormant/promoted patterns, preference estimates, recurrent
valence/pressure, and neural predictions. A verbalized belief would remain a
historical HEAR/SPEAK record and learned byte pattern; it is not automatically
treated as truth.

## 4. End-to-end operating cycle

A normal developmental interaction currently proceeds as follows:

1. A caller submits one or more `DevelopmentalInput` objects.
2. HEAR is converted to fixed byte/bigram receptors; SEE and NOTICE must supply
   numeric sensor vectors.
3. The inputs enter a durable inbox with hashes, source IDs, and lane metadata.
4. The pulse kernel validates the complete frame before reserving a pulse.
5. Each input is persisted as an immutable record and deposited through SELF,
   its causal trunk, and its current preference band.
6. Sparse sensory fibers and same-pulse cross-trunk intersections grow.
7. Inward route traces create excitation for one recurrent reconciliation.
8. A model/field/payload-bound cortical proposal contributes bounded learned
   support while SELF ranks graph output routes and creates up to three one-use
   authorizations.
9. The graph field and current HEAR bytes advance and persist the cortex hidden
   state inside the SELF transaction.
10. Generic DO/LOOK actions may still use caller-supplied environmental carrier
    content. For SPEAK, SELF can instead choose among self-grown whole-form
    motors plus cortex byte likelihood, emit the exact bytes, and persist its
    post-output neural state. The output record then opens an experience cycle.
11. The environment later queues a return through the output's paired input
    receptor.
12. The next pulse persists that return, closes the cycle, credits the exact
    graph path, and applies recurrent satisfaction/frustration.
13. A caller derives validated training episodes and explicitly invokes neural
    consolidation.
14. The optimizer update receives a checkpoint and plasticity receipt only if
    all provenance and numerical checks pass.

There is deliberately no direct “input text → model response” shortcut in this
cycle. The completed SPEAK path currently covers raw learned-stop generation
and selection among promoted whole-form motors; it does not yet recognize heard
forms as grounded input or compose novel multi-form responses.

The product overlay takes a deliberately different final step: after this same
pulse selects `SPEAK`, a local Ollama model renders only the current event plus a
compact qualitative transduction of current dominant drive and stability. After
an exact `/remember`, `/recall`,
`/open`, or `/run` opportunity is selected, the registered handler executes and
its verified return passes through the same next-pulse route. This gives the
shipped application coherent conversation and bounded useful actions while the
cortex-only composition work continues.

## 5. What is genuinely unified—and what is not

The current build is substantially more unified than a memory sidecar around a
chat model:

- every sense and return passes through the same trunk/pulse authority;
- graph growth, recurrent state, output authorization, and cortex state share
  one pulse identity;
- external reward requires a durable action and observed return;
- neural updates are tied back to those same records and routes;
- no pretrained model is covertly supplying the learned behavior in this path.

It is still a coordinated hybrid rather than one homogeneous weight field.
There are at least four distinct adaptive state systems:

1. graph edge strengths, conflict penalties, and topology;
2. recurrent scalar activation/pressure/valence state;
3. GRU parameters and hidden tensors;
4. curriculum/form statistics and promotion state.

The SELF transaction synchronizes them at selected boundaries, but they do not
form one differentiable dynamical system. Causality now runs in both directions
at explicit seams: graph/recurrent state conditions the cortex, and a hashed
cortical proposal contributes bounded action, consequence, and confidence
evidence to SELF's ordinary output competition. The graph still owns candidate
eligibility and one-use authorization; the proposal cannot bypass it.

## 6. Measured evidence

### 6.1 Automated tests

The complete CPU suite is run with `make test`; `make verify` also checks
portability, documentation links, and the offline integrated playground.
Focused coverage includes:

- trunk topology and conserved graph mass;
- bounded sensory growth and cross-trunk promotion;
- deterministic sensory settling and pulse rollback;
- context-conditioned output valuation and one-use authorization;
- kernel-backed tool execution and paired return lanes;
- curriculum admission gates and byte-form promotion;
- cortex restart, lineage, checkpoint, and evidence validation;
- causal cortical participation in SELF output ranking;
- learned neural stops, UTF-8-constrained byte generation, and persisted
  post-speech hidden state;
- whole-form motor promotion that excludes recurring substrings;
- a cold-versus-trained listener/world communication trial;
- refusal to train behavioral heads without a closed verified return;
- non-finite-loss rejection before a plasticity receipt;
- a small CPU developmental nursery.

The regular suite now exercises both a meaningful but tiny learned signaling
convention and the integrated conversation/memory/file-action wiring on CPU.
It does not exercise ROCm on every run, live Ollama generation, autonomous tool
argument creation, or long-duration development.

### 6.2 Radeon execution

The strict full-model probe ran locally with PyTorch `2.12.0+rocm7.14.1`, HIP
`7.14.60850`, and the Radeon 780M:

| Measurement | Result |
|---|---:|
| Parameters | 18,016,166 |
| Compute dtype | FP32 |
| Real forward/backward/Adam update | passed |
| Loss finite | yes |
| All parameters finite after update | yes |
| Weights changed | yes |
| Peak reserved GPU memory | 440,401,920 bytes (~420 MiB) |
| Safety ceiling | 6 GiB |

A bounded full-size integration run also completed nine SELF/cortex pulses,
eight closed action cycles spanning all three motor trunks, one verified
plasticity update, a 205 MiB checkpoint, and zero graph invariant errors. Its
single optimization step is an integration proof, not a learning result.

### 6.3 Extended tiny-cortex nursery

The 78,958-parameter FP32 nursery ran 33 pulses and produced:

| Measurement | Result |
|---|---:|
| Receipt-backed action cycles | 16 |
| Verified behavioral examples in first update | 24 |
| Motor counts | DO 4, LOOK 2, SPEAK 10 |
| Exploration / self-ranked selections | 8 / 8 |
| Held-out graph sensory discrimination | 1.00 |
| Held-out cortex consequence-sign accuracy | 1.00 |
| Held-out next-byte top-1 / top-5 | 0.15 / 0.625 |
| Held-out route cosine | 0.372 |
| Byte-form candidates / promoted forms | 21 / 60 |
| Reported stage after gates | `functional_exchange` |
| Graph invariant errors | 0 |

Interpretation requires restraint:

- sensory discrimination is computed by a graph-feature Jaccard classifier in
  the harness, not by the cortex;
- consequence accuracy covers four synthetic held-out sensor states, each
  represented by two episode rows;
- the byte test uses four invented syllables and four simple frame patterns;
- 60 promoted “lexemes” are recurring 2–16 byte spans, not 60 confidently known
  words;
- reaching `functional_exchange` means the grounded-form gate opened that
  stage; it does not mean functional conversation was demonstrated;
- no free-generation quality was measured.

The final tiny-run database contained 97 records, 33 SELF states, 33 cortex
states, 454 graph nodes, and 21,643 edges, of which 4,584 were archived. The
26 MiB database produced by only 33 pulses is early evidence that edge growth
will become a scaling concern.

### 6.4 Receipt-backed communication nursery

Three 78,958-parameter communication runs each generated three arbitrary forms
from a different seed, exposed each only beside its changing numeric referent,
and promoted only the whole experienced utterances into bounded motor options.
During the held-out phase the agent received `SEE` only. No expected form,
current `HEAR` text, transcript, record retrieval, tokenizer, or pretrained
model entered speech selection. The emitted bytes alone reached a separately
demonstrated listener, and the listener's world action produced the return
receipt.

| Measurement | Result |
|---|---:|
| Seeds | 3 |
| Cold accepted emissions / trials | 0 / 9 |
| Grounded whole-form motors | 9 total; 3 per seed |
| Verified bootstrap communications | 18 / 18 |
| Held-out listener actions correct | 18 / 18 |
| SPEAK dominant on held-out pulses | 18 / 18 |
| Graph-only form choice | 18 / 18 |
| Cortex-only form choice | 13 / 18 |
| Coupled form choice | 18 / 18 |
| Output/internal-state/return receipts complete | 18 / 18 |
| Checkpoint hash preserved across restart | 3 / 3 |
| Open cycles / graph invariant errors | 0 / 0 |

An earlier run that reset hidden state for every training episode produced only
2/6 context-sensitive cortex choices. After pulse-ordered truncated recurrent
carry was introduced and exposure order was randomized, the same-size test
reached 5/6 without changing the graph result. A controlled single-variable
ablation is still needed to assign the improvement precisely.

This is genuine causal referential signaling, but its scope is deliberately
small. The graph already solves each three-way task, the surface outputs are
promoted whole-form motor units rather than composed sentences, the listener is
synthetic, three seeds remain a small sample, and the communication nursery
explicitly asserts the separately tested prelinguistic prerequisite. It is not
evidence of open conversation or language-scale comprehension.

### 6.5 Historical evidence manifest

These report hashes were recorded during the audited development run. The
files themselves are not shipped in a clean source clone, so the table is a
historical manifest rather than currently reproducible evidence. New reports
should go in a run-specific, repository-relative evidence directory before
publication.

| Artifact | SHA-256 |
|---|---|
| `habitus-rocm-cortex-full-fp32-final.json` | `a62886f13a9818fb9180ca3e0577a9a4aab89f8b27d3eb23cfd3fdf74f7f248b` |
| `habitus-cortex-amd-full-v17.json` | `e31e76469117bf903c715b7489a54c81c284f935796fe8d321b7a7b0e5cc904d` |
| `habitus-cortex-amd-v16.json` | `7073f51f36243ed8200e7cf987e5f2baab8dd9308215e17ec659bfe4a9f91ab9` |
| `habitus-rocm-cortex-tiny-fp16-rejection.json` | `268491ded109e395ce065ef5d48741eecd002fb259fd05f041b0f6a1e1811d03` |
| `habitus-comm-nursery-recurrent-01.json` | `8ef22eb156a22196238bfcc9005ce38b919d17a62d3577981365a691424345d0` |
| `habitus-comm-seed-3141.json` | `a6c4d4832bf2936c537f4e1c8b6d5ef12290fc43ac2929a848484b0d09972862` |
| `habitus-comm-seed-1618.json` | `5bbe5ef47e997f8cc36537d1c7a7beb45a705fc456a4f7217aedde39681d454c` |

ROCm emits a `rocSHMEM Could not open libnuma` warning in this environment, but
the tested PyTorch/HIP operations complete successfully. It remains an
operational warning to resolve, not evidence of a failed cortex run.

## 7. Technical audit findings

### High priority: close the actual cognitive loop

**1. Cortex predictions are causal, but the seam remains shallow.** A read-only
proposal is bound to the current model hash, numeric graph-field hash, and
sensory-byte hash. Its action support, expected consequence, and confidence now
alter SELF ranking under a bounded weight. This closes the former one-way seam,
but it is still an explicit scalar coupling rather than one learned weight
field.

**2. The basic SPEAK loop is integrated; composition remains open.** HEAR
experience may create low-weight, explicitly unverified motor rehearsal. The
agent's own exact outbound bytes can receive full SPEAK training only after a
closed verified social return. Generation has a learned stop head, valid-UTF-8
constraints, per-byte internal events, and a persisted post-output hidden
state. Promoted whole utterances can also terminate through their learned graph
boundary. Current evidence covers selection among tiny whole-form motors, not
novel sentence composition or open dialogue.

**3. General learned tool use remains externally scaffolded.** The
communication nursery no longer supplies outbound message text: SELF and the
cortex/graph choose a learned motor form. Earlier motor nurseries still use
Python carriers. The integrated product now selects and executes four exact
memory/workspace affordances with receipts and sensory returns, but it does not
discover unfamiliar tools, derive arbitrary arguments from ordinary language,
or autonomously compose multi-step plans.

**4. Product drives are integrated, but new drive categories are not
self-grown.** The product seeds six primitive and three composite opaque desire
nodes whose pressures and valences evolve through experience. No developmental
rule yet promotes recurring preference/pressure patterns into genuinely new
drive nodes. Minimal cortex nurseries intentionally omit the product seeds.

### High priority: strengthen learning authority

**5. Curriculum gates trust unreceipted metrics.** Any caller can submit perfect
scalar metrics and advance one stage. The transition is immutable but its
evaluation corpus, split, scorer, and artifact hashes are absent.

**6. Neural consolidation is not fully transactional.** Numerical checks stop
a bad update from receiving a receipt, but the in-memory model/optimizer are not
restored to their pre-update state after a late failure. The checkpoint file is
renamed before its SQLite receipt transaction; a database failure can leave an
orphan file and an unreceipted in-memory model. A two-phase update with rollback
is needed before unattended training.

**7. The automatic learning scheduler is absent.** `advance()` does not train.
The caller must choose episodes, batching, optimization steps, timing, and
curriculum metrics, then call `consolidate()`. This is appropriate for a
controlled experiment but not yet a self-maintaining developmental daemon.

### Medium priority: improve developmental validity

**8. Training now carries life order, but remains truncated.** Consolidation
sorts evidence by pulse and carries detached recurrent state across episodes,
which reduced a measured train/inference mismatch in the communication trial.
Gradients still stop at every episode boundary and do not traverse the complete
perception, action, delay, and return trajectory. Any successful weight update
also invalidates the old hidden coordinate system, so the next pulse wakes from
the durable graph rather than migrating neural continuity.

**9. “Blank” still contains significant priors.** The six trunks, preference
bands, direction/action classes, English seed labels/terms, hash receptors,
promotion thresholds, exploration schedules, return-lane mapping, loss weights,
and curriculum order are hand-designed. That is not inherently wrong, but
experiments should distinguish learned structure from supplied ontology.

**10. Nursery metrics remain too narrow for broad agent claims.** Cold-versus-
trained communication, graph/cortex/coupled ablations, hidden carrier text, and
a causally acting listener now exist. There is still no unseen tool task,
long-delay credit test, human speech judge, broad seed sweep, or
open compositional language evaluation. The graph-only condition is already
perfect on the current task, so this test does not establish that the cortex is
necessary for communication.

### Medium priority: bound computational growth

**11. Several hot paths scan the whole graph/history.** Weight snapshots and
recurrent propagation load all edges; some planners load all concepts; output
valuation reloads all verified output history. Consolidation loops over every
episode for every optimizer step and synchronizes loss components back to CPU.
These choices are inspectable and acceptable at nursery scale but will not
remain interactive as the mind grows.

**12. Byte-form linking can still create dense edge multiplication.** Language
forms now bind to co-settled nonlanguage routes rather than every globally
active route, fixing the worst overlap leak. The 33-pulse earlier tiny run still
created 21,643 edges. Candidate budgets constrain work per record, but there is
no global edge budget, consolidation/pruning rule, or sparse active-subgraph
index.

**13. The integrated runtime bypasses the kernel's normal token budget.**
`BornInHabitusRuntime.advance()` explicitly pins all pending/new item IDs and
sets the per-lane item limit to the total frame size. The explicit-selection
branch does not apply `maximum_bucket_tokens`. The cortex clips each HEAR record,
but canonical storage, graph admission, and frame work can still receive an
arbitrarily large caller-supplied batch. The developmental API needs a real
frame-size and byte budget.

### Lower priority: clarify atomic and reproducibility boundaries

**14. Pulse atomicity is intentionally partial.** Recurrent state, SELF receipt,
authorizations, and cortex state commit together. Canonical records, graph
growth, cycle closure, outcome records, and edge reinforcement may commit before
that final transaction. A failed SELF reconciliation therefore leaves
inspectable/retryable evidence but not a fully all-or-nothing world transition.

**15. Source is commit-addressable; historical evidence is not bundled.** The
functional-alpha source and tests are now versioned, but the reports listed in
Section 6.5 and their databases/checkpoints are not shipped. A future research
release must freeze code, reports, state artifacts, and an environment manifest
together before those numerical results count as clean-clone reproducibility.

## 8. Claim matrix

| Claim | Current status | Evidence boundary |
|---|---|---|
| Runs locally on Radeon 780M | demonstrated | strict full FP32 update and bounded integration run |
| Neural weights start without a pretrained model | demonstrated | lineage constraints, tests, checkpoint metadata |
| Uses no rolling text prompt/context window | demonstrated for born-in path | code inspection and record-access-disabled generation test |
| Persists graph, recurrent, and neural state | demonstrated | restart and atomic-extension tests |
| Learns verified simple consequences | demonstrated narrowly | four-state synthetic held-out nursery |
| Grows cross-modal opaque patterns | demonstrated structurally | focused tests and nursery graph |
| Learns useful signals | demonstrated narrowly | three seeded conventions caused 18/18 held-out listener actions |
| Maintains evolving primitive/composite desires | demonstrated in integrated runtime | seeded categories, persisted pressure/valence, receipt tests |
| Grows entirely new desire categories | not demonstrated | developmental promotion rule is absent |
| Chooses and performs useful explicit abilities | demonstrated for four commands | exact SELF affordance, receipt, sensory return, file/memory tests |
| Discovers tools or composes autonomous multi-step plans | not demonstrated | explicit command adapter currently supplies ability and arguments |
| Produces coherent conversation | demonstrated through current-event Ollama motor | cortex authorizes SPEAK; fluency is supplied by open weights, not cortex-only byte generation |
| Forms beliefs comparable to a person or Helix | not established | only distributed dispositions and records exist |
| Is one singular neural weight system | no | coordinated graph, recurrent, curriculum, and GRU state |
| Is a mature AI agent | no | developmental substrate only |

## 9. Recommended next research sequence

### Phase 0: freeze this baseline — source release passed

The source, runnable product, and deterministic tests are commit-addressable.
The remaining research-release work is to copy the historical evidence reports
and their databases/checkpoints into a versioned run directory and add a
machine-readable environment manifest. Preserve the FP16 rejection as a
regression case.

### Phase 1: make cortical outputs causally matter — narrow gate passed

The bounded cortical proposal now carries trunk probabilities, predicted
consequence/valence, graph-field hash, model hash, and confidence into SELF
without bypassing authorization. Graph-only, cortex-only, and coupled form
selection are reported separately. The remaining work is to replace the fixed
blend with a learned/calibrated arbitration rule and test tasks on which each
subsystem contributes nonredundant information.

### Phase 2: complete the SPEAK loop — whole-form gate passed

The distinct motor-language path now separates low-weight HEAR rehearsal from
receipt-backed self-produced speech. Every selected form and byte receives an
internal-output event, changes recurrent state, and ends through either a
learned neural stop or a promoted whole-form boundary; invalid UTF-8 is rejected
before actualization. The remaining gate is productive composition and response
to recognized HEAR input rather than selection among complete learned forms.

### Phase 3: connect specific embodied capabilities — explicit gate passed

The integrated runtime now exposes memory commit/recall, folder/file inspection,
and Python execution
beneath exact DO and LOOK affordances. A constrained adapter owns execution,
and success/error returns re-enter the cortex as SEE or NOTICE. The remaining
gate is learned selection and composition across unfamiliar multi-step
trajectories such as open → write → reread → execute → explain, including
interrupted and misleading-return controls.

### Phase 4: make development evidence-authoritative

Bind every curriculum transition to a frozen evaluation manifest: corpus IDs,
train/held-out partition, seeds, scorer version, predictions, artifact hashes,
and minimum sample size. Rename the prevalence baseline or add a real shuffled
control. Add cold-versus-trained, ablation, paraphrase, delayed-credit, and
negative-control evaluations.

### Phase 5: harden plasticity and scale

Implement optimizer/model rollback, two-phase checkpoint receipts, orphan
recovery, and single-writer training locks. Replace global graph/history scans
with indexed active subgraphs and bounded context-matched outcome summaries.
Batch compatible neural episodes and eliminate per-episode GPU synchronization.
Add edge aging, evidence-aware merging, and explicit global growth budgets.

### Phase 6: investigate emergent drives and beliefs

Only after the closed motor loop works, test whether recurring unresolved
pressure and cross-context outcome patterns can earn opaque drive nodes under
the same candidate/promotion discipline. Treat verbal belief statements as
events, not installed truth. Evaluate whether behavior predicts the inferred
disposition better than the statement itself.

## 10. Conclusion

The present system is a credible experimental foundation. Its strongest
advance is not fluent language; it is the replacement of unverifiable prompt
sidecars with a local causal ledger connecting sensation, graph growth,
recurrent SELF state, authorized action, observed return, and receipted neural
plasticity. The Radeon execution proves that the intended full cortex can be
trained locally with substantial memory headroom.

The first causal language closure now exists: cortical state influences an
authorized SPEAK route, SELF selects and emits the exact motor bytes, a listener
acts because of them, and the verified return changes later preference and
plasticity. The cortex-only next milestone is compositionality: recognized HEAR
forms must activate their lived routes, learned transitions must construct
novel multi-form outputs, and success must survive unseen combinations,
multi-seed controls, and eventually a human communication channel. Meanwhile,
the shippable composition already provides coherent current-event conversation,
explicit long-term recall, evolving drives, and four receipt-backed abilities
through its open-weight motor and deterministic adapters. Those product
capabilities should not be mistaken for cortex-only developmental achievements.
