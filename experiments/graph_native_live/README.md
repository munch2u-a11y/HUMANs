# Graph-native live seam

> **Optional research directory.** The shippable conversation, persistent
> memory, and file-action runtime is `habitus-mind`, documented in the
> [project README](../../README.md) and
> [Integrated Mind Runtime](../../docs/INTEGRATED_MIND.md). Nothing in this
> directory is required for `make playground`, `make test`, or normal product
> use. These programs may require a local GGUF, matching llama.cpp headers and
> libraries, substantial compute, or long-running developmental curricula.

## Born-in developmental cortex

The new from-scratch path is documented in
[`docs/DEVELOPMENTAL_CORTEX.md`](../../docs/DEVELOPMENTAL_CORTEX.md). It couples
a persistent raw-byte recurrent cortex to the same SELF transaction, grows
byte forms only after grounded recurrence, and permits behavioral gradients
only from authorized actions with verified terminal returns. It loads no
tokenizer, GGUF, transcript window, or pretrained weights. Run its controlled
nursery with:

```bash
make -C experiments/graph_native_live cortex-nursery
```

## Receipt-backed communication nursery

`developmental_communication_nursery.py` tests the first genuinely causal
speech milestone above byte imitation. A seeded caregiver convention is
generated procedurally rather than installed as a vocabulary. Each form is
heard repeatedly only beside a changing numeric referent. Recurrent spans must
first promote into opaque lexical nodes, and only a promoted span that covered
an entire heard utterance may become a motor option.

On held-out trials the agent receives only a varied numeric `SEE` state: no
`HEAR` text, expected form, transcript, retrieved record, tokenizer, or
pretrained model enters generation. SELF must make `SPEAK` dominant, graph
grounding offers all learned whole-form alternatives, and the recurrent cortex
scores their byte trajectories. The exact selected bytes go to a separately
demonstrated listener. Success means that listener changed the hidden world in
the intended way; only that observed action returns as reinforcement.

```bash
make -C experiments/graph_native_live communication-nursery
```

The JSON report retains cold, graph-only, cortex-only, and coupled scores plus
proposal, output, internal-byte, post-speech-state, listener-return, checkpoint,
and invariant receipts. This is grounded referential signaling, not open-ended
conversation. The script explicitly assumes the measured prelinguistic gate
owned by `developmental_cortex_nursery.py`; it does not claim that prerequisite
from its own language trial.

## Unified recurrent open-weight experiment

This earlier no-transcript GGUF lineage is documented in
[`docs/UNIFIED_OPEN_WEIGHT_RUNTIME.md`](../../docs/UNIFIED_OPEN_WEIGHT_RUNTIME.md).
It adds persistent recurrent activation, endogenous drive pressure, composite
desire nodes, target-free output competition, receipt-gated satisfaction, and
a zero-history developmental language path to the six-lane graph. Speech is
recalculated one token at a time from branching transitions and convergent
semantic support. Before any word is attached, each admitted NOTICE, SEE, or
HEAR event now grows a bounded set of opaque numeric fibers from its actual
trunk and stages repeated within- and cross-trunk intersections. Current word
geometries then bind sparsely to those self-grown routes. Repeated HEAR events
also promote weak lexical candidates into productive vocabulary and build
source-conditioned local sequence habits; there are no desire-owned response
templates. The live planner runs through the shared `SelfPulseKernel`: all
three receptors can collapse into one recurrent SELF update, selected
DO/LOOK/SPEAK routes receive one-use authorizations, and lexical output steps
remain private subevents of that pulse. Run it with:

```bash
make -C experiments/graph_native_live open-weight
```

This is the preferred path for no-transcript experiments. The older seams
below remain as controlled ablations and developmental evidence.

## Original fixed bridge

This is the first narrow end-to-end assembly of:

```text
user input -> Habitus X nomination -> input Y traversal
           -> shared crown -> output Y traversal
           -> bounded graph activations -> continuous GGUF input rows
           -> ordinary generated response
```

The live user text enters Habitus and is persisted as an immutable input
record. It is **not** passed to llama.cpp. The native packet contains only
categorical basis IDs and scalar activations selected from the admitted graph
endpoints. Retrieved memory text and rendered RAG context are also excluded.

The native bridge uses an empty Qwen chat envelope and constructs one
continuous, non-token semantic slot for each activation by combining a small
fixed codebook of token-embedding anchors. Generated tokens then use the
ordinary llama.cpp path. This is a train-free bootstrap bridge, not the final
learned graph projector.

The default model location is the repository-relative
`models/Qwen3-0.6B-Q8_0.gguf`. Its native input width is exactly 1024, so the
graph boundary and decoder require no dimension-changing projection. Supply a
different local location with `MODEL=path/to/model.gguf` or the script's
`--model` flag. No user home, package-manager prefix, or fallback model path is
assumed.

## Run it

The build uses the llama.cpp revision matched to the installed Ollama runtime:

```bash
make -C experiments/graph_native_live build
make -C experiments/graph_native_live live MODEL=models/Qwen3-0.6B-Q8_0.gguf
```

The default source checkout is `third_party/llama.cpp`. To use locations
elsewhere, pass absolute values assembled by your shell rather than embedding a
developer's home in source:

```bash
make -C experiments/graph_native_live build \
  LLAMA_CPP_SOURCE="$PWD/vendor/llama.cpp" \
  OLLAMA_LIB_DIR="$PWD/vendor/native-libraries"
```

`OLLAMA_LIB_DIR` is optional when the dynamic linker can already find the
required libraries. It is honored only when explicitly set.

Or run one auditable turn:

```bash
PYTHONPATH=src python3 experiments/graph_native_live/live_tester.py \
  --once "hello there" \
  --show-trace
```

Each turn writes two artifacts under `experiments/graph_native_live/runs/`:

- `*.packet`: the exact bounded packet consumed by the native adapter;
- `*.json`: graph traces, packet assertions, native-run metadata, and output.

The persistent experimental mind defaults to
`experiments/graph_native_live/live_mind.sqlite`.

## What the current result establishes

- A real Habitus input pulse can nominate a shared crown endpoint and traverse
  both halves of the bicone.
- That graph result can become continuous model input without serializing the
  live input or recalled memories as a prompt.
- A frozen local Qwen GGUF can generate coherent continuations from those
  graph-derived soft slots.
- Different admitted concepts can cause different response categories.

## What it does not establish

- The codebook is not learned and covers only a few broad categories.
- Novel factual content does not cross the adapter yet.
- The model still requires fixed role-delimiter embeddings to enter assistant
  generation mode; those delimiters contain no user or memory content.
- This does not yet outperform prompt RAG or prove upper-layer injection.
- A response can be coherent but generic. That is acceptable for this seam
  test and is not counted as factual recall.

The next real gate is replacing the fixed codebook with a trained projector
from a canonical graph packet while holding this native input and receipt path
constant.

## Opaque, label-free baseline

`opaque_skeleton.py` removes the codebook entirely. It creates two opaque
layer-three branches below separate preference paths, pulses them with numeric
stability values, and then connects them through a third opaque node. Four
1024D rows are derived solely from:

- the active input path;
- current dynamic edge weights;
- recent pulse order and numeric stability;
- the active output path.

Node IDs are opaque, the embedder has no lexical-similarity behavior, and the
native packet contains no labels, words, or model embedding anchors. Run it
with:

```bash
make -C experiments/graph_native_live opaque
```

The matrix includes separate branches, their connected state, an exact repeat,
row reversal, sign inversion, and an unrelated opaque control. Coherent text
from these packets demonstrates only that Qwen maps arbitrary native-width
directions onto its language manifold. It does not give those directions
grounded meaning.

## Developmental lexical nursery

`nursery.py` tests a narrower version of early label formation. Internal
concept nodes remain opaque and nonverbal. A word exists as a separate lexical
surface node, using the exact tokenizer and 1024D token-embedding row from the
local Qwen GGUF. Ordinary input and output graph edges act as vertical fibers
between that surface and the lower concept active during the same pulse.

Run the deterministic curriculum and controls with:

```bash
make -C experiments/graph_native_live nursery
```

The primary curriculum presents `I`, ` like`, and ` Josh` separately. It never
presents `I like Josh` as a complete phrase. A held-out output traversal must
recover the ordered lower path, select the learned productive fibers, and emit
the complete token sequence. The run also includes a substitution curriculum,
a deliberately shuffled binding control, an untrained control, receptive
probes, one-pulse-delayed caregiver feedback, invariant checks, and a JSON
receipt under `nursery_runs/`.

This establishes graph-native receptive/productive label binding and ordered
composition in a tiny controlled curriculum. It is not yet free language
generation: the graph currently chooses the strongest learned lexical fiber
at each point, and Qwen supplies its tokenizer and lexical embeddings rather
than transformer inference. The next gate is to turn fiber activations into
bounded candidate logits or hidden-state guidance and let the frozen model
perform token selection.

## Embodied action nursery

`embodied_nursery.py` exercises the other half of development without a GGUF
or a language prompt. It starts with five opaque motor fibers and a sealed
virtual room. Each native memory cycle begins with an output, records a real
success or error return, links that return to the output, updates the shared
per-layer experience projections, and closes back at `SELF`.

```bash
make -C experiments/graph_native_live embodied-nursery
```

The curriculum covers orient, open, read, write, and run behavior, but those
names exist only in the developer report. The graph sees IDs such as
`ability:a0`, structured feature vectors, and changing opaque state nodes.
Wrong actions are safe verified errors and leave the environment unchanged;
successful actions advance it. Output selection is target-free during probes,
so the post-training score measures whether transient state plus learned edge
weights choose the appropriate action path—not whether a test supplied the
answer as an endpoint.

The environment rejects absolute paths and parent traversal, exposes no shell
or network, limits writing to its scratch area, and interprets only a
whitelisted toy instead of executing arbitrary code. A JSON receipt reports
baseline/trained action accuracy, closed causal cycles, verified successes and
errors, edge-mass conservation, boundary checks, and graph invariants.

This remains a controlled developmental curriculum, not evidence of general
agency. It does establish that the same graph can carry perception, action,
consequence, and habit formation without translating the action fibers into a
natural-language prompt.

## Backwards lexical projection

`reverse_nursery.py` removes the diagnostic token lookup from the production
path. The caregiver still knows which token accompanied an experience, but the
graph stores neither that token ID nor its word as node metadata. Label
exposure anchors Qwen's 1024D lexical geometry to the co-active opaque concept.
On output, current fiber weights blend those geometries into one state per
point on the ordered lower path. The native codec compares each state with the
model's complete vocabulary projection and returns its nearest tokens.

```bash
make -C experiments/graph_native_live reverse-nursery
```

The same primary, substitution, shuffled, and untrained controls apply. This
is a stronger inverse-membrane test, but still not transformer inference: it
uses Qwen's vocabulary projection directly. A later experiment can inject
these graph states before or within the transformer so context and learned
language dynamics influence token selection too.

## Accelerated gestation

`accelerated_gestation.py` grows a larger persistent mind without waiting for
wall-clock childhood. It mass-embeds a controlled developmental curriculum in
the exact Qwen GGUF token geometry and replays those episodes through the
ordinary experience vault and overlap-growth kernel. Topic patterns promote
into opaque children and semantic ports. Session coactivation then invokes the
same kernel with those learned concepts as parents, producing category and
domain assemblies rather than a separately hard-coded hierarchy.

The promotion is now explicitly two-stage. Curriculum stimuli first form opaque
nonverbal children under their causal HEAR/SEE/NOTICE branches; their raw lesson
text is developer evidence and cannot populate a semantic vault. A separate
caregiver message through HEAR then attaches the word-bearing semantic port.
Recursive assemblies use the same nonverbal-growth-then-HEAR-label sequence.

```bash
make -C experiments/graph_native_live gestate-fast
```

The compiler adds mirrored output paths, temporal coactivation edges, and an
opaque lexical membrane, then closes and reopens the database before applying
its hatch gate. Its controls include label-absent semantic paraphrases, direct
graph-to-vocabulary production, shuffled expected labels, global/local mass
invariants, recursive path reachability, exact holdout leakage checks, and
restart persistence.

The curriculum is controlled and its category sessions are teacher supplied.
Topic labels receive deliberate caregiver emphasis during productive-fiber
formation. Consequently, productive vocabulary accuracy measures whether the
grown graph preserves and reverses those bindings; it is not evidence that
language or the curriculum categories emerged without supervision. The mass
embedding pass uses averaged native token rows rather than contextual
transformer states in this first scalable build.

Probe a completed snapshot as a minimal graph-native conversational mind with:

```bash
PYTHONPATH=src python3 experiments/graph_native_live/probe_hatched_mind.py \
  --database experiments/graph_native_live/accelerated_gestation_runs/MIND.sqlite
```

Every probe enters through the ordinary `HEAR` trunk. X selects a semantic
endpoint from native geometry, Y must find a legal path from SELF, and the
selected concept's weighted output fibers are projected through the complete
GGUF vocabulary. No natural-language prompt is sent to the transformer. The
result is currently a single lexical response, not fluent conversation.

## Graph-to-transformer hatch

`transformer_hatch.py` takes the next step. A novel message is embedded and
resolved entirely by the hatched graph. The selected concept's productive
lexical fibers are ordered by directed word-transition edges learned from
developmental episodes. Up to eight 1024D lexical-geometry rows enter Qwen
inside a fixed empty-chat envelope, after which the ordinary transformer and
sampler generate a response.

```bash
PYTHONPATH=src python3 experiments/graph_native_live/transformer_hatch.py \
  --database experiments/graph_native_live/accelerated_gestation_runs/MIND.sqlite
```

The matrix holds model, seed, sampling, and structural delimiters constant. It
compares the selected state with reversed row order, the least-related learned
concept, and unrelated opaque rows. Receipts assert that no user string,
retrieved episode text, token ID sequence, or hand-authored semantic codebook
crosses the native boundary. `HABITUS_NATIVE_SKIP_THINK` preloads only Qwen's
empty reasoning delimiters so the bounded run reaches visible answer text.

This is native input-embedding guidance, not upper-layer activation injection.
The rows contain learned lexical geometry and therefore carry language
information even though they contain no serialized words. Current evidence is
limited to broad learned concepts; arbitrary episodic facts, grammar learned
outside the controlled curriculum, and open-ended action selection remain
separate gates.

## Graph-structured language pulse

`latent_language_pulse.py` combines the learned membrane with the graph state
instead of sending only the selected concept's word fibers. Every pulse first
activates its input paths, immediately recomputes per-node sibling softmaxes and
the conserved flow from `SELF`, optionally applies an observed input-stability
signal, activates its output paths, and performs one final recalibration. The
adapter frame is built only from that final snapshot.

The frame contains a whole-mind edge field, directional input and output
Y-route fields, the overlapping activated Layer 3 concepts, and up to four
learned membrane geometries. At the native boundary those graph fields are
overlaid onto each membrane row, gradually shifting from the input route toward
the output route. This preserves a normal word-length sequence instead of
inserting graph fields as extra pseudo-words. The query selects graph endpoints
but its text and embedding are not included in the packet. The largest current
root action gate decides the route: `SPEAK` sends generated language outside;
`LOOK` or `DO` marks it private and self-originated.

Run it against a copied gestated database because a pulse intentionally changes
edge recency and therefore the next propagated flow state:

```bash
mkdir -p state/experiments
cp experiments/graph_native_live/accelerated_gestation_runs/MIND.sqlite \
  state/experiments/habitus-mind.sqlite
PYTHONPATH=src:experiments/graph_native_live python3 \
  experiments/graph_native_live/latent_language_pulse.py \
  --database state/experiments/habitus-mind.sqlite \
  --once "People consistently keep promises, making cooperation feel safe." \
  --ablations
```

This is still an input-embedding adapter, not a trained upper-layer projector.
Its purpose is to test whether the existing transformer responds more usefully
to the combined graph structure and learned labels while preserving a receipt
for every real-time softmax recalibration.

### Initial controlled result

Using the same `habitus-1787966680339559785.sqlite` snapshot, Qwen3 0.6B Q8,
seed 42, and a 48-token cap, the mean output similarity to the X-selected
concept across the trust, fear, evidence, and music probes was:

| Input condition | Mean similarity |
| --- | ---: |
| contextual membrane overlay | 0.473 |
| membrane labels only | 0.461 |
| reversed contextual rows | 0.461 |
| graph fields as separate rows | 0.342 |
| graph structure only | 0.343 |
| unrelated opaque rows | 0.210 |

The overlay produced useful topic-directed language for trust, fear, and
evidence; the music response remained an unhelpful clarification request. The
small gain over membrane-only is encouraging but not decisive. In particular,
the strong reversed-row result means this trial does not yet establish learned
sentence order. It establishes only that the contextual graph field can alter
and sometimes improve the learned label projection without prompt or RAG text.

A separately observed positive output outcome moved the live `SPEAK` root gate
from 0.5134 to 0.5248. Every credited output edge increased, the genuinely
competing root mass decreased, and the pre- and post-feedback root flow budgets
were exactly 1.0. Deeper unrelated regions no longer dilute that route.

## Target-free outbound focus

`outbound_focus.py` removes the symmetric output shortcut used by the first
language pulse. Input X still nominates concepts, but output receives only the
transient activation left on the graph. A bounded Y beam starts again at
`SELF`, follows the live output weights, and discovers its own Layer 3
terminals. No X endpoint is accepted as an output target.

The output distribution is hierarchical:

1. the three `SELF -> output trunk` edges provide learned membrane priors;
2. the strongest transiently relevant path supplies current membrane focus;
3. a separate softmax chooses a complete trajectory inside each membrane; and
4. gate probability multiplied by within-membrane probability conserves one
   total output mass.

This separation matters. An earlier implementation derived the membrane gate
from mean complete-path probability. Because every local softmax divides mass
among its children, a sparse action library then beat a richer communication
branch simply by having fewer endpoints. The regression fixture reproduces
that failure with thirteen speech endpoints and one action endpoint.

Communication, navigation, and action now terminate at different interfaces.
Only communication constructs GGUF language geometry. Navigation emits a
structured affordance and action emits an ability identifier; neither passes
through the language adapter. Up to two membranes may be admitted in one pulse,
so an operation and a reply do not have to corrupt one another's surface form.

Run one live pulse against a disposable database copy:

```bash
PYTHONPATH=src:experiments/graph_native_live python3 \
  experiments/graph_native_live/outbound_focus.py \
  --database state/experiments/habitus-mind.sqlite \
  --once "People consistently keep promises, making cooperation feel safe." \
  --maximum-membranes 2 \
  --run-directory state/experiments/habitus-outbound-run
```

On the accelerated gestation snapshot, the trust pulse assigned `0.878` gate
mass to communication and `0.056` to action. Without prompt text or user tokens
at the GGUF boundary, it began: "The terms 'trust,' 'feel,' 'behavior,' and
'cooperation'..." Explicit execution language instead assigned `0.565` to
action. File/search/evidence language assigned `0.760` to navigation.

`outbound_focus_ablation.py` evaluates routing only. It holds a single edge
snapshot fixed, uses a read-only local Dijkstra pass, performs no activation or
reinforcement, and never invokes generation:

```bash
PYTHONPATH=src:experiments/graph_native_live python3 \
  experiments/graph_native_live/outbound_focus_ablation.py \
  --database state/experiments/habitus-mind.sqlite \
  --output state/experiments/habitus-outbound-ablation.json
```

Across the 36 accelerated-nursery topics, the hierarchical gate recovered the
curriculum's intended output membrane for 32/36 top choices (`88.9%`) and 35/36
top-two choices (`97.2%`). The rejected full-path gate recovered 8/36 (`22.2%`).
The complete logical SQLite dump had the same SHA-256 before and after the
corrected run. The four top-one misses were calm, curiosity, confidence, and
tests; their X nominations emphasized adjacent concepts such as danger,
checking, or uncertainty. These remain diagnostic retrieval/representation
cases rather than gate-tuning targets.

The same read-only matrix was also run across the three saved developmental
snapshots:

| Snapshot state | HEAR coverage | New top-1 | New top-2 | Old top-1 |
| --- | ---: | ---: | ---: | ---: |
| before language schooling | 30/36 | 12/30 | 29/30 | 8/30 |
| 35 concepts language-schooled | 36/36 | 32/36 | 35/36 | 8/36 |
| plus 358 lexical transition edges | 36/36 | 32/36 | 35/36 | 8/36 |

The first change added language-schooling records and input-side connections;
coverage and top-one modality selection improved sharply. Adding surface
lexical transitions afterward did not alter the routing score. This is
consistent with the intended separation: language alignment can make incoming
concept activation usable without letting word-order fibers rewrite deeper
action selection. These are sequential artifacts from one controlled
curriculum, not independent random seeds, so cross-seed replication remains
open.

This result establishes target-free modality routing on this controlled mind.
It does not establish fluent graph-native language. Trust was coherent, while
fear and evidence were related but rough and music was on-topic but
pragmatically poor. Tool packets are structural intentions only until a real
ability registry, execution receipt, and verified outcome loop are connected.

## LoCoMo developmental replay

`locomo_self_pulse_trial.py` tests episodic conversation without turning the
runtime back into a prompt-based RAG system. Every utterance is admitted in
chronological order through `HEAR`. Session boundaries enter through `NOTICE`,
and image captions enter through `SEE` beside the associated utterance. Both
corpus speakers remain external sources; their lines are never represented as
speech generated by Habitus.

Questions and reference answers are held out until development finishes. Each
question is asked independently of an untouched nursery baseline and a copy of
the trained mind. Graph-native speech is scored first. Ordinary record recall
is inspected only afterward and reported separately, so retrieving an evidence
turn cannot silently become a context injection. Interrupted replay can resume
from a trained database without applying already-committed turns twice.

```bash
python3 experiments/graph_native_live/locomo_self_pulse_trial.py \
  --sessions 5 \
  --questions 5 \
  --categories 1 \
  --with-native-continuation
```

The receipt distinguishes exact answer accuracy, answer-token overlap, exact
evidence-turn recall, graph growth, per-turn trunk-rooted growth receipts,
sensory modality counts, restart history, phase timings, and invariant
failures. It is a developmental diagnostic; teacher-forced speech, fluent but
incorrect text, and post-generation retrieval are not counted as graph-native
answering.
