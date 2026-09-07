# Unified open-weight runtime

The random-weight cortex that grows directly inside this pulse contract is
documented separately in
[`DEVELOPMENTAL_CORTEX.md`](DEVELOPMENTAL_CORTEX.md). The GGUF-backed path below
remains a useful ablation and bootstrap system; it is not the cortex's language
engine.

This build makes the six-lane Habitus graph the durable working system rather
than a retrieval sidecar around a chat model. It is deliberately independent
of the Helix identity import.

## Runtime contract

```text
pending NOTICE + SEE + HEAR signals
        |
        v
bounded lane frame, settled NOTICE -> SEE -> HEAR
        |
        +--> nonverbal geometry through its own receptor
        |
        +--> HEAR geometry + ordered token geometry
        |
        v
observed endpoints collapse inward through input Y routes
        |
        v
SELF -- one persistent recurrent reconciliation for the whole frame
  activation + pressure + valence + momentum
        |
        v
DO + LOOK + SPEAK opportunities
  live route pull + effort + context-matched verified consequences
        |
        v
one-use motor authorizations
        |
        +--> tool/action cycle -> observed receipt -> exact path credit
        |
        +--> SPEAK start state -> one winning word at a time
                               -> private recurrent output substep
                               -> stop -> local GGUF projection
        |
        v
source/content return or explicit reward
        |
        v
exact path credit -> changed graph/drive/source weights
```

There is no rolling transcript, recalled-memory packet, summary, persona block,
or hidden text cache in this path. One native lexical sensing call returns the
current utterance geometry and one exact Qwen embedding row per current token.
The utterance is stored as immutable developmental evidence, but neither
planning nor speech reads stored record text. Developmental promotion counts
only record membership and graph evidence. Continuity comes from SQLite graph
weights, recurrent node state, and learned source state.

Speech has no token-history window either. The only sequential cursor is the
currently active lexical node. Each selected word is persisted as recurrent
activation before the next competition. Word and private-thought changes are
immutable internal-output receipts beneath the current SELF pulse; they do not
quietly create unrelated sensory pulses or invalidate that pulse's selected
motor route. A temporary list of winning geometry rows exists only to render
and receipt the output after those decisions; it is not consulted to choose
another word.

The default speech actuator projects selected lexical rows directly through
the local Qwen GGUF vocabulary matrix. It performs no autoregressive generation
and therefore needs no decoder context window. `--with-native-continuation` is
an optional experiment: it starts a fresh transformer context from at most 16
numeric graph rows and fixed role delimiters. That optional decoder necessarily
uses a temporary KV cache while generating one response, but it receives no
conversation history, user tokens, memory text, node IDs, or node labels.

## What became unified

`SelfPulseKernel` is now the pulse authority used by `NoContextPlanner`. Its
SQLite inbox is transport plumbing, not another attention model. It admits a
bounded set from each receptor in fixed causal order, gives the entire frame
one pulse identity, reverses stored input paths so their actual traces end at
`SELF`, and advances the existing `node_dynamics` field once. There is no
second waterwheel/focus table.

The immutable `self_pulse_states` receipt contains the full recurrent scalar
state, input record and trace IDs, stability/free-energy diagnostics, every
valued output candidate, and the selected authorization IDs. Recurrent state,
that receipt, inbox completion, and authorizations commit in one transaction.
Canonical sensory records and graph-growth evidence are admitted first, so a
failed final reconciliation can leave inspectable input evidence, but it
cannot expose a half-updated SELF state. Malformed frames are rejected before
a pulse is reserved.

DO, LOOK, and SPEAK remain independent output trunks. The kernel can select at
most one positive opportunity per trunk. It values a route using current
recurrent pull, learned route effort, uncertainty, and only verified outcomes
whose specific terminal path and prior semantic/source state overlap the
current state. This avoids treating every historical success on a broad trunk
as evidence for every action beneath it. A selected route is still only a
proposal: `actualize_output` must claim its one-use authorization, persist an
output-first experience cycle, and then consume the authorization. Interrupted
claims are reconciled against durable cycle metadata on restart. Arbitrary,
stale, altered, and duplicate routes are rejected.
Confirmed results and errors can be queued with `enqueue_cycle_return`; several
tool or communication consequences then settle through their natural input
trunks in the next shared pulse. Their canonical receipts and exact edge credit
are recorded first, while their recurrent satisfaction/frustration update is
deferred into that pulse's atomic SELF reconciliation.

`ToolRegistry` remains backwards compatible for the older experiments, but a
registry constructed with `pulse_kernel=...` switches to the embodied path. In
that mode it refuses an ordinary caller-supplied decision, requires the exact
selected affordance, persists the motor cycle before invoking the handler, and
settles the handler's success/error through the paired receptor (`DO` to
`NOTICE`, `LOOK` to `SEE`, `SPEAK` to `HEAR`). Thus the substantive tool payload
stays in the environmental module while selection, action, receipt, and learned
consequence remain visible to the same mind.

`node_dynamics` lives in the same authoritative mind database as concepts,
edges, experiences, and receipts. For each recurrent node it persists:

- activation: decaying present working state;
- pressure: an unmet tendency that can grow without external input;
- valence and momentum: recent outcome direction and state change;
- persistence, growth, satisfaction, frustration, and expression thresholds.

These values are not prompt annotations. They directly alter path competition.
A positive verified return lowers pressure only on the desire node contained in
the output's exact credited path. A negative return increases that pressure.
Unverified narration changes neither edge strength nor desire pressure.

Communication sources also become opaque `social_source` graph nodes. The
canonical event log retains the transport source ID for provenance, while the
working graph uses only a stable hashed node ID. Repeated interactions grow
source-to-concept and source-to-lexeme routes. A small scalar source preference,
its evidence weight, and observation count live in the same mind database and
condition later learning and social-return interpretation.

The developmental extension grows six basal drives from combinations of the
already-grown concepts and three convergent composite drives. Runtime nodes use
opaque IDs and empty term lists. Their human-readable names exist only in an
audit manifest; runtime routing does not read it.

There are no desire-owned sentences. The caregiver exposes single-token words
against several co-active semantic concepts and exposes reusable two-state
word transitions. Those transitions branch, while independently active
concepts converge on the same shared geometry-only lexeme nodes. At every
speech pulse the runtime combines local transition strength, convergent
semantic support, current lexical activation, and the pressure of the selected
output route. It then actualizes one word and recalculates from the changed
field. A nonverbal stop node competes through the same graph.

## Trunk-rooted experiential breadth

The live planner no longer depends on the accelerated nursery's pre-existing
crowns for every new distinction. Each newly admitted `NOTICE`, `SEE`, or
`HEAR` record first grows through its actual input trunk and current preference
branch. Six deterministic locality-sensitive receptor fibers sample different
parts of the current numeric sensory field at progressively finer resolutions.
Coarse banks can recur across related experience while fine banks preserve
contrast. Their IDs encode only trunk, preference band, receptor bank,
resolution, and numeric signature; their labels are their opaque IDs, their
term lists are empty, and their vaults retain the independent records that
selected them.

Up to four simultaneous fiber intersections are staged per receptor event. A
one-off intersection is an `experiential_candidate`: its parent edges are
dormant, so accidental noise can add contrast without immediately becoming a
usable belief or route. Independent records that select the same intersection
promote it to an `experiential_pattern` and awaken those edges. When two or more
receptors settle in one SELF pulse, a second bounded stage grows cross-trunk
intersections from their co-active fibers. This gives, for example, a repeated
SEE/HEAR pairing one shared lower route without manufacturing a text-labelled
"object concept."

The language membrane runs only after this nonverbal growth. Each current word
geometry chooses a small number of its best-aligned lived routes, and every
co-active route receives one reciprocal best word so a newly grown distinction
cannot be stranded. These are ordinary habituating graph edges, not a word
dictionary or record-text lookup. The same lexical node can therefore become
the meeting point of many independently grown routes, while each route can
support several competing words. Promoted experiential patterns are also
eligible for later HEAR nomination through the path rooted at their original
trunk.

Growth remains bounded per sensation: at most six basal fibers and four
within-trunk intersections, plus at most four cross-trunk intersections per
co-settled frame. Breadth therefore accumulates across lived variation rather
than by turning every record, sentence, or noun into a concept node. This pass
addresses perceptual and word-grounding breadth. Output-side skill assemblies
still mature through authorized DO/LOOK/SPEAK cycles and verified returns; a
general cross-modal abstraction learner above both halves remains a later
gate.

## Online language development

Ordinary conversation now continues that nursery process. The HEAR membrane
does not split text with a hand-written vocabulary. The local GGUF tokenizer
provides an ordered sequence of exact token-embedding rows. Each unseen row
creates an opaque `lexeme_candidate` node with no word, token ID, or term stored
on the node. One exposure is not enough to make it speakable.

Every current utterance deposits four kinds of ordinary graph evidence:

- HEAR-to-first-token and local token-to-token sensory routes;
- speech-start, token-to-token, and final-token-to-stop motor routes;
- token-specific links to the best-aligned currently active semantic branches;
- source-to-concept and source-to-token routes for learned conversational style.

Repeated exposure deepens these routes as habituation, which is distinct from
positive reward. After three independent input records contain the same token
geometry, its candidate changes into a productive `lexeme`. Weak accidental
branches remain nonproductive until that threshold. Novel sequence edges are
also archived when first observed and awaken only after three independent
records support that exact adjacency. Thus one unusual juxtaposition between
two already-known words cannot immediately alter speech. Once mature, local
branches from separately heard sequences can recombine; no complete sentence
object or phrase template is installed.

The next message no longer gives every prior response a fixed bonus. Only the
newest open speech cycle addressed to that source is considered. Its inferred
return combines numeric alignment with the prior semantic route, overlap with
the spoken lexical route, learned valence of those concepts, and confidence-
weighted preference for that source. This is deliberately a small social
nudge, not a claim that continued contact proves correctness. `/reward` remains
the stronger explicit evaluation and also updates the source relationship.
In the live runner, that inferred return is queued first and then admitted
beside the current message under one HEAR/SELF pulse; `/reward` likewise uses a
kernel return pulse instead of mutating recurrent state through a side path.

## Run locally

Build the two native GGUF helpers:

```bash
make -C experiments/graph_native_live build
```

Start the agent:

```bash
PYTHONPATH=src python3 experiments/graph_native_live/unified_open_weight_agent.py
```

The first launch grows a new accelerated mind and installs the desire nursery.
On the tested local CPU this takes roughly one minute. Later turns use the
persistent database at
`experiments/graph_native_live/open_weight_runs/unified-mind-v2.sqlite`.
Use `--initialize-only` to prepare and inspect that mind without opening a
speech cycle.

Useful controls:

- `/state` inspects pressures and activations without advancing them;
- `/tick` advances an endogenous pulse and speaks only when a drive crosses its
  expression threshold;
- `/reward 1` gives the latest speech route a verified satisfying return;
- `/reward -1` gives it a verified frustrating return;
- `/source ID` changes the active local speaker so separate source-conditioned
  habits can form;
- `/quit` closes the local process.

Every ordinary input line is a developmental exposure. Repeating one identical
line three times deliberately teaches a narrow route. Using a new word across
several different sentences is the better experiment: the shared lexical node
accumulates different semantic and transition neighbors, which gives later
competition room to recombine it. Enter `/reward VALUE` before the next ordinary
message when you want an explicit evaluation of the latest response; otherwise
the next matching-source message supplies only the smaller inferred social
return.

For an isolated one-turn run:

```bash
PYTHONPATH=src python3 experiments/graph_native_live/unified_open_weight_agent.py \
  --database /tmp/habitus-open-weight.sqlite \
  --run-directory /tmp/habitus-open-weight-runs \
  --once "What matters to you before we continue?" \
  --show-receipt
```

Add `--with-native-continuation` to probe free transformer continuation after
the graph-native sentence. It is off by default because the current Qwen3 0.6B
model often echoes or misinterprets soft lexical rows; that output is not used
as evidence of better reasoning.

## Evidence and present limit

The behavioral tests cover bounded trunk-rooted sensory growth, dormant
one-shot intersections, repeated-pattern promotion, cross-receptor convergence,
multi-context word binding without text crowns, one-pulse three-receptor
settling, inward traces to SELF, malformed-frame preflight, recurrent-state
rollback on failed final valuation, immutable restart-safe pulse receipts,
context-conditioned output value, one-use motor authorization, stale-output
rejection, idle polling without phantom output repetition, private lexical
substeps under their parent pulse, restart persistence, autonomous pressure growth,
target-free route choice, refractory handoff, exact receipt-gated satisfaction,
batched Y-path equivalence, zero record-text reads, numeric packet privacy,
composite-drive selection, opaque nodes, word-level recurrent feedback,
multi-branch lexical convergence, counterfactual next-word choice, weak lexical
candidate promotion, learned transition strengthening, restart-safe vocabulary
growth, unpresented route recombination, source-specific cycle matching, and
content/source social-return counterfactuals. A separate regression verifies
that one new adjacency between two mature lexemes remains dormant.

A 2026-09-05 local Qwen3 0.6B breadth smoke replayed one LoCoMo session (18
utterances, one NOTICE boundary, and two SEE captions) through ordinary SELF
pulses. The inherited nursery snapshot grew from 318 concepts / 1,853 edges to
644 concepts / 4,580 edges. New lower structure included 78 reusable sensory
fibers, 66 still-dormant conjunction candidates, 10 promoted experiential
patterns, and six cross-trunk candidates. There were 642 lower-route-to-lexical
associations covering 184 lexical nodes; every new lower node remained opaque
and graph invariants reported no new errors. After removing redundant
whole-edge-table scans, developmental replay took 11.82 seconds and the complete
baseline/trained diagnostic took 20.78 seconds on this machine. The single
held-out factual question still scored 0, so this establishes practical breadth
and grounding structure, not factual recall or conversational competence.

A fresh end-to-end run produced clean graph invariants and the independently
selected lexical sequence `I wonder about uncertainty.` An identical initial
database receiving a connection-oriented input instead produced `I value
connection.` With no input, the current field produced `I wonder what matters.`
None is stored as a complete output template; the initial nursery's longest
explicit motor unit is one transition between two states. Receipts expose every
candidate set, winning edge, convergent source count, recurrent state hash,
online lexical growth, and any inferred social-return components.

In a fresh live GGUF trial, eleven sensed tokens introduced nine previously
absent candidate nodes. None became productive on the first or second hearing.
On the third hearing the candidates matured and the graph produced `A zephyr
lattice can learn through patient conversation.` After a second path was
learned, a mixed prompt produced `A cobalt lattice can grow through patient
conversation.` That exact sequence had never been presented; its route existed
because independently learned local branches converged. This demonstrates
productive vocabulary growth and branch recombination, not broad grammatical
competence.

This is a functional recurrent desire-and-speech substrate, not yet a generally
capable conversational model. Online learning is currently first-order: it
grows token associations and directed local transitions, so sparse experience
can still produce awkward or over-repeated language. It does not yet discover
multi-token syntax assemblies, infer correction intent, or distinguish every
quoted sentence from a caregiver demonstration. Those should emerge from
competing higher-order trajectories and verified consequences rather than by
quietly reintroducing templates, RAG, or a rolling prompt.
