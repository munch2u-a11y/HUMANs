# Born-in Habitus developmental cortex

For a subsystem-level explanation, evidence matrix, and candid audit of the
current integrated model, see
[`HABITUS_BORN_IN_CORTEX_TECHNICAL_BRIEF.md`](HABITUS_BORN_IN_CORTEX_TECHNICAL_BRIEF.md).

This extension is the from-scratch alternative to placing a corporate-trained
GGUF beside Habitus. It is an experimental substrate, not yet a mature
conversational model. Its neural weights begin from a recorded random seed and
can change only through admitted local experience.

## Unified runtime

```text
current NOTICE / SEE numeric fields + current HEAR UTF-8 bytes
                              |
                              v
             trunk-rooted experiential growth
                              |
                              v
               recurrent SELF pulse commit
                  | graph field | byte state
                  v             v
       persistent Habitus state + persistent GRU cortex
                              |
                 +------------+------------+
                 |                         |
          byte production          consequence / valence /
                                   DO-LOOK-SPEAK heads
                 |                         |
                 +------ authorized action+
                                      |
                         observed sensory return
                                      |
                         receipt-backed plasticity
```

`BornInHabitusRuntime` is the integration boundary. Every input first becomes a
canonical record and grows from its actual input trunk. The same SELF
transaction commits recurrent graph state, output authorizations, curriculum
admissions, byte-form growth, and the cortex hidden state. A failure in a later
extension rolls that whole SELF commit back.

The cortex receives only:

- bytes from HEAR records admitted in the current pulse;
- a fixed-width numeric field derived from the current graph and recurrent
  state;
- a direction receptor (`hear`, `speak`, or nonverbal settling);
- its previously persisted hidden tensor when that tensor belongs to the
  current model hash.

It does not read a transcript, retrieval packet, prompt, tokenizer, vocabulary
file, pretrained embedding model, or GGUF. After consolidation changes the
coordinate system, the next pulse wakes from the durable graph with a zeroed
neural hidden state instead of replaying text through an incompatible state.

The default cortex has 18,015,141 trainable parameters: byte and boundary
receptors, a 3-layer 1024-wide GRU, and byte, route, consequence, valence, and
motor heads. `--tiny` replaces it with a 78,893-parameter configuration for
fast structural tests. Developmental training uses FP32 on both CPU and ROCm.
The full model is small enough for that stable policy on the 16 GiB APU, and a
non-finite loss, gradient, loaded parameter, or post-update parameter aborts
before any plasticity receipt is written.

## Developmental controls

The curriculum advances one measured stage at a time:

1. `prelinguistic`: numeric sensory distinctions and verified consequences;
2. `grounded_forms`: recurring raw byte spans grounded in lived routes;
3. `functional_exchange`: grounded language-action behavior;
4. `experienced_narrative`: sequences referring to routes already lived;
5. `broader_narrative`: external stories after the earlier gates hold.

Early stages also add bounded target-free motor variation. A deterministic
seed/pulse draw occasionally chooses the least-observed route among the motor
fibers that SELF actually authorized. Familiarity pressure and this selection
rate both anneal as milestones advance; they reach zero in the broader
narrative stage. The draw, transient pull, route probability, travel cost, and
selection mode remain in the ordinary SELF/output-cycle receipts. No correct
action is supplied by the explorer, so only later verified consequences can
turn variation into preference.

Language ahead of its stage is retained as canonical experience but rejected
as training. No list of target words exists. Repeated 2-16 byte spans first
remain evidence-only, then become opaque candidates, and promote only when
independent records provide compression gain and route specificity above a
population-derived shuffled control. Surface bytes are absent from concept
nodes. Promoted spans remain recoverable for audit by slicing their immutable
source records; a span that also covered a complete heard utterance may be
copied once into an immutable, hash-checked motor engram so the agent can
execute it without runtime record retrieval.

Behavioral labels have a stricter boundary. A training episode cannot merely
set `verified=True`. It must reference:

- an input admitted by the curriculum;
- an output cycle authorized by that exact SELF pulse;
- the canonical output record;
- a verified terminal return and its canonical record;
- a consequence equal to the return receipt's observed stability change.

Tampered, open, unrelated, missing, and unverified cycles are rejected before
an optimizer step. Each successful update emits an immutable SQLite plasticity
receipt plus a hashed checkpoint owned by this cortex lineage.

## Run the evidence nursery

The fast CPU run is useful before touching GPU configuration:

```bash
mkdir -p state/experiments/cortex-checkpoints
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python \
  experiments/graph_native_live/developmental_cortex_nursery.py \
  --device cpu --tiny --cycles 4 --optimization-steps 160 \
  --database state/experiments/cortex.sqlite \
  --checkpoints state/experiments/cortex-checkpoints
```

This procedural nursery separates training and held-out sensor variations. It
reports graph discrimination and cortex consequence prediction separately. It
then exposes arbitrary byte carriers across varied grounded frames and reports
the forms that actually pass promotion. Motor acts are selected from the
current SELF affordances, persisted before their consequences, and trained only
after those returns settle on the next pulse.

The current full tiny-model run cleared the first two gates with 1.0 held-out
sensory discrimination, 1.0 held-out consequence-sign accuracy, 16 closed
action cycles spanning all three motor trunks, 24 verified behavior examples,
60 promoted recurrent byte spans, and no graph invariant errors. Eight actions
were exploratory and eight were top-ranked SELF selections. On held-out byte
sequences, top-1 prediction was 0.15 and top-5 was 0.625. That is evidence of
early byte learning, not fluent speech; these remain controlled nursery
results, not evidence of general intelligence.

The next communication nursery closes one more loop:

```bash
make -C experiments/graph_native_live communication-nursery
```

Across three tiny CPU seeds, cold generation emitted no accepted messages in
0/9 trials. After grounded exposure, low-weight motor rehearsal, receipt-backed
SPEAK training, and a checkpoint restart, all 18 held-out `SEE`-only trials made
SPEAK dominant and caused the correct listener action. Graph-only and coupled
selection were 18/18; cortex-only sequence scoring was 13/18. Each output has a
motor-selection event, one internal event per byte, an accepted stop event, a
persisted cortex output state, an outbound record, and a verified listener
return. This is a learned referential convention, not free-form conversation.

## Radeon 780M / ROCm

This optional section records the exact ROCm environment used for the reported
Radeon 780M (`gfx1103`) probe. The setup script intentionally pins that
reproduction environment; it is not part of `make setup` and should not be
treated as a universal or automatically current AMD installer. Check the linked
AMD compatibility material before changing the pin or a host system.

The pinned environment is installed with:

```bash
experiments/graph_native_live/setup_rocm_cortex.sh
```

Equivalent package command:

```bash
.venv-rocm/bin/python -m pip install \
  --index-url https://repo.amd.com/rocm/whl-multi-arch/ \
  'torch[device-gfx1103]==2.12.0+rocm7.14.1'
```

Then run a real forward, backward, and optimizer step:

```bash
make -C experiments/graph_native_live cortex-rocm-probe
```

The probe defaults to the same FP32 precision as developmental training. It
fails rather than silently falling back to CPU, verifies that weights changed,
checks finite loss and parameters, and caps peak reserved GPU memory at 6 GiB
so the 16 GiB shared-memory machine keeps headroom for the desktop and graph
store. `--precision float16` remains available as an explicit hardware stress
and rejection probe; it is not a persistent-learning policy.

On the originally audited machine the wheel reported PyTorch
`2.12.0+rocm7.14.1` and HIP `7.14.60850`. The account running the probe must
belong to both GPU-access groups:

```bash
sudo usermod -aG render,video "$USER"
```

An already-running shell can activate the new membership with `sg render -c`;
a fresh login inherits it directly. The default 18,015,141-parameter FP32 probe
passes on the Radeon 780M with a real forward/backward/Adam step, finite weights,
and a 420 MiB peak reservation. A bounded full-size developmental run also
passed the pulse, receipt, optimizer, checkpoint, and graph-invariant path. The
160-step tiny nursery reached `functional_exchange` in FP32 with finite held-out
predictions. In contrast, the explicit FP16 probe has a finite first loss but
rejects its first optimizer update because parameters become non-finite; the
earlier FP16 nursery likewise diverged. Persistent learning is therefore full
precision by measured necessity, not merely a conservative default.

Do not run the repository setup script with sudo. Also note that AMD's Ryzen
APU instructions document specific Ubuntu kernel combinations; the currently
installed Ubuntu 24.04 kernel 7.0 combination is not listed. Change group
membership and rerun the strict probe first. Consider a kernel change only if
the device remains unavailable, and follow AMD's current installation guide:

- <https://rocm.docs.amd.com/en/docs-7.14.1/compatibility/compatibility-matrix.html>
- <https://rocm.docs.amd.com/en/docs-7.14.1/install/rocm.html>
- <https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html>

## Current evidence boundary

Implemented and directly tested:

- random-only model lineage and checkpoint hash enforcement;
- persistent neural state without transcript replay;
- atomic coupling to the recurrent SELF pulse;
- trunk-rooted and cross-trunk sensory growth;
- milestone-gated raw-byte form emergence;
- receipt-backed action/consequence learning;
- bounded cortical proposals that causally alter SELF output ranking;
- learned-stop raw generation and hash-checked whole-form motor execution;
- causal held-out communication through an independently learned listener;
- deterministic CPU nursery and restart tests;
- a strict, memory-bounded ROCm execution probe.

Not yet established:

- useful free-form or compositional byte speech;
- language-driven response from recognized HEAR forms;
- broad story comprehension;
- mature autonomous tool use;
- equivalence to an LLM trained on a large language corpus.
