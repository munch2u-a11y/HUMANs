# Working on Habitus Mind

This file is the entry point for coding agents and human contributors. The
shippable product is `habitus-mind`, implemented by
`src/habitus_ai/integrated_agent.py`. Start with [README.md](README.md), then
read [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) and
[docs/INTEGRATED_MIND.md](docs/INTEGRATED_MIND.md).

## Safe first pass

Run these from the repository root:

```bash
python3 scripts/check_portability.py
python3 scripts/check_docs.py
make setup
make playground
make test
```

`make playground` is disposable, offline, CPU-only, and uses a tiny cortex. It
exercises conversation, explicit memory, recall, file reading, file execution,
receipts, and state persistence without Ollama or a model download.

Do not enable GPU execution, download models, run native binaries, or start the
developmental nurseries unless the user asks. The full standard test suite uses
CPU and does not need Ollama. The optional native GGUF tests skip when their
local build artifacts are absent.

## Source map

- `integrated_agent.py`: current conversation, memory, workspace, and receipt
  surface.
- `developmental_runtime.py`: authoritative graph, recurrent, cortex, and SELF
  pulse composition.
- `developmental_cortex.py`: persistent random-born cortex and checkpoints.
- `self_pulse.py`: sensory settling and one-use output authorization.
- `recurrent.py`: persistent primitive and composite desire dynamics.
- `graph.py`, `store.py`, `pipeline.py`: graph structure, canonical SQLite
  authority, and the conserved memory substrate.
- `tools.py`, `functional_agent.py`: receipt lifecycle and root-confined local
  file adapters.
- `examples/api_playground.py`: smallest editable end-to-end integration.
- `experiments/graph_native_live/`: optional research programs and ablations,
  not the default product runtime.

`habitus-rag` is the older prompt-context comparison surface. Do not silently
replace the integrated runtime with it.

## Behavioral boundaries to preserve

- Current conversation reaches the Ollama renderer only after SELF authorizes
  `SPEAK`.
- The current renderer receives no transcript, retrieved text, `identity.md`,
  or skill document.
- Old exact text enters behavior through explicit receipt-backed `/recall`, not
  automatic prompt injection.
- `/open` and `/run` require a selected one-use affordance. A narrated action is
  not execution.
- An explicit command constrains only its corresponding motor trunk to the
  currently sensed ability; a previously strong ability must not hijack it.
- Tool success requires an execution receipt and a sensory return through
  `SEE` or `NOTICE` before it is reported.
- Canonical records are evidence. Graph strength, recurrent state, neural state,
  and affect-like variables are learned interpretations, not factual authority.
- `HEAR` may preserve language-derived features. Tool returns cross the
  cognitive membrane as opaque numeric sensor embeddings while raw receipts
  remain inspectable in the developer ledger.

## Portability and local state

Never commit a developer-specific home path. Resolve repository assets from
`Path(__file__)`, accept user locations through CLI arguments or environment
variables, and use `tempfile` for disposable artifacts. Do not assume a package
manager, Ollama library directory, GPU, or shell startup layout. Run
`make portability` after changing source, configuration, examples, or docs.

The default writable locations are repo-relative and ignored by Git:

- `state/` for mind databases and cortex checkpoints;
- `workspace/` for explicitly authorized files;
- `models/` for optional local weights;
- experiment-specific `runs/` directories for generated evidence.

Do not delete or overwrite a user's database, workspace, checkpoint, model, or
experiment artifact. Tests and examples should use pytest temporary paths or
the standard-library temporary-directory APIs.

## Verification expectations

Use the smallest relevant check while developing, then run `make verify` before
shipping. For state or action changes, tests should assert authoritative
artifacts: database records, selected affordances, tool receipts, observed
returns, hashes, restart state, or final file contents. Response prose alone is
not proof of execution.

Keep claims narrow:

- deterministic tests establish coded invariants;
- the offline playground establishes wiring with a fake renderer;
- `make smoke` establishes one live Ollama-backed turn;
- developmental experiment receipts establish only their documented controlled
  result.

The historical white paper and native experiment notes describe research
lineages. If they conflict with current source, tests, or
`docs/INTEGRATED_MIND.md`, update the docs and treat the implementation plus
passing behavioral tests as the current evidence.
