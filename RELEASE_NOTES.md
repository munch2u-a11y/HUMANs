# Habitus Mind 0.1.0

Release date: 2026-09-07

This functional-alpha snapshot packages the previously separate Habitus graph,
recurrent SELF pulse, developmental cortex, explicit memory, local abilities,
and open-weight speech surface as one runnable process.

## Included

- persistent 18,016,166-parameter random-born cortex and hidden state;
- evolving primitive and composite desire pressures;
- ordinary current-event conversation through a local Ollama speech motor;
- explicit durable `/remember` and receipt-backed `/recall`;
- root-confined `/open` folder/file inspection and bounded `/run` abilities;
- machine-readable pulse, action, evidence, and tool receipts;
- the conventional `habitus-rag` path as an A/B comparison;
- full Python tests and the source for developmental/native experiments.

## Intentionally not included

- private memory databases or prior conversations;
- trained/generated cortex checkpoints;
- local model weights, compiled native helpers, or virtual environments;
- generated nursery and benchmark runs.

The repository therefore starts as a reproducible source release rather than a
copy of one already-lived mind. Each user creates and owns a distinct local
lineage on first launch.

## Repository hardening in the current revision

- `make playground` now exercises the complete integrated loop offline with a
  disposable tiny cortex and editable speech motor;
- `habitus-doctor` checks the live local-model prerequisites without creating
  state or enabling a GPU;
- `make verify` combines portability, documentation, playground, and full CPU
  checks;
- `AGENTS.md` and the getting-started/testing guides provide a clean handoff to
  another human or coding agent;
- CI runs the same path and documentation checks before behavioral tests;
- GitHub Actions uses the Node 24-based `checkout@v6` and `setup-python@v6`
  runtimes; and
- hard-coded user homes, temporary directories, and native-library install
  prefixes have been removed from runtime defaults.

The live acceptance pass also added bounded folder listing through `/open`,
made repeated identical affordances unique across pulses, tightened explicit
recall to exclude command/result echoes, declared NumPy as a cortex dependency,
and selected `qwen3.5:2b` as the default speech motor after a direct local
0.8B/2B response comparison.
