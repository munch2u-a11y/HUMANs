# Contributing to Habitus Mind

Habitus Mind welcomes implementation, testing, documentation, and controlled
experiment contributions. Keep claims tied to observable behavior and preserve
the distinction between narrated action and verified execution.

## Local setup

```bash
git clone YOUR_FORK_URL HUMANs
cd HUMANs
make setup
make playground
make test
```

Before opening a pull request, run the full local release gate:

```bash
make verify
```

The standard gate is CPU-only and does not require Ollama. Run `make doctor`
and `make smoke` separately when your change touches the real local-model
boundary.

Native GGUF experiments have additional llama.cpp requirements documented in
[`experiments/graph_native_live/README.md`](../experiments/graph_native_live/README.md).
They are not required for the integrated Ollama runtime.

## Portability

Do not commit a contributor-specific home path, package-manager prefix, or
temporary-directory assumption. Resolve packaged assets from the source file,
accept user-selected locations through CLI/env inputs, and use platform temp
APIs for disposable output. `make portability` enforces the common cases.

## Change requirements

- Add behavioral tests for changed behavior, including restart or receipt
  assertions when persistence or actions are involved.
- Preserve canonical records as evidence authority and keep derived graph or
  neural state distinguishable from those records.
- Do not count model narration as tool execution. State-changing actions need a
  receipt and an observed return.
- Keep language admission causal: ordinary word-derived features enter through
  `HEAR`; non-language `SEE` and `NOTICE` payloads stay opaque unless a design
  explicitly changes and tests that boundary.
- Run the complete suite before opening a pull request.

## Pull requests

Explain the user-visible change, the state or causal boundary it touches, and
the verification you ran. Generated databases, checkpoints, model weights,
virtual environments, binaries, and experiment run directories must not be
committed.

Contributions are licensed under the [Apache License 2.0](../LICENSE).
