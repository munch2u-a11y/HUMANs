# Contributing to Habitus Mind

Habitus Mind welcomes implementation, testing, documentation, and controlled
experiment contributions. Keep claims tied to observable behavior and preserve
the distinction between narrated action and verified execution.

## Local setup

```bash
git clone YOUR_FORK_URL habitus-mind
cd habitus-mind
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test,cortex]'
.venv/bin/python -m pytest
```

Native GGUF experiments have additional llama.cpp requirements documented in
[`experiments/graph_native_live/README.md`](../experiments/graph_native_live/README.md).
They are not required for the integrated Ollama runtime.

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
