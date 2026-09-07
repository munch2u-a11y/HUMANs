# Getting started

This guide takes a clean checkout through an offline integrated exercise and an
optional live local-model conversation. Run commands from the repository root.

## 1. Clone and install

Requirements are Python 3.11 or newer and a POSIX system such as Linux or
macOS. The live speech path additionally requires a running Ollama service.

```bash
git clone https://github.com/munch2u-a11y/HUMANs.git
cd HUMANs
make setup
```

`make setup` creates `.venv` inside the checkout and installs the cortex and
test dependencies. It does not create a mind, use a GPU, or download an Ollama
model.

## 2. Prove the integration offline

```bash
make playground
```

The playground creates a temporary tiny cortex and disposable workspace. It
exercises an ordinary conversational event, `/remember`, `/recall`, `/open`,
and `/run`, prints machine-readable receipts, then removes its temporary state.
It uses a deliberately simple editable speech renderer, so no network, Ollama,
GPU, or model download is involved.

For complete turn envelopes rather than the compact receipt summary:

```bash
.venv/bin/python examples/api_playground.py --full
```

Run the complete CPU verification suite when you are ready:

```bash
make test
```

See [Testing](TESTING.md) for focused checks and evidence boundaries.

## 3. Start the live speech motor

Install and start Ollama using its platform instructions, then pull the default
small model:

```bash
ollama pull qwen3.5:0.8b
make doctor
```

The doctor is read-only. It checks Python, platform support, the installed
package, PyTorch, the configured Ollama endpoint, and the configured speech
model. It never creates a mind or enables the GPU.

Exercise one live turn and inspect its JSON receipt:

```bash
make smoke
```

Then enter the interactive runtime:

```bash
make run HUMAN_NAME="Your name" AGENT_NAME="Mira"
```

The initial identity and taste are stored only when the database is new. If you
reuse `state/habitus.sqlite`, its existing lineage wins over later name flags.
To create another lineage, select another database:

```bash
make run MIND_DATABASE=state/second-mind.sqlite AGENT_NAME="Second"
```

All product defaults are repository-relative. You may also choose paths outside
the checkout explicitly:

```bash
make run \
  MIND_DATABASE="path/to/local-state/mind.sqlite" \
  WORKSPACE="path/to/authorized-workspace"
```

The database parent is created by the CLI. The workspace itself must exist;
the Make target creates it when needed.

## 4. Try the working abilities

Copy the harmless example into the authorized workspace:

```bash
cp examples/workspace/hello.py workspace/hello.py
```

At the prompt, try:

```text
Hello. What should I call you?
remember that my launch color is ultraviolet
/recall launch color
/open hello.py
/run hello.py
/state
```

`/open` accepts one UTF-8 file inside the authorized workspace. `/run` accepts
one Python file there and applies wall-time, CPU-time, memory, output, file-size,
and descriptor limits. Those limits are not a hostile-code sandbox: run only
code you trust, or add an operating-system sandbox.

## 5. Give the repository to another coding agent

This prompt is enough to establish the right entry point and safety boundary:

```text
Read AGENTS.md, README.md, and docs/GETTING_STARTED.md. Run the portability and
documentation checks, then the offline playground and relevant CPU tests. Do
not enable a GPU, download a model, run a developmental nursery, or delete local
state unless I ask. Inspect source and test receipts before making claims. Tell
me what you verified and propose one bounded change.
```

The smallest editable API example is
[`examples/api_playground.py`](../examples/api_playground.py). Replace its
`EditableSpeechMotor` with your agent or local model adapter while keeping the
`IntegratedMind` and `BornInHabitusRuntime` boundaries intact.

## Troubleshooting

- `make: ... .venv ... not found`: run `make setup` first.
- PyTorch import fails: rerun `.venv/bin/python -m pip install -e '.[cortex]'`.
- The doctor cannot reach Ollama: start the local service or override
  `OLLAMA_URL` with an endpoint you control.
- The model is missing: run `ollama pull MODEL_NAME`, then set the same
  `MODEL=MODEL_NAME` for `make doctor`, `make smoke`, and `make run`.
- An old mind reports a different name or model profile: choose a new database
  path or deliberately continue that existing lineage.
- A native GGUF test skips: this is expected unless the optional native
  artifacts described in the
  [experiment README](../experiments/graph_native_live/README.md) are built.
