# Documentation map

The documentation is split by purpose so a new user does not have to infer
which experimental lineage is the current application.

## Start here

- [Project README](../README.md): what ships and the shortest run path.
- [Getting started](GETTING_STARTED.md): setup, offline exploration, live
  launch, persistence, and troubleshooting.
- [Testing](TESTING.md): verification commands and what each one proves.
- [Integrated mind runtime](INTEGRATED_MIND.md): exact language, memory,
  identity, sensing, desire, and action boundaries.
- [Architecture contract](../ARCHITECTURE.md): invariants shared by the graph,
  evidence store, and integrated runtime.

## Development and extension

- [Developer guide](../DEVELOPMENT.md): runtime flow, module ownership, and
  safe extension points.
- [Agent guide](../AGENTS.md): concise repository instructions for a coding
  agent.
- [Contributing](../.github/CONTRIBUTING.md): contribution and pull-request
  expectations.
- [Security](../.github/SECURITY.md): local execution and remote-endpoint
  boundaries.

## Research lineages

- [Born-in cortex technical brief](HABITUS_BORN_IN_CORTEX_TECHNICAL_BRIEF.md):
  current subsystem implementation and measured developmental evidence.
- [Developmental cortex](DEVELOPMENTAL_CORTEX.md): cortex configuration,
  controlled nurseries, and optional ROCm setup.
- [Unified open-weight runtime](UNIFIED_OPEN_WEIGHT_RUNTIME.md): optional native
  GGUF continuous-input research path; not required by `habitus-mind`.
- [Historical white paper](../WHITEPAPER.md): the pre-0.1 conserved substrate
  and archived experiment snapshot.
- [Six-lane experiment](../EXPERIMENT.md): causal lane and membrane subsystem.
- [Native experiment README](../experiments/graph_native_live/README.md):
  specialized experiment commands and dependencies.

## Release records

- [Release notes](../RELEASE_NOTES.md)
- [Citation metadata](../CITATION.cff)

Generated databases, checkpoints, model weights, native libraries, and
experiment outputs are intentionally not shipped. Commands in the current
guides use repository-relative paths or explicit user-selected paths.
