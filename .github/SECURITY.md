# Security policy

## Supported versions

| Version | Supported |
| --- | --- |
| 0.1.x | Yes |

## Local execution boundary

Habitus Mind is local-first and does not expose a network service by default.
Its `--workspace` setting confines which folder or file a user can select with
`/open` and which file can be selected with `/run`. `/run` applies CPU-time,
address-space, file-descriptor, file-size,
wall-time, and output limits.

Those controls are damage limits, not a hostile-code sandbox. An authorized
Python file still runs as the current operating-system user and may access
resources available to that account. Do not point Habitus at untrusted code.
Use an operating-system sandbox or disposable container when code provenance is
uncertain.

The Ollama endpoint is configurable. Treat any non-loopback endpoint as an
external data recipient: the current conversational event is sent to that
endpoint for surface rendering.

## Reporting a vulnerability

Please use a private security advisory on the repository host. Include the
affected version, reproducible steps, impact, and any suggested mitigation. Do
not post secrets, personal memory databases, model checkpoints, or exploit
payloads in a public issue.
