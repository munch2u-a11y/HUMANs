#!/usr/bin/env python3
"""Prove that the born-in recurrent cortex can really train on the AMD iGPU."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from habitus_ai.developmental_cortex import (  # noqa: E402
    BornInRecurrentCortex,
    CortexConfig,
    require_torch,
    rocm_runtime_report,
)


def configuration(tiny: bool) -> CortexConfig:
    if not tiny:
        return CortexConfig()
    return CortexConfig(
        byte_embedding_width=32,
        graph_width=32,
        direction_width=8,
        hidden_width=64,
        recurrent_layers=2,
        maximum_event_bytes=32,
    )


def run_probe(
    *,
    allow_cpu: bool,
    tiny: bool,
    sequence_bytes: int,
    precision: str = "float32",
) -> dict[str, object]:
    require_torch()
    import torch
    import torch.nn.functional as F

    runtime = dict(rocm_runtime_report())
    rocm = bool(torch.cuda.is_available() and torch.version.hip is not None)
    if not rocm and not allow_cpu:
        raise RuntimeError("ROCm PyTorch does not expose the AMD GPU")
    device = torch.device("cuda" if rocm else "cpu")
    if precision not in {"float32", "float16"}:
        raise ValueError("precision must be float32 or float16")
    dtype = torch.float16 if rocm and precision == "float16" else torch.float32
    config = configuration(tiny)
    torch.manual_seed(config.seed)
    if rocm:
        torch.cuda.manual_seed_all(config.seed)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model = BornInRecurrentCortex(config).to(device=device, dtype=dtype)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    length = max(2, min(int(sequence_bytes), config.maximum_event_bytes))
    previous = torch.randint(0, 256, (1, length), device=device)
    previous[:, 0] = -1
    target = torch.randint(0, 256, (1, length), device=device)
    field = torch.randn((1, config.graph_width), device=device, dtype=dtype)
    field = field / field.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    direction = torch.zeros((1,), device=device, dtype=torch.long)
    before = next(model.parameters()).detach().clone()
    started = time.perf_counter()
    outputs, _ = model(previous, field, direction)
    loss = (
        F.cross_entropy(outputs["byte_logits"].reshape(-1, 256), target.reshape(-1))
        + 0.4 * F.mse_loss(outputs["route"], field)
        + 0.2 * outputs["consequence"].square().mean()
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), config.gradient_clip, error_if_nonfinite=True
    )
    optimizer.step()
    if rocm:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    changed = not torch.equal(before, next(model.parameters()).detach())
    finite = math.isfinite(float(loss.detach().to(torch.float32).cpu()))
    nonfinite_parameters = [
        name
        for name, parameter in model.named_parameters()
        if not bool(torch.isfinite(parameter).all().item())
    ]
    finite_parameters = not nonfinite_parameters
    peak_allocated = torch.cuda.max_memory_allocated() if rocm else 0
    peak_reserved = torch.cuda.max_memory_reserved() if rocm else 0
    maximum_working_set = 6 * 1024**3
    return {
        "schema": "habitus.rocm-cortex-probe.v1",
        "passed": bool(
            changed
            and finite
            and finite_parameters
            and (not rocm or peak_reserved <= maximum_working_set)
        ),
        "real_backward_pass": True,
        "optimizer_step_changed_weights": changed,
        "finite_loss": finite,
        "finite_parameters": finite_parameters,
        "nonfinite_parameters": nonfinite_parameters,
        "loss": float(loss.detach().to(torch.float32).cpu()),
        "elapsed_seconds": elapsed,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "configuration": "tiny" if tiny else "planned_18m_class",
        "sequence_bytes": length,
        "compute_dtype": str(dtype),
        "peak_allocated_bytes": int(peak_allocated),
        "peak_reserved_bytes": int(peak_reserved),
        "working_set_limit_bytes": maximum_working_set,
        "runtime": runtime,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--sequence-bytes", type=int, default=32)
    parser.add_argument(
        "--precision", choices=("float32", "float16"), default="float32"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_probe(
        allow_cpu=args.allow_cpu,
        tiny=args.tiny,
        sequence_bytes=args.sequence_bytes,
        precision=args.precision,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
