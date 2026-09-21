"""Policy inference latency per execution provider, batch 1, as the control loop would call it.

    python deploy/thor/bench.py --run <export dir> [--iters 10000] [--out timing.json]

Runs every provider onnxruntime exposes on this machine (CPU, CUDA, TensorRT on a Jetson with the
GPU wheel) and reports p50/p99/max microseconds per call. Also covers x86 for the comparison table.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

PROVIDERS = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]


def bench(onnx: Path, obs_dim: int, provider: str, iters: int) -> dict:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(onnx), opts, providers=[provider])
    name = sess.get_inputs()[0].name
    x = np.random.default_rng(0).standard_normal((1, obs_dim)).astype(np.float32)
    for _ in range(200):  # warm-up (TensorRT builds its engine here)
        sess.run(None, {name: x})
    t = np.empty(iters)
    for i in range(iters):
        t0 = time.perf_counter_ns()
        sess.run(None, {name: x})
        t[i] = (time.perf_counter_ns() - t0) / 1e3
    return {
        "p50": float(np.percentile(t, 50)),
        "p99": float(np.percentile(t, 99)),
        "max": float(t.max()),
        "n": iters,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--iters", type=int, default=10000)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    manifest = json.loads((a.run / "policy.json").read_text())
    obs_dim = sum(t["dim"] for t in manifest["obs_terms"])
    available = ort.get_available_providers()
    results = {
        "host": {"machine": platform.machine(), "node": platform.node(), "ort": ort.__version__},
        "providers": {
            prov: bench(a.run / manifest["onnx"], obs_dim, prov, a.iters)
            for prov in PROVIDERS
            if prov in available
        },
    }
    print(json.dumps(results, indent=2))
    if a.out:
        a.out.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
