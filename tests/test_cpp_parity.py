"""The C++ loop must reproduce the Python plant exactly: same manifest, same model, same policy.

Skips unless the binary exists (``build/hls_loop`` or ``$HLS_LOOP``).
"""

import json
import os
import subprocess
from pathlib import Path

import mujoco
import numpy as np
import pytest

from humanoid_loco.mujoco.metrics import evaluate
from humanoid_loco.mujoco.rollout import Schedule, Trajectory, rollout
from humanoid_loco.mujoco.scene import build_model
from humanoid_loco.policy import OnnxPolicy

BINARY = Path(os.environ.get("HLS_LOOP", "build/hls_loop"))
pytestmark = pytest.mark.skipif(not BINARY.exists(), reason="cpp/hls_loop not built")


def test_cpp_loop_matches_python_bit_for_bit(env, manifest, run_dir, menagerie):
    schedule = Schedule.from_yaml("configs/commands/walk.yaml")
    mujoco.mj_saveModel(
        build_model(manifest, menagerie, env.control_dt), str(run_dir / "scene.mjb")
    )
    (run_dir / "schedule.json").write_text(json.dumps(schedule.to_dict()))
    out = run_dir / "cpp.json"
    subprocess.run(
        [
            BINARY,
            "--run",
            run_dir,
            "--schedule",
            run_dir / "schedule.json",
            "--no-realtime",
            "--out",
            out,
            "--timing",
            run_dir / "timing.json",
        ],
        check=True,
        capture_output=True,
    )
    cpp = Trajectory.from_dict(json.loads(out.read_text()))
    policy = OnnxPolicy(manifest, run_dir / "policy.onnx")
    py = rollout(env, policy, schedule, seed=0, init_noise=0.0)

    assert cpp.fell == py.fell and len(cpp.time) == len(py.time)
    for key, a in py.arrays().items():
        np.testing.assert_array_equal(np.asarray(cpp.arrays()[key]), a, err_msg=key)
    assert evaluate(cpp, manifest) == evaluate(py, manifest)
    timing = json.loads((run_dir / "timing.json").read_text())
    assert timing["compute_us"]["p50"] < 1000  # a tick must fit the 1 ms budget
