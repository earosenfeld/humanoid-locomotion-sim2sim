"""Isaac-free fixtures: a G1 manifest with Isaac Lab-like numbers and a synthetic ONNX policy.

Used by the tests and by the CI smoke run so the MuJoCo/ONNX layers are exercised end-to-end
without an Isaac Sim install. Numbers are representative, not the trained policy's.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from humanoid_loco.manifest import HeldJoint, ObsTerm, PolicyManifest

# Isaac Lab orders joints breadth-first through the USD tree, which interleaves left/right.
# Keeping that order here makes the tests prove the name-based remap into MJCF order.
LEG_JOINTS = [
    f"{side}_{j}_joint"
    for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
    for side in ("left", "right")
]
_GAINS = {  # joint stem -> (kp, kd, torque limit, default pos, saturation effort, vel limit)
    "hip_pitch": (100.0, 2.5, 88.0, -0.10, 180.0, 32.0),
    "hip_roll": (100.0, 2.5, 88.0, 0.0, 180.0, 32.0),
    "hip_yaw": (100.0, 2.5, 88.0, 0.0, 180.0, 32.0),
    "knee": (200.0, 5.0, 139.0, 0.30, 180.0, 20.0),
    "ankle_pitch": (20.0, 0.2, 50.0, -0.20, 80.0, 37.0),
    "ankle_roll": (20.0, 0.1, 50.0, 0.0, 80.0, 37.0),
}
UPPER_BODY = ["waist_yaw", "waist_roll", "waist_pitch"] + [
    f"{s}_{j}"
    for s in ("left", "right")
    for j in (
        "shoulder_pitch",
        "shoulder_roll",
        "shoulder_yaw",
        "elbow",
        "wrist_roll",
        "wrist_pitch",
        "wrist_yaw",
    )
]


def g1_manifest(onnx_path: str = "policy.onnx") -> PolicyManifest:
    stems = [n.split("_", 1)[1].removesuffix("_joint") for n in LEG_JOINTS]
    col = lambda i: [_GAINS[s][i] for s in stems]  # noqa: E731
    n = len(LEG_JOINTS)
    return PolicyManifest(
        robot="unitree_g1",
        joint_names=LEG_JOINTS,
        default_joint_pos=col(3),
        kp=col(0),
        kd=col(1),
        torque_limit=col(2),
        action_scale=[0.5] * n,
        saturation_effort=col(4),
        velocity_limit=col(5),
        armature=[0.03] * n,
        joint_friction=[0.0] * n,
        obs_terms=[
            ObsTerm("base_ang_vel", 3),
            ObsTerm("projected_gravity", 3),
            ObsTerm("velocity_commands", 3),
            ObsTerm("joint_pos", n),
            ObsTerm("joint_vel", n),
            ObsTerm("actions", n),
        ],
        physics_dt=0.005,
        decimation=4,
        held_joints={f"{j}_joint": HeldJoint(0.0, 40.0, 10.0) for j in UPPER_BODY},
        onnx=onnx_path,
        source={"note": "synthetic test fixture"},
    )


def write_synthetic_policy(path: str | Path, manifest: PolicyManifest, gain: float = 0.0) -> Path:
    """Linear policy ``actions = gain * obs @ W`` with fixed W; gain=0 holds the default pose."""
    rng = np.random.default_rng(0)
    w = numpy_helper.from_array(
        (gain * rng.standard_normal((manifest.obs_dim, manifest.num_actions))).astype(np.float32),
        "W",
    )
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["obs", "W"], ["actions"])],
        "synthetic",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, manifest.obs_dim])],
        [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, manifest.num_actions])],
        initializer=[w],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9
    onnx.save(model, str(path))
    return Path(path)
