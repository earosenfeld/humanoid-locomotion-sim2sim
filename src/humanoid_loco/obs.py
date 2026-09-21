"""Observation terms as pure numpy, mirroring ``isaaclab.envs.mdp`` semantics.

Each term is a function of the robot state, the velocity command and the previous action.
The manifest names which terms appear and in what order; ``assemble_obs`` does the rest.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from humanoid_loco.manifest import PolicyManifest


@dataclass
class RobotState:
    """Minimal proprioception. Joint arrays are in manifest (policy) order."""

    base_quat: np.ndarray  # (4,) w, x, y, z  world->base
    base_ang_vel: np.ndarray  # (3,) rad/s, base frame (IMU gyro)
    joint_pos: np.ndarray  # (n,)
    joint_vel: np.ndarray  # (n,)
    base_lin_vel: np.ndarray | None = None  # (3,) base frame; privileged, critic-only


def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate world-frame ``v`` into the frame described by unit quaternion ``q`` (w,x,y,z)."""
    w, xyz = q[0], q[1:]
    return v * (2.0 * w * w - 1.0) - 2.0 * w * np.cross(xyz, v) + 2.0 * xyz * np.dot(xyz, v)


def projected_gravity(base_quat: np.ndarray) -> np.ndarray:
    return quat_rotate_inverse(base_quat, np.array([0.0, 0.0, -1.0]))


ObsFn = Callable[[RobotState, np.ndarray, np.ndarray, PolicyManifest], np.ndarray]

TERMS: dict[str, ObsFn] = {
    "base_lin_vel": lambda s, c, a, m: s.base_lin_vel,
    "base_ang_vel": lambda s, c, a, m: s.base_ang_vel,
    "projected_gravity": lambda s, c, a, m: projected_gravity(s.base_quat),
    "velocity_commands": lambda s, c, a, m: c,
    "joint_pos": lambda s, c, a, m: s.joint_pos - np.asarray(m.default_joint_pos),
    "joint_vel": lambda s, c, a, m: s.joint_vel,
    "actions": lambda s, c, a, m: a,
}


def assemble_obs(
    manifest: PolicyManifest,
    state: RobotState,
    command: np.ndarray,
    last_action: np.ndarray,
    group: list | None = None,
) -> np.ndarray:
    """Concatenate manifest terms (or an explicit term list) with their scale and clip."""
    parts = []
    for term in group if group is not None else manifest.obs_terms:
        try:
            value = np.asarray(TERMS[term.name](state, command, last_action, manifest), np.float32)
        except KeyError as e:
            raise KeyError(f"unknown observation term {term.name!r}") from e
        if value.size != term.dim:
            raise ValueError(f"{term.name}: got {value.size} values, manifest says {term.dim}")
        value = value.ravel() * term.scale
        if term.clip is not None:
            value = np.clip(value, *term.clip)
        parts.append(value)
    return np.concatenate(parts)
