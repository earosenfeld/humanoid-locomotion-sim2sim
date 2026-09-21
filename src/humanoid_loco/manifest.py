"""Policy manifest: the single source of truth shared by Isaac Lab export, MuJoCo and C++.

Everything a deployment target needs to run ``policy.onnx`` correctly lives here, in the
order the policy expects it. Nothing in the MuJoCo or C++ layers restates these numbers.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ObsTerm:
    """One observation block, in policy order. ``dim`` is the flattened width."""

    name: str
    dim: int
    scale: float = 1.0
    clip: tuple[float, float] | None = None


@dataclass(frozen=True)
class HeldJoint:
    """A joint the policy does not command; deployment targets PD-hold it at ``pos``."""

    pos: float
    kp: float
    kd: float


@dataclass
class PolicyManifest:
    robot: str
    joint_names: list[str]  # policy order: actions and joint observations use this order
    default_joint_pos: list[float]
    kp: list[float]
    kd: list[float]
    torque_limit: list[float]
    action_scale: list[float]
    obs_terms: list[ObsTerm]
    physics_dt: float
    decimation: int
    held_joints: dict[str, HeldJoint] = field(default_factory=dict)
    action_clip: float | None = None  # rsl_rl ``clip_actions``; None = no clipping
    # Isaac Lab DCMotor model: available torque shrinks linearly with joint speed.
    saturation_effort: list[float] | None = None
    velocity_limit: list[float] | None = None
    # Training-asset joint dynamics the deployment plant must reproduce (sim-to-sim only).
    armature: list[float] | None = None  # reflected rotor inertia [kg m^2]
    joint_friction: list[float] | None = None  # dry friction torque [N m]
    onnx: str = "policy.onnx"
    source: dict = field(default_factory=dict)  # provenance: task id, checkpoint, git sha

    # ---- derived -------------------------------------------------------------------
    @property
    def num_actions(self) -> int:
        return len(self.joint_names)

    @property
    def obs_dim(self) -> int:
        return sum(t.dim for t in self.obs_terms)

    @property
    def policy_dt(self) -> float:
        return self.physics_dt * self.decimation

    def obs_slices(self) -> dict[str, slice]:
        out, start = {}, 0
        for t in self.obs_terms:
            out[t.name] = slice(start, start + t.dim)
            start += t.dim
        return out

    def joint_targets(self, action: np.ndarray) -> np.ndarray:
        """Map a raw policy action to PD position targets (Isaac Lab ``JointPositionAction``)."""
        return np.asarray(self.default_joint_pos) + np.asarray(self.action_scale) * action

    def torque_bounds(self, joint_vel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per-joint (min, max) torque at the current speed (Isaac Lab ``DCMotor`` clip)."""
        limit = np.asarray(self.torque_limit)
        if self.saturation_effort is None:
            return -limit, limit
        sat, vmax = np.asarray(self.saturation_effort), np.asarray(self.velocity_limit)
        hi = np.clip(sat * (1.0 - joint_vel / vmax), 0.0, limit)
        lo = np.clip(-sat * (1.0 + joint_vel / vmax), -limit, 0.0)
        return lo, hi

    # ---- io ------------------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")
        return path

    @classmethod
    def load(cls, path: str | Path) -> PolicyManifest:
        raw = json.loads(Path(path).read_text())
        raw["obs_terms"] = [
            ObsTerm(**{**t, "clip": tuple(t["clip"]) if t.get("clip") else None})
            for t in raw["obs_terms"]
        ]
        raw["held_joints"] = {k: HeldJoint(**v) for k, v in raw.get("held_joints", {}).items()}
        m = cls(**raw)
        m.validate()
        return m

    def validate(self) -> None:
        n = self.num_actions
        for name in (
            "default_joint_pos",
            "kp",
            "kd",
            "torque_limit",
            "action_scale",
            "saturation_effort",
            "velocity_limit",
        ):
            if getattr(self, name) is not None and len(getattr(self, name)) != n:
                raise ValueError(f"{name} has {len(getattr(self, name))} entries, expected {n}")
        if self.decimation < 1 or self.physics_dt <= 0:
            raise ValueError("decimation must be >= 1 and physics_dt > 0")
        if (self.saturation_effort is None) != (self.velocity_limit is None):
            raise ValueError("saturation_effort and velocity_limit must be given together")
        if len({t.name for t in self.obs_terms}) != len(self.obs_terms):
            raise ValueError("duplicate observation term names")
