"""MuJoCo plant for the G1: explicit PD torque at the control rate, policy-rate stepping.

Training (Isaac Lab) uses an implicit PD inside PhysX at ``manifest.physics_dt``. Hardware runs
an explicit PD in the motor driver at ~1 kHz, so this plant does the same: ``control_dt`` is a
deployment parameter, independent of the training step. The policy rate comes from the manifest.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from humanoid_loco.manifest import PolicyManifest
from humanoid_loco.mujoco.scene import SENSORS, build_model
from humanoid_loco.obs import RobotState

_JOINT, _ACT, _SENSOR, _BODY = (
    mujoco.mjtObj.mjOBJ_JOINT,
    mujoco.mjtObj.mjOBJ_ACTUATOR,
    mujoco.mjtObj.mjOBJ_SENSOR,
    mujoco.mjtObj.mjOBJ_BODY,
)


class MujocoG1:
    """All joint arrays cross this boundary in manifest (policy) order via index maps."""

    def __init__(
        self, manifest: PolicyManifest, menagerie_dir: str | Path, control_dt: float = 0.001
    ):
        self.manifest = manifest
        self.control_dt = control_dt
        self.substeps = round(manifest.policy_dt / control_dt)
        if not np.isclose(self.substeps * control_dt, manifest.policy_dt):
            raise ValueError(
                f"control_dt {control_dt} does not divide policy_dt {manifest.policy_dt}"
            )
        self.model = build_model(manifest, menagerie_dir, control_dt)
        self.data = mujoco.MjData(self.model)
        m = self.model

        jids = [mujoco.mj_name2id(m, _JOINT, n) for n in manifest.joint_names]
        self._qadr = m.jnt_qposadr[jids]
        self._vadr = m.jnt_dofadr[jids]
        self._act = np.array([mujoco.mj_name2id(m, _ACT, n) for n in manifest.joint_names])
        self._sensor = {
            n: slice(m.sensor_adr[mujoco.mj_name2id(m, _SENSOR, n)], None) for n in SENSORS
        }
        for n, (_, dim) in SENSORS.items():
            self._sensor[n] = slice(self._sensor[n].start, self._sensor[n].start + dim)
        self._held_act = np.array(
            [mujoco.mj_name2id(m, _ACT, n) for n in manifest.held_joints], dtype=int
        )
        self._held_qadr = np.array(
            [m.jnt_qposadr[mujoco.mj_name2id(m, _JOINT, n)] for n in manifest.held_joints],
            dtype=int,
        )
        self._held_pos = np.array([h.pos for h in manifest.held_joints.values()])
        self.pelvis = mujoco.mj_name2id(m, _BODY, "pelvis")
        self._foot_geoms = [
            g
            for g in range(m.ngeom)
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE
            and "ankle_roll" in mujoco.mj_id2name(m, _BODY, m.geom_bodyid[g])
        ]

        self.kp = np.asarray(manifest.kp)
        self.kd = np.asarray(manifest.kd)
        self.default_pos = np.asarray(manifest.default_joint_pos)
        # nominal plant parameters, so perturbations can be applied idempotently
        self.nominal = {
            "body_mass": m.body_mass.copy(),
            "geom_friction": m.geom_friction.copy(),
            "kp": self.kp.copy(),
        }
        self.reset()

    # ---- state ---------------------------------------------------------------------
    def reset(self, keyframe: str = "stand") -> RobotState:
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key(keyframe).id)
        self.data.qpos[self._qadr] = self.default_pos
        self.data.qpos[self._held_qadr] = self._held_pos
        self.data.ctrl[self._held_act] = self._held_pos
        self.data.qvel[:] = 0.0
        self.data.xfrc_applied[:] = 0.0
        self.last_torque = np.zeros(self.manifest.num_actions)
        mujoco.mj_forward(self.model, self.data)
        self._settle_feet()
        return self.state()

    def reset_from_data(self) -> RobotState:
        """Re-derive kinematics after an external edit of ``data.qpos`` (e.g. initial noise)."""
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._settle_feet()
        return self.state()

    def _settle_feet(self, clearance: float = 0.002) -> None:
        """Lower the base so the lowest foot contact sphere just touches the floor."""
        m, d = self.model, self.data
        lowest = min(d.geom_xpos[g, 2] - m.geom_size[g, 0] for g in self._foot_geoms)
        d.qpos[2] += clearance - lowest
        mujoco.mj_forward(m, d)

    def state(self) -> RobotState:
        s = self.data.sensordata
        return RobotState(
            base_quat=s[self._sensor["imu_quat"]].copy(),
            base_ang_vel=s[self._sensor["imu_gyro"]].copy(),
            joint_pos=self.data.qpos[self._qadr].copy(),
            joint_vel=self.data.qvel[self._vadr].copy(),
            base_lin_vel=s[self._sensor["imu_vel"]].copy(),
        )

    @property
    def base_height(self) -> float:
        return float(self.data.xpos[self.pelvis, 2])

    @property
    def time(self) -> float:
        return float(self.data.time)

    # ---- control -------------------------------------------------------------------
    def pd_torque(self, targets: np.ndarray) -> np.ndarray:
        q, qd = self.data.qpos[self._qadr], self.data.qvel[self._vadr]
        return np.clip(self.kp * (targets - q) - self.kd * qd, *self.manifest.torque_bounds(qd))

    def step(self, targets: np.ndarray) -> RobotState:
        """Advance one policy step: ``substeps`` control ticks, each a PD update + physics step."""
        for _ in range(self.substeps):
            self.last_torque = self.pd_torque(targets)
            self.data.ctrl[self._act] = self.last_torque
            mujoco.mj_step(self.model, self.data)
        return self.state()

    def push(self, force_world: np.ndarray) -> None:
        """Set (or clear with zeros) an external force on the pelvis, in world frame."""
        self.data.xfrc_applied[self.pelvis, :3] = force_world
