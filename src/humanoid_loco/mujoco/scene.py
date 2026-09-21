"""Build the MuJoCo model for sim-to-sim from MuJoCo Menagerie + the policy manifest.

The Menagerie G1 ships with stiff position actuators (kp=500). We replace them with:
  * torque motors on every policy joint  -> explicit PD in ``env.py`` (what a real SDK does)
  * position actuators on held joints    -> upper body parked at the training default pose
and add an IMU (gyro, orientation, velocimeter) at the pelvis so observations come from
sensors, not from privileged simulator state.
"""

from __future__ import annotations

from pathlib import Path

import mujoco

from humanoid_loco.manifest import PolicyManifest

IMU_SITE = "imu_in_pelvis"
SENSORS = {  # name -> (type, dim)
    "imu_gyro": (mujoco.mjtSensor.mjSENS_GYRO, 3),
    "imu_quat": (mujoco.mjtSensor.mjSENS_FRAMEQUAT, 4),
    "imu_vel": (mujoco.mjtSensor.mjSENS_VELOCIMETER, 3),
}


def build_model(
    manifest: PolicyManifest, menagerie_dir: str | Path, timestep: float
) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(Path(menagerie_dir) / "scene.xml"))
    spec.option.timestep = timestep

    for act in list(spec.actuators):
        spec.delete(act)
    for name, limit in zip(manifest.joint_names, manifest.torque_limit, strict=True):
        act = _joint_actuator(spec, name)
        act.set_to_motor()
        act.ctrllimited, act.ctrlrange = True, [-limit, limit]
    model_joints = {j.name for j in spec.joints}
    for name, held in manifest.held_joints.items():
        if name in model_joints:  # training asset may carry joints this model lacks (hands)
            _joint_actuator(spec, name).set_to_position(kp=held.kp, kv=held.kd)
    uncovered = (
        model_joints
        - set(manifest.joint_names)
        - set(manifest.held_joints)
        - {"floating_base_joint"}
    )
    if uncovered:
        raise ValueError(f"MuJoCo joints with no policy or hold actuator: {sorted(uncovered)}")

    for name, (kind, _) in SENSORS.items():
        spec.add_sensor(name=name, type=kind, objtype=mujoco.mjtObj.mjOBJ_SITE, objname=IMU_SITE)
    return spec.compile()


def _joint_actuator(spec: mujoco.MjSpec, joint: str) -> mujoco.MjsActuator:
    return spec.add_actuator(name=joint, target=joint, trntype=mujoco.mjtTrn.mjTRN_JOINT)
