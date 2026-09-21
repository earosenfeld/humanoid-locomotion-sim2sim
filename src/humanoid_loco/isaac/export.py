"""Export a trained rsl_rl policy to ONNX plus the manifest, read from the live environment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from humanoid_loco.manifest import HeldJoint, ObsTerm, PolicyManifest


def export(runner, env, out_dir: str | Path, source: dict) -> PolicyManifest:
    from isaaclab_rl.rsl_rl import export_policy_as_onnx

    out_dir = Path(out_dir)
    policy = runner.alg.policy
    export_policy_as_onnx(policy, str(out_dir), normalizer=policy.actor_obs_normalizer)
    manifest = build_manifest(env.unwrapped, source)
    manifest.save(out_dir / "policy.json")
    return manifest


def build_manifest(env, source: dict) -> PolicyManifest:
    robot = env.scene["robot"]
    action = env.action_manager.get_term("joint_pos")
    ids = (
        list(range(robot.num_joints)) if isinstance(action._joint_ids, slice) else action._joint_ids
    )
    scale = action._scale
    scale = [float(scale)] * len(ids) if isinstance(scale, float) else scale[0].tolist()

    # Gains and limits live on the actuator models (explicit actuators zero the sim's own drive).
    gains = _per_joint_actuator_params(robot)
    dc = not np.isnan(gains["saturation_effort"][ids]).any()
    om = env.observation_manager
    terms = [
        ObsTerm(name, int(np.prod(dim)), *_scale_clip(cfg))
        for name, dim, cfg in zip(
            om.active_terms["policy"],
            om.group_obs_term_dim["policy"],
            om._group_obs_term_cfgs["policy"],
            strict=True,
        )
    ]
    default = robot.data.default_joint_pos[0].cpu().numpy()
    held = {
        robot.joint_names[j]: HeldJoint(
            float(default[j]), float(gains["kp"][j]), float(gains["kd"][j])
        )
        for j in range(robot.num_joints)
        if j not in ids
    }
    return PolicyManifest(
        robot="unitree_g1",
        joint_names=[robot.joint_names[j] for j in ids],
        default_joint_pos=default[ids].tolist(),
        kp=gains["kp"][ids].tolist(),
        kd=gains["kd"][ids].tolist(),
        torque_limit=gains["effort_limit"][ids].tolist(),
        action_scale=scale,
        obs_terms=terms,
        physics_dt=float(env.physics_dt),
        decimation=int(env.cfg.decimation),
        held_joints=held,
        action_clip=getattr(env.cfg, "clip_actions", None),
        saturation_effort=gains["saturation_effort"][ids].tolist() if dc else None,
        velocity_limit=gains["velocity_limit"][ids].tolist() if dc else None,
        source=source,
    )


def _per_joint_actuator_params(robot) -> dict[str, np.ndarray]:
    """Flatten every actuator group's tensors into (num_joints,) arrays; NaN where absent."""
    out = {k: np.full(robot.num_joints, np.nan) for k in _PARAMS}
    for act in robot.actuators.values():
        idx = act.joint_indices
        idx = idx.cpu().numpy() if torch.is_tensor(idx) else idx
        for key, attr in _PARAMS.items():
            value = getattr(act, attr, None)
            if value is not None:
                value = value[0].cpu().numpy() if torch.is_tensor(value) else float(value)
                out[key][idx] = value
    return out


_PARAMS = {  # manifest field -> isaaclab.actuators.ActuatorBase attribute
    "kp": "stiffness",
    "kd": "damping",
    "effort_limit": "effort_limit",
    "velocity_limit": "velocity_limit",
    "saturation_effort": "saturation_effort",
}


def _scale_clip(cfg) -> tuple[float, tuple[float, float] | None]:
    if cfg.scale is not None and not isinstance(cfg.scale, (int, float)):
        raise NotImplementedError("per-element observation scale is not supported by the manifest")
    return float(cfg.scale) if cfg.scale is not None else 1.0, tuple(cfg.clip) if cfg.clip else None
