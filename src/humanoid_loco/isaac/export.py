"""Export a trained rsl_rl policy to ONNX plus the manifest, read from the live environment."""

from __future__ import annotations

from pathlib import Path

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
    data = robot.data
    action = env.action_manager.get_term("joint_pos")
    ids = (
        list(range(robot.num_joints)) if isinstance(action._joint_ids, slice) else action._joint_ids
    )
    row = lambda t: t[0].tolist()  # noqa: E731  (env 0; identical across envs)
    scale = action._scale
    scale = [float(scale)] * len(ids) if isinstance(scale, float) else row(scale)

    # per-joint DC-motor limits; None if every leg actuator is a plain implicit PD
    sat = torch.full((robot.num_joints,), float("nan"), device=env.device)
    vmax = torch.full_like(sat, float("nan"))
    for act in robot.actuators.values():
        if hasattr(act, "saturation_effort"):
            sat[act.joint_indices] = float(act.saturation_effort)
            vmax[act.joint_indices] = act.velocity_limit[0]
    dc = not torch.isnan(sat[ids]).any()

    om = env.observation_manager
    terms = [
        ObsTerm(name, int(torch.tensor(dim).prod()), *_scale_clip(cfg))
        for name, dim, cfg in zip(
            om.active_terms["policy"],
            om.group_obs_term_dim["policy"],
            om._group_obs_term_cfgs["policy"],
            strict=True,
        )
    ]
    held = {
        robot.joint_names[j]: HeldJoint(
            float(data.default_joint_pos[0, j]),
            float(data.joint_stiffness[0, j]),
            float(data.joint_damping[0, j]),
        )
        for j in range(robot.num_joints)
        if j not in ids
    }
    clip = env.cfg.__dict__.get("clip_actions", None)
    return PolicyManifest(
        robot="unitree_g1",
        joint_names=[robot.joint_names[j] for j in ids],
        default_joint_pos=row(data.default_joint_pos[:, ids]),
        kp=row(data.joint_stiffness[:, ids]),
        kd=row(data.joint_damping[:, ids]),
        torque_limit=row(data.joint_effort_limits[:, ids]),
        action_scale=scale,
        obs_terms=terms,
        physics_dt=float(env.physics_dt),
        decimation=int(env.cfg.decimation),
        held_joints=held,
        action_clip=clip,
        saturation_effort=sat[ids].tolist() if dc else None,
        velocity_limit=vmax[ids].tolist() if dc else None,
        source=source,
    )


def _scale_clip(cfg) -> tuple[float, tuple[float, float] | None]:
    if cfg.scale is not None and not isinstance(cfg.scale, (int, float)):
        raise NotImplementedError("per-element observation scale is not supported by the manifest")
    return float(cfg.scale) if cfg.scale is not None else 1.0, tuple(cfg.clip) if cfg.clip else None
