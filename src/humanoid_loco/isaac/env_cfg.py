"""Deployable 12-DOF G1 velocity task: stock Isaac Lab G1 flat task, three changes.

1. Robot is the 29-DOF G1 (joint names identical to MuJoCo Menagerie), legs commanded
   by the policy, waist/arms parked under their stiff implicit PD at the default pose.
2. Policy observations drop ``base_lin_vel`` (not measurable on hardware); an asymmetric
   ``critic`` group keeps it, noise-free.
3. Rewards that reference joints the policy no longer commands are removed.
"""

from __future__ import annotations

import copy

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.unitree import G1_29DOF_CFG
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import (
    G1FlatEnvCfg,
    G1FlatEnvCfg_PLAY,
)

LEG_JOINTS = [
    ".*_hip_yaw_joint",
    ".*_hip_roll_joint",
    ".*_hip_pitch_joint",
    ".*_knee_joint",
    ".*_ankle_pitch_joint",
    ".*_ankle_roll_joint",
]


def _to_12dof(cfg: G1FlatEnvCfg) -> None:
    cfg.scene.robot = G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cfg.scene.robot.spawn.activate_contact_sensors = True  # feet / torso contact rewards

    cfg.actions.joint_pos.joint_names = LEG_JOINTS

    policy = cfg.observations.policy
    critic = copy.deepcopy(policy)  # privileged copy: keeps base_lin_vel, no noise
    critic.enable_corruption = False
    policy.base_lin_vel = None
    for term in (policy.joint_pos, policy.joint_vel, critic.joint_pos, critic.joint_vel):
        term.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=LEG_JOINTS)  # one per term
    cfg.observations.critic = critic

    cfg.rewards.joint_deviation_arms = None
    cfg.rewards.joint_deviation_fingers = None
    cfg.rewards.joint_deviation_torso = None


@configclass
class G1Flat12DofEnvCfg(G1FlatEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _to_12dof(self)


@configclass
class G1Flat12DofEnvCfg_PLAY(G1FlatEnvCfg_PLAY):
    def __post_init__(self) -> None:
        super().__post_init__()
        _to_12dof(self)
