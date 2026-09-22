"""Deployable 12-DOF G1 velocity task: stock Isaac Lab G1 flat task, three changes.

1. Robot is the 29-DOF G1 (joint names identical to MuJoCo Menagerie), legs commanded
   by the policy, waist/arms parked under their stiff implicit PD at the default pose.
2. Policy observations drop ``base_lin_vel`` (not measurable on hardware); an asymmetric
   ``critic`` group keeps it, noise-free.
3. Rewards that reference joints the policy no longer commands are removed.
4. Kneeling is not a solution: the stock task terminates only on torso contact, and with the
   29-DOF model's softer leg gains PPO found a stable knee-fall (base at 0.15 m, zero speed)
   that never terminates. Low base height and bad orientation now terminate, and a base-height
   reward keeps the gait upright.
5. Actions reach the PD with a per-env transport delay of 0-2 policy steps (0-40 ms), resampled
   at reset, so the policy tolerates real inference-to-actuator latency.
"""

from __future__ import annotations

import copy

import isaaclab.envs.mdp as mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab_assets.robots.unitree import G1_29DOF_CFG
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import (
    G1FlatEnvCfg,
    G1FlatEnvCfg_PLAY,
)

from humanoid_loco.isaac.actions import DelayedJointPositionActionCfg

STAND_HEIGHT = 0.74  # pelvis height of the 29-DOF G1 in its default (slightly crouched) pose

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

    cfg.actions.joint_pos = DelayedJointPositionActionCfg(
        asset_name="robot", joint_names=LEG_JOINTS, scale=0.5, use_default_offset=True, max_delay=2
    )

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

    cfg.terminations.base_height = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": 0.5}
    )
    cfg.terminations.bad_orientation = DoneTerm(
        func=mdp.bad_orientation, params={"limit_angle": 0.8}
    )
    cfg.rewards.base_height = RewTerm(
        func=mdp.base_height_l2, weight=-1.0, params={"target_height": STAND_HEIGHT}
    )


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
