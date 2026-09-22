"""Joint position action with a per-environment transport delay, resampled at every reset.

Real deployments have one or two control periods between inference and the PD target taking
effect. Training without it produced a policy that fell 80% of the time under a single 20 ms
step of delay in MuJoCo (see the README latency sweep).
"""

from __future__ import annotations

from dataclasses import MISSING

import torch
from isaaclab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from isaaclab.utils import configclass
from isaaclab.utils.buffers import DelayBuffer


class DelayedJointPositionAction(JointPositionAction):
    cfg: DelayedJointPositionActionCfg

    def __init__(self, cfg: DelayedJointPositionActionCfg, env) -> None:
        super().__init__(cfg, env)
        self._delay = DelayBuffer(cfg.max_delay, self.num_envs, device=self.device)

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        ids = list(range(self.num_envs)) if env_ids is None else env_ids
        lag = torch.randint(
            0, self.cfg.max_delay + 1, (len(ids),), device=self.device, dtype=torch.int
        )
        self._delay.set_time_lag(lag, ids)
        self._delay.reset(ids)

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(self._delay.compute(actions))


@configclass
class DelayedJointPositionActionCfg(JointPositionActionCfg):
    class_type: type = DelayedJointPositionAction
    max_delay: int = MISSING
    """Maximum delay in policy steps; each env samples uniformly from [0, max_delay] at reset."""
