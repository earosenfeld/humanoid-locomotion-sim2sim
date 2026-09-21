"""PPO runner config: stock G1 flat hyper-parameters with an asymmetric critic."""

from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg import (
    G1FlatPPORunnerCfg,
)


@configclass
class G1Flat12DofPPORunnerCfg(G1FlatPPORunnerCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.experiment_name = "g1_flat_12dof"
        self.obs_groups = {"policy": ["policy"], "critic": ["critic"]}
