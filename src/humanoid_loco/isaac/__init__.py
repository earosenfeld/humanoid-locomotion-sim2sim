"""Isaac Lab task registration. Import only under Isaac Sim's Python, after ``AppLauncher``."""

import gymnasium as gym

TASK = "HLS-G1-Flat-v0"
TASK_PLAY = "HLS-G1-Flat-Play-v0"
TASK_EVAL = "HLS-G1-Flat-Eval-v0"  # deterministic resets for schedule evaluation

for task_id, cfg in (
    (TASK, "G1Flat12DofEnvCfg"),
    (TASK_PLAY, "G1Flat12DofEnvCfg_PLAY"),
    (TASK_EVAL, "G1Flat12DofEnvCfg_EVAL"),
):
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.env_cfg:{cfg}",
            "rsl_rl_cfg_entry_point": f"{__name__}.agent_cfg:G1Flat12DofPPORunnerCfg",
        },
    )
