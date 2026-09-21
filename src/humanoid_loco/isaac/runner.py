"""train / play / export runners. Isaac Lab's pip package ships no scripts; these are ours.

Every function launches Isaac Sim first (``AppLauncher``) and only then imports Isaac Lab,
which is the required import order.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

LOG_ROOT = Path("runs")


def _launch(headless: bool, cameras: bool = False):
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    from isaaclab.app import AppLauncher

    return AppLauncher(headless=headless, enable_cameras=cameras).app


def _make_env(task: str, num_envs: int | None, device: str, video_dir: Path | None = None):
    import gymnasium as gym
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    import humanoid_loco.isaac  # noqa: F401  registers the tasks

    env_cfg = parse_env_cfg(task, device=device, num_envs=num_envs)
    agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    env_cfg.seed = agent_cfg.seed
    env = gym.make(task, cfg=env_cfg, render_mode="rgb_array" if video_dir else None)
    if video_dir:
        env = gym.wrappers.RecordVideo(
            env,
            str(video_dir),
            step_trigger=lambda s: s == 0,
            video_length=600,
            disable_logger=True,
        )
    return RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions), env_cfg, agent_cfg


def train(task: str, num_envs: int, max_iterations: int | None, headless: bool, device: str):
    app = _launch(headless)
    from isaaclab.utils.io import dump_yaml
    from rsl_rl.runners import OnPolicyRunner

    env, env_cfg, agent_cfg = _make_env(task, num_envs, device)
    log_dir = LOG_ROOT / agent_cfg.experiment_name / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(log_dir), device=agent_cfg.device)
    dump_yaml(str(log_dir / "params" / "env.yaml"), env_cfg)
    dump_yaml(str(log_dir / "params" / "agent.yaml"), agent_cfg)
    runner.learn(max_iterations or agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()
    app.close()


def play(
    task: str,
    checkpoint: str,
    num_envs: int,
    steps: int,
    headless: bool,
    device: str,
    video_dir: Path | None,
    export_dir: Path | None,
):
    app = _launch(headless, cameras=video_dir is not None)
    import torch
    from rsl_rl.runners import OnPolicyRunner

    from humanoid_loco.isaac.export import export

    env, _, agent_cfg = _make_env(task, num_envs, device, video_dir)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    if export_dir:
        export(runner, env, export_dir, {"task": task, "checkpoint": str(checkpoint)})
    obs = env.get_observations()
    with torch.inference_mode():
        for _ in range(steps):
            obs, *_ = env.step(policy(obs))
    env.close()
    app.close()


def resolve_checkpoint(spec: str) -> str:
    """'latest' -> newest model_*.pt under runs/; otherwise the path as given."""
    if spec != "latest":
        return spec
    ckpts = sorted(LOG_ROOT.glob("*/*/model_*.pt"), key=lambda p: (p.stat().st_mtime, p.name))
    if not ckpts:
        raise FileNotFoundError(f"no checkpoints under {LOG_ROOT}/")
    return str(ckpts[-1])
