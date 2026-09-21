"""``hls`` command line. Isaac subcommands need Isaac Sim's Python; the rest run anywhere."""

from __future__ import annotations

import argparse
from pathlib import Path

MENAGERIE = Path("third_party/mujoco_menagerie/unitree_g1")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="hls", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="PPO training in Isaac Lab")
    t.add_argument("--task", default="HLS-G1-Flat-v0")
    t.add_argument("--num-envs", type=int, default=2048)
    t.add_argument("--max-iterations", type=int)
    t.add_argument("--headless", action="store_true")
    t.add_argument("--device", default="cuda:0")

    pl = sub.add_parser("play", help="roll out a checkpoint in Isaac Lab; optionally export")
    pl.add_argument("--task", default="HLS-G1-Flat-Play-v0")
    pl.add_argument("--checkpoint", default="latest")
    pl.add_argument("--num-envs", type=int, default=16)
    pl.add_argument("--steps", type=int, default=1000)
    pl.add_argument("--headless", action="store_true")
    pl.add_argument("--device", default="cuda:0")
    pl.add_argument("--video-dir", type=Path)
    pl.add_argument("--export-dir", type=Path, help="write policy.onnx + policy.json here")

    s2s = sub.add_parser("sim2sim", help="run the exported policy in MuJoCo and report metrics")
    s2s.add_argument("--run", type=Path, required=True, help="dir with policy.onnx + policy.json")
    s2s.add_argument("--commands", type=Path, default=Path("configs/commands/walk.yaml"))
    s2s.add_argument("--seeds", type=int, default=10)
    s2s.add_argument("--menagerie", type=Path, default=MENAGERIE)
    s2s.add_argument("--video", type=Path, help="write a .mp4/.gif of seed 0")
    s2s.add_argument("--out", type=Path, help="report json (default: <run>/sim2sim-<name>.json)")

    sw = sub.add_parser("sweep", help="fall-rate grid over pelvis mass x floor friction")
    sw.add_argument("--run", type=Path, required=True)
    sw.add_argument("--commands", type=Path, default=Path("configs/commands/walk.yaml"))
    sw.add_argument("--seeds", type=int, default=5)
    sw.add_argument("--mass", type=float, nargs="+", default=[0.8, 1.0, 1.2, 1.4])
    sw.add_argument("--friction", type=float, nargs="+", default=[0.4, 0.7, 1.0])
    sw.add_argument("--menagerie", type=Path, default=MENAGERIE)
    sw.add_argument("--out", type=Path)

    pc = sub.add_parser("prepare-cpp", help="write scene.mjb + schedule json for cpp/hls_loop")
    pc.add_argument("--run", type=Path, required=True)
    pc.add_argument("--commands", type=Path, default=Path("configs/commands/walk.yaml"))
    pc.add_argument("--menagerie", type=Path, default=MENAGERIE)
    pc.add_argument("--control-dt", type=float, default=0.001)

    rc = sub.add_parser("score", help="score a trajectory json (e.g. from cpp/hls_loop)")
    rc.add_argument("--run", type=Path, required=True)
    rc.add_argument("trajectory", type=Path)

    a = p.parse_args(argv)
    {
        "train": _train,
        "play": _play,
        "sim2sim": _sim2sim,
        "sweep": _sweep,
        "prepare-cpp": _prepare_cpp,
        "score": _score,
    }[a.cmd](a)


def _train(a) -> None:
    from humanoid_loco.isaac.runner import train

    train(a.task, a.num_envs, a.max_iterations, a.headless, a.device)


def _play(a) -> None:
    from humanoid_loco.isaac.runner import play, resolve_checkpoint

    play(
        a.task,
        resolve_checkpoint(a.checkpoint),
        a.num_envs,
        a.steps,
        a.headless,
        a.device,
        a.video_dir,
        a.export_dir,
    )


def _load_run(a):
    from humanoid_loco.manifest import PolicyManifest
    from humanoid_loco.mujoco.env import MujocoG1
    from humanoid_loco.mujoco.rollout import Schedule
    from humanoid_loco.policy import OnnxPolicy

    manifest = PolicyManifest.load(a.run / "policy.json")
    policy = OnnxPolicy(manifest, a.run / manifest.onnx)
    make_env = lambda: MujocoG1(manifest, a.menagerie)  # noqa: E731
    return manifest, policy, make_env, Schedule.from_yaml(a.commands)


def _sim2sim(a) -> None:
    from humanoid_loco.mujoco.metrics import evaluate, evaluate_schedule, write_report
    from humanoid_loco.mujoco.rollout import rollout, save_video

    manifest, policy, make_env, schedule = _load_run(a)
    env = make_env()
    frames = [] if a.video else None
    first = evaluate(rollout(env, policy, schedule, seed=0, frames=frames), manifest)
    if a.video:
        save_video(frames, a.video, fps=0.5 / manifest.policy_dt)
    summary = evaluate_schedule(env, policy, schedule, range(a.seeds))
    out = a.out or a.run / f"sim2sim-{schedule.name}.json"
    write_report(out, schedule=schedule.name, seed0=first, summary=summary, source=manifest.source)
    print(
        f"{schedule.name}: fall_rate={summary['fall_rate']:.2f} "
        f"vx_rmse={summary['vx_rmse']:.3f} yaw_rmse={summary['yaw_rate_rmse']:.3f} -> {out}"
    )


def _sweep(a) -> None:
    from humanoid_loco.mujoco.metrics import markdown_table, sweep, write_report

    manifest, policy, make_env, schedule = _load_run(a)
    rows = sweep(make_env, policy, schedule, a.mass, a.friction, range(a.seeds))
    out = a.out or a.run / f"sweep-{schedule.name}.json"
    write_report(out, schedule=schedule.name, grid=rows, source=manifest.source)
    print(markdown_table(rows, ["mass_scale", "friction_scale", "fall_rate", "vx_rmse"]))


def _prepare_cpp(a) -> None:
    import json

    import mujoco

    from humanoid_loco.manifest import PolicyManifest
    from humanoid_loco.mujoco.rollout import Schedule
    from humanoid_loco.mujoco.scene import build_model

    manifest = PolicyManifest.load(a.run / "policy.json")
    mujoco.mj_saveModel(build_model(manifest, a.menagerie, a.control_dt), str(a.run / "scene.mjb"))
    schedule = Schedule.from_yaml(a.commands)
    out = a.run / f"schedule-{schedule.name}.json"
    out.write_text(json.dumps(schedule.to_dict()) + "\n")
    print(f"wrote {a.run / 'scene.mjb'} and {out}")


def _score(a) -> None:
    import json

    from humanoid_loco.manifest import PolicyManifest
    from humanoid_loco.mujoco.metrics import evaluate
    from humanoid_loco.mujoco.rollout import Trajectory

    manifest = PolicyManifest.load(a.run / "policy.json")
    traj = Trajectory.from_dict(json.loads(a.trajectory.read_text()))
    print(json.dumps(evaluate(traj, manifest), indent=2))
