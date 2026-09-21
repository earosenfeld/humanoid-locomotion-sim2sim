"""Generate the README figures from the REAL pipeline: exported policy, MuJoCo plant, reports.

No hand-authored data. Run after `hls play --export-dir <run>` and `hls sweep --run <run>`:

    .venv/bin/python scripts/make_figures.py --run <export dir> [--tb <rsl_rl log dir>]

Outputs into ``assets/``:
    velocity_tracking.png   commanded vs achieved base velocity in MuJoCo (walk schedule)
    gait_cycle.png          leg joint trajectories over two seconds of 1 m/s walking
    robustness_sweep.png    fall rate over pelvis-mass x floor-friction (from sweep json)
    push_recovery.png       fall rate vs push impulse on the pelvis (from sweep-push json)
    reward_curve.png        PPO mean reward vs iteration (from the TensorBoard run dir)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "savefig.bbox": "tight",
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#334155", "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": "#e2e8f0", "grid.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.labelsize": 11, "legend.frameon": False, "lines.linewidth": 2.0,
})  # fmt: skip
PALETTE = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2"]

from humanoid_loco.manifest import PolicyManifest  # noqa: E402
from humanoid_loco.mujoco.env import MujocoG1  # noqa: E402
from humanoid_loco.mujoco.rollout import Schedule, rollout  # noqa: E402
from humanoid_loco.policy import OnnxPolicy  # noqa: E402

ASSETS = Path(__file__).resolve().parents[1] / "assets"
MENAGERIE = Path("third_party/mujoco_menagerie/unitree_g1")


def figure_velocity_tracking(traj, out: Path) -> Path:
    a = traj.arrays()
    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    achieved = np.column_stack([a["base_lin_vel"][:, :2], a["base_ang_vel"][:, 2]])
    for i, (ax, label) in enumerate(
        zip(axes, ["v_x [m/s]", "v_y [m/s]", "yaw rate [rad/s]"], strict=True)
    ):
        ax.plot(a["time"], a["command"][:, i], color="#334155", ls="--", lw=1.5, label="command")
        ax.plot(a["time"], achieved[:, i], color=PALETTE[i], label="achieved (MuJoCo)")
        ax.set_ylabel(label)
    axes[0].set_title("Sim-to-sim velocity tracking, walk schedule")
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("time [s]")
    fig.savefig(out)
    return out


def figure_gait_cycle(traj, manifest: PolicyManifest, out: Path, t0: float = 8.0) -> Path:
    a = traj.arrays()
    sel = (a["time"] >= t0) & (a["time"] < t0 + 2.0)
    fig, axes = plt.subplots(2, 3, figsize=(11, 5), sharex=True)
    for ax, stem in zip(
        axes.ravel(),
        ["hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"],
        strict=True,
    ):
        for side, color in (("left", PALETTE[0]), ("right", PALETTE[1])):
            j = manifest.joint_names.index(f"{side}_{stem}_joint")
            ax.plot(a["time"][sel] - t0, a["joint_pos"][sel, j], color=color, label=side)
        ax.set_title(stem, fontsize=11)
    axes[0, 0].legend()
    for ax in axes[-1]:
        ax.set_xlabel("time [s]")
    axes[0, 0].set_ylabel("angle [rad]"), axes[1, 0].set_ylabel("angle [rad]")
    fig.suptitle("Leg joint trajectories at 1 m/s (MuJoCo)", fontweight="bold")
    fig.savefig(out)
    return out


def figure_robustness_sweep(sweep_json: Path, out: Path) -> Path:
    rows = json.loads(sweep_json.read_text())["grid"]
    masses = sorted({r["mass_scale"] for r in rows})
    frictions = sorted({r["friction_scale"] for r in rows})
    grid = np.full((len(masses), len(frictions)), np.nan)
    for r in rows:
        grid[masses.index(r["mass_scale"]), frictions.index(r["friction_scale"])] = r["fall_rate"]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(grid, cmap="RdYlGn_r", vmin=0, vmax=1, origin="lower")
    ax.set_xticks(range(len(frictions)), [f"{f:g}x" for f in frictions])
    ax.set_yticks(range(len(masses)), [f"{m:g}x" for m in masses])
    ax.set_xlabel("floor friction scale"), ax.set_ylabel("pelvis mass scale")
    ax.set_title("Fall rate under domain shift (MuJoCo)")
    ax.grid(False)
    for (i, j), v in np.ndenumerate(grid):
        ax.text(j, i, f"{v:.0%}", ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, label="fall rate")
    fig.savefig(out)
    return out


def figure_push_recovery(sweep_json: Path, out: Path) -> Path:
    rep = json.loads(sweep_json.read_text())
    rows = sorted(rep["grid"], key=lambda r: r["push_scale"])
    push = Schedule.from_yaml("configs/commands/push.yaml")
    force = max(np.linalg.norm(s.push) for s in push.segments if s.push)
    dur = next(s.duration for s in push.segments if s.push)
    impulse = [r["push_scale"] * force * dur for r in rows]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(impulse, [100 * r["fall_rate"] for r in rows], marker="o", color=PALETTE[1])
    ax.set_xlabel(f"push impulse on pelvis [N s]  ({dur:g} s pulse while walking at 0.5 m/s)")
    ax.set_ylabel("fall rate [%]")
    ax.set_ylim(-5, 105)
    ax.set_title("Push recovery limit (MuJoCo)")
    fig.savefig(out)
    return out


def figure_reward_curve(tb_dir: Path, out: Path) -> Path:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    acc = EventAccumulator(str(tb_dir), size_guidance={"scalars": 0})
    acc.Reload()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax2 = ax.twinx()
    ax2.grid(False)
    for tag, axis, color in (
        ("Train/mean_reward", ax, PALETTE[0]),
        ("Train/mean_episode_length", ax2, PALETTE[2]),
    ):
        if tag in acc.Tags()["scalars"]:
            ev = acc.Scalars(tag)
            axis.plot([e.step for e in ev], [e.value for e in ev], color=color)
            axis.set_ylabel(tag.split("/")[1].replace("_", " "), color=color)
    ax.set_xlabel("PPO iteration")
    ax.set_title("Isaac Lab PPO, 2048 envs (episode cap 1000 steps)")
    fig.savefig(out)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run", type=Path, required=True, help="dir with policy.json/onnx and sweep json"
    )
    p.add_argument("--tb", type=Path, help="rsl_rl log dir with events.out.tfevents.*")
    p.add_argument("--menagerie", type=Path, default=MENAGERIE)
    a = p.parse_args()
    ASSETS.mkdir(exist_ok=True)

    manifest = PolicyManifest.load(a.run / "policy.json")
    policy = OnnxPolicy(manifest, a.run / manifest.onnx)
    env = MujocoG1(manifest, a.menagerie)
    traj = rollout(env, policy, Schedule.from_yaml("configs/commands/walk.yaml"), seed=0)
    print(figure_velocity_tracking(traj, ASSETS / "velocity_tracking.png"))
    print(figure_gait_cycle(traj, manifest, ASSETS / "gait_cycle.png"))
    sweep = a.run / "sweep-walk.json"
    if sweep.exists():
        print(figure_robustness_sweep(sweep, ASSETS / "robustness_sweep.png"))
    push = a.run / "sweep-push.json"
    if push.exists():
        print(figure_push_recovery(push, ASSETS / "push_recovery.png"))
    if a.tb:
        print(figure_reward_curve(a.tb, ASSETS / "reward_curve.png"))


if __name__ == "__main__":
    main()
