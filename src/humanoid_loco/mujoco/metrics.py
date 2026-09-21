"""Sim-to-sim evaluation metrics and the robustness sweep."""

from __future__ import annotations

import itertools
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from humanoid_loco.mujoco.env import MujocoG1
from humanoid_loco.mujoco.rollout import Perturbation, Schedule, Trajectory, rollout
from humanoid_loco.policy import OnnxPolicy


def evaluate(traj: Trajectory, manifest) -> dict[str, float]:
    a = traj.arrays()
    err = np.column_stack([a["base_lin_vel"][:, :2], a["base_ang_vel"][:, 2]]) - a["command"]
    tau_sat = np.abs(a["torque"]) >= 0.98 * np.asarray(manifest.torque_limit)
    d_action = np.diff(a["action"], axis=0) if len(a["action"]) > 1 else np.zeros((1, 1))
    return {
        "survived_s": float(a["time"][-1]) if len(a["time"]) else 0.0,
        "fell": float(traj.fell),
        "vx_rmse": float(np.sqrt(np.mean(err[:, 0] ** 2))),
        "vy_rmse": float(np.sqrt(np.mean(err[:, 1] ** 2))),
        "yaw_rate_rmse": float(np.sqrt(np.mean(err[:, 2] ** 2))),
        "torque_saturation_pct": float(100 * tau_sat.mean()),
        "action_rate_rms": float(np.sqrt(np.mean(d_action**2))),
        "mean_base_height_m": float(np.mean(a["base_height"])),
    }


def aggregate(runs: Iterable[dict[str, float]]) -> dict[str, float]:
    runs = list(runs)
    keys = runs[0].keys()
    out = {k: float(np.mean([r[k] for r in runs])) for k in keys}
    out["fall_rate"] = out.pop("fell")
    out["n_runs"] = len(runs)
    return out


def evaluate_schedule(
    env: MujocoG1, policy: OnnxPolicy, schedule: Schedule, seeds: range, **kw
) -> dict[str, float]:
    return aggregate(
        evaluate(rollout(env, policy, schedule, seed=s, **kw), env.manifest) for s in seeds
    )


def sweep(
    make_env,
    policy: OnnxPolicy,
    schedule: Schedule,
    mass_scales: Iterable[float],
    friction_scales: Iterable[float],
    seeds: range,
) -> list[dict]:
    """Fall rate across a mass x friction grid. ``make_env`` returns a fresh plant per cell."""
    rows = []
    for ms, fs in itertools.product(mass_scales, friction_scales):
        env = make_env()
        pert = Perturbation(mass_scale=ms, friction_scale=fs)
        m = evaluate_schedule(env, policy, schedule, seeds, perturbation=pert)
        rows.append({"mass_scale": ms, "friction_scale": fs, **m})
    return rows


def write_report(path: str | Path, **sections) -> Path:
    path = Path(path)
    path.write_text(json.dumps(sections, indent=2) + "\n")
    return path


def markdown_table(rows: list[dict], columns: list[str], fmt: str = "{:.3f}") -> str:
    head = "| " + " | ".join(columns) + " |\n|" + "---|" * len(columns) + "\n"
    body = "".join(
        "| "
        + " | ".join(fmt.format(r[c]) if isinstance(r[c], float) else str(r[c]) for c in columns)
        + " |\n"
        for r in rows
    )
    return head + body
