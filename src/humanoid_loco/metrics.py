"""Trajectory container and plant-agnostic metrics: the same scoring for MuJoCo, Isaac and C++."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np


@dataclass
class Trajectory:
    """Policy-rate time series in manifest joint order. ``fell`` marks early termination."""

    time: list[float] = field(default_factory=list)
    command: list[np.ndarray] = field(default_factory=list)
    base_lin_vel: list[np.ndarray] = field(default_factory=list)
    base_ang_vel: list[np.ndarray] = field(default_factory=list)
    base_height: list[float] = field(default_factory=list)
    joint_pos: list[np.ndarray] = field(default_factory=list)
    joint_vel: list[np.ndarray] = field(default_factory=list)
    action: list[np.ndarray] = field(default_factory=list)
    torque: list[np.ndarray] = field(default_factory=list)
    fell: bool = False

    def arrays(self) -> dict[str, np.ndarray]:
        return {f.name: np.asarray(getattr(self, f.name)) for f in fields(self) if f.name != "fell"}

    @classmethod
    def from_dict(cls, d: dict) -> Trajectory:
        """Inverse of the JSON the C++ loop writes (same keys as the fields here)."""
        return cls(**{f.name: d[f.name] for f in fields(cls)})


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
    out = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
    out["fall_rate"] = out.pop("fell")
    out["n_runs"] = len(runs)
    return out


def write_report(path: str | Path, **sections) -> Path:
    path = Path(path)
    path.write_text(json.dumps(sections, indent=2) + "\n")
    return path


METRIC_KEYS = (
    "survived_s",
    "fall_rate",
    "vx_rmse",
    "vy_rmse",
    "yaw_rate_rmse",
    "torque_saturation_pct",
    "action_rate_rms",
    "mean_base_height_m",
    "n_runs",
)


def markdown_report(report: dict) -> str:
    """Render a sim2sim or sweep report json as Markdown tables (used for the README)."""
    if "grid" in report:
        axes = [k for k in report["grid"][0] if k not in METRIC_KEYS]
        return markdown_table(report["grid"], axes + ["fall_rate", "vx_rmse", "yaw_rate_rmse"])
    cols = ["fall_rate", "vx_rmse", "vy_rmse", "yaw_rate_rmse", "torque_saturation_pct", "n_runs"]
    return markdown_table(
        [{"schedule": report["schedule"], **report["summary"]}], ["schedule"] + cols
    )


def markdown_table(rows: list[dict], columns: list[str], fmt: str = "{:.3f}") -> str:
    head = "| " + " | ".join(columns) + " |\n|" + "---|" * len(columns) + "\n"
    body = "".join(
        "| "
        + " | ".join(fmt.format(r[c]) if isinstance(r[c], float) else str(r[c]) for c in columns)
        + " |\n"
        for r in rows
    )
    return head + body
