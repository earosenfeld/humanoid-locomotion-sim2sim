"""MuJoCo-side evaluation: schedules over seeds and perturbation sweeps (scoring is shared)."""

from __future__ import annotations

import itertools
from collections.abc import Iterable

from humanoid_loco.metrics import aggregate, evaluate, markdown_report, markdown_table, write_report
from humanoid_loco.mujoco.env import MujocoG1
from humanoid_loco.mujoco.rollout import Perturbation, rollout
from humanoid_loco.policy import OnnxPolicy
from humanoid_loco.schedule import Schedule

__all__ = [
    "aggregate",
    "evaluate",
    "evaluate_schedule",
    "markdown_report",
    "markdown_table",
    "sweep",
    "write_report",
]


def evaluate_schedule(
    env: MujocoG1, policy: OnnxPolicy, schedule: Schedule, seeds: range, **kw
) -> dict[str, float]:
    return aggregate(
        evaluate(rollout(env, policy, schedule, seed=s, **kw), env.manifest) for s in seeds
    )


def sweep(
    make_env, policy: OnnxPolicy, schedule: Schedule, seeds: range, **axes: Iterable
) -> list[dict]:
    """Metrics over the cartesian product of ``Perturbation`` fields, e.g.
    ``sweep(..., mass_scale=[0.8, 1.2], friction_scale=[0.4, 1.0])``. Fresh plant per cell."""
    rows = []
    for values in itertools.product(*axes.values()):
        cell = dict(zip(axes, values, strict=True))
        m = evaluate_schedule(
            make_env(), policy, schedule, seeds, perturbation=Perturbation(**cell)
        )
        rows.append({**cell, **m})
    return rows
