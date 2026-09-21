"""Closed-loop rollouts: command schedules, domain perturbations, trajectories, video."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np
import yaml

from humanoid_loco.mujoco.env import MujocoG1
from humanoid_loco.policy import OnnxPolicy


@dataclass(frozen=True)
class Segment:
    duration: float
    command: tuple[float, float, float]
    push: tuple[float, float, float] | None = None  # world-frame force on the pelvis [N]


@dataclass(frozen=True)
class Schedule:
    name: str
    segments: tuple[Segment, ...]

    @classmethod
    def from_yaml(cls, path: str | Path) -> Schedule:
        raw = yaml.safe_load(Path(path).read_text())
        return cls(raw["name"], tuple(Segment(**s) for s in raw["segments"]))

    @property
    def duration(self) -> float:
        return sum(s.duration for s in self.segments)

    def to_dict(self) -> dict:
        return {"name": self.name, "segments": [asdict(s) for s in self.segments]}

    def at(self, t: float) -> Segment:
        for seg in self.segments:
            if t < seg.duration:
                return seg
            t -= seg.duration
        return self.segments[-1]


@dataclass(frozen=True)
class Perturbation:
    """Domain shifts a real deployment sees and training did not."""

    mass_scale: float = 1.0  # pelvis mass
    friction_scale: float = 1.0  # floor sliding friction
    kp_scale: float = 1.0  # actuator stiffness (gain mismatch on hardware)
    action_delay: int = 0  # policy steps of transport delay between inference and PD target
    gyro_noise: float = 0.0  # rad/s, white noise on the IMU rate
    joint_vel_noise: float = 0.0  # rad/s, white noise on encoder-differenced velocity

    def apply(self, env: MujocoG1) -> None:
        env.model.body_mass[env.pelvis] *= self.mass_scale
        env.model.geom_friction[env.model.geom("floor").id, 0] *= self.friction_scale
        env.kp = env.kp * self.kp_scale

    def corrupt(self, state, rng: np.random.Generator):
        if self.gyro_noise:
            state.base_ang_vel = state.base_ang_vel + rng.normal(0, self.gyro_noise, 3)
        if self.joint_vel_noise:
            state.joint_vel = state.joint_vel + rng.normal(
                0, self.joint_vel_noise, state.joint_vel.size
            )
        return state


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


def rollout(
    env: MujocoG1,
    policy: OnnxPolicy,
    schedule: Schedule,
    seed: int = 0,
    perturbation: Perturbation | None = None,
    fall_height: float = 0.4,
    init_noise: float = 0.05,
    frames: list[np.ndarray] | None = None,
    frame_stride: int = 2,
) -> Trajectory:
    rng = np.random.default_rng(seed)
    state = env.reset()
    if perturbation:
        perturbation.apply(env)
    env.data.qpos[env._qadr] += rng.uniform(-init_noise, init_noise, env.manifest.num_actions)
    env.reset_from_data()
    policy.reset()
    renderer = _Renderer(env) if frames is not None else None

    traj = Trajectory()
    pert = perturbation or Perturbation()
    pending = deque([env.default_pos] * pert.action_delay)  # transport-delay line for targets
    steps = int(round(schedule.duration / env.manifest.policy_dt))
    for k in range(steps):
        seg = schedule.at(env.time)
        command = np.asarray(seg.command, np.float32)
        env.push(np.asarray(seg.push or (0.0, 0.0, 0.0)))
        pending.append(policy(pert.corrupt(state, rng), command))
        state = env.step(pending.popleft())

        traj.time.append(env.time)
        traj.command.append(command)
        traj.base_lin_vel.append(state.base_lin_vel)
        traj.base_ang_vel.append(state.base_ang_vel)
        traj.base_height.append(env.base_height)
        traj.joint_pos.append(state.joint_pos)
        traj.joint_vel.append(state.joint_vel)
        traj.action.append(policy.last_action.copy())
        traj.torque.append(env.last_torque.copy())
        if renderer and k % frame_stride == 0:
            frames.append(renderer.render())
        if env.base_height < fall_height:
            traj.fell = True
            break
    return traj


class _Renderer:
    def __init__(self, env: MujocoG1, width: int = 960, height: int = 540):
        import mujoco

        self.env = env
        self.renderer = mujoco.Renderer(env.model, height, width)
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.camera.trackbodyid = env.pelvis
        self.camera.distance, self.camera.azimuth, self.camera.elevation = 3.0, 135.0, -15.0

    def render(self) -> np.ndarray:
        self.renderer.update_scene(self.env.data, self.camera)
        return self.renderer.render()


def save_video(frames: list[np.ndarray], path: str | Path, fps: float) -> Path:
    import imageio.v3 as iio

    path = Path(path)
    iio.imwrite(path, np.stack(frames), fps=fps, **({"loop": 0} if path.suffix == ".gif" else {}))
    return path
