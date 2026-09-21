import numpy as np

from humanoid_loco.mujoco.rollout import Perturbation, Schedule, Segment, rollout
from humanoid_loco.policy import OnnxPolicy


def test_action_delay_shifts_targets(env, manifest, run_dir):
    policy = OnnxPolicy(manifest, run_dir / "policy.onnx")
    schedule = Schedule("t", (Segment(0.2, (0.0, 0.0, 0.0)),))
    env.model.opt.gravity[:] = 0.0
    env.data.qpos[2] += 0.5
    env.data.qpos[env._qadr] += 0.2  # start away from the default pose
    env.reset_from_data()
    base = rollout(env, policy, schedule, init_noise=0.0, fall_height=0.0).arrays()["joint_pos"]
    env.reset()
    env.data.qpos[2] += 0.5
    env.data.qpos[env._qadr] += 0.2
    env.reset_from_data()
    delayed = rollout(
        env,
        policy,
        schedule,
        init_noise=0.0,
        fall_height=0.0,
        perturbation=Perturbation(action_delay=3),
    ).arrays()["joint_pos"]
    # with delay the PD holds the default pose for 3 steps before the (identical) targets arrive
    assert np.abs(delayed[0] - base[0]).max() < 1e-9
    assert (
        np.abs(delayed[:3] - 0.2 - env.default_pos).max()
        < np.abs(base[:3] - 0.2 - env.default_pos).max() + 1e-9
    )


def test_noise_corrupts_state_not_plant(manifest):
    from humanoid_loco.obs import RobotState

    s = RobotState(np.array([1, 0, 0, 0.0]), np.zeros(3), np.zeros(12), np.zeros(12))
    out = Perturbation(gyro_noise=0.1, joint_vel_noise=0.5).corrupt(s, np.random.default_rng(0))
    assert out.base_ang_vel.std() > 0 and out.joint_vel.std() > 0 and out.joint_pos.std() == 0
