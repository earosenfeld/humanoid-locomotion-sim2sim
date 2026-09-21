import mujoco
import numpy as np
import pytest

from humanoid_loco.mujoco.metrics import aggregate, evaluate
from humanoid_loco.mujoco.rollout import Perturbation, Schedule, Segment, rollout
from humanoid_loco.mujoco.scene import build_model
from humanoid_loco.policy import OnnxPolicy


def test_scene_actuators_cover_manifest_exactly(manifest, menagerie):
    m = build_model(manifest, menagerie, timestep=0.001)
    names = [m.actuator(i).name for i in range(m.nu)]
    assert names[:12] == manifest.joint_names  # policy joints first, in policy order
    assert set(names[12:]) == set(manifest.held_joints)
    assert m.opt.timestep == 0.001
    np.testing.assert_allclose(m.actuator_ctrlrange[:12, 1], manifest.torque_limit)
    dofs = [m.jnt_dofadr[m.joint(n).id] for n in manifest.joint_names]
    np.testing.assert_allclose(m.dof_armature[dofs], manifest.armature)  # Menagerie ships 0.01
    np.testing.assert_allclose(m.dof_frictionloss[dofs], manifest.joint_friction)


def test_reset_puts_feet_on_floor_and_imu_upright(env):
    s = env.reset()
    np.testing.assert_allclose(s.base_quat, [1, 0, 0, 0], atol=1e-6)
    np.testing.assert_allclose(s.joint_pos, env.manifest.default_joint_pos, atol=1e-9)
    mujoco.mj_forward(env.model, env.data)
    lowest = min(env.data.geom_xpos[g, 2] - env.model.geom_size[g, 0] for g in env._foot_geoms)
    assert 0 <= lowest < 0.005


def test_joint_order_is_remapped_by_name(env):
    """Manifest order interleaves left/right; MJCF is left leg then right leg."""
    mj_order = [env.model.joint(i).name for i in range(1, 13)]
    assert mj_order != env.manifest.joint_names
    for k, name in enumerate(env.manifest.joint_names):
        assert env._qadr[k] == env.model.joint(name).qposadr[0]


def test_pd_tracks_targets_at_control_rate(env):
    assert env.substeps == 20  # 50 Hz policy, 1 kHz PD
    env.model.opt.gravity[:] = 0.0  # free-floating: no contact or load to fight
    env.data.qpos[2] += 0.5
    target = env.default_pos + 0.1
    for _ in range(50):  # 1 s with the manifest's (deployment) gains
        s = env.step(target)
    np.testing.assert_allclose(s.joint_pos, target, atol=0.01)
    assert np.all(np.abs(env.last_torque) <= env.manifest.torque_limit)


def test_control_dt_must_divide_policy_dt(manifest, menagerie):
    from humanoid_loco.mujoco.env import MujocoG1

    with pytest.raises(ValueError, match="divide"):
        MujocoG1(manifest, menagerie, control_dt=0.003)


def test_rollout_records_trajectory_and_detects_fall(env, manifest, run_dir):
    policy = OnnxPolicy(manifest, run_dir / "policy.onnx")
    schedule = Schedule("t", (Segment(1.0, (0.3, 0.0, 0.0)), Segment(1.0, (0.0, 0.0, 0.0))))
    traj = rollout(env, policy, schedule, seed=1, fall_height=2.0)  # forces fall at step 1
    assert traj.fell and len(traj.time) == 1
    traj = rollout(env, policy, schedule, seed=1, fall_height=0.0)
    a = traj.arrays()
    assert a["command"].shape == (100, 3) and a["torque"].shape == (100, 12)
    assert np.all(a["command"][:50, 0] == 0.3) and np.all(a["command"][50:, 0] == 0.0)
    assert np.isfinite(a["joint_pos"]).all()


def test_perturbation_is_idempotent(env):
    m0 = env.model.body_mass[env.pelvis]
    f0 = env.model.geom_friction[:, 0].copy()
    for _ in range(3):  # repeated application must not compound
        Perturbation(mass_scale=1.5, friction_scale=0.5, kp_scale=2.0).apply(env)
    assert env.model.body_mass[env.pelvis] == pytest.approx(1.5 * m0)
    np.testing.assert_allclose(env.model.geom_friction[:, 0], 0.5 * f0)
    assert env.kp[0] == pytest.approx(2.0 * env.manifest.kp[0])
    Perturbation().apply(env)
    assert env.model.body_mass[env.pelvis] == pytest.approx(m0)


def test_metrics_on_perfect_tracking(manifest):
    from humanoid_loco.mujoco.rollout import Trajectory

    t = Trajectory()
    for k in range(10):
        t.time.append(0.02 * (k + 1))
        t.command.append(np.array([0.5, 0.0, 0.2]))
        t.base_lin_vel.append(np.array([0.5, 0.0, 0.0]))
        t.base_ang_vel.append(np.array([0.0, 0.0, 0.2]))
        t.base_height.append(0.75)
        t.joint_pos.append(np.zeros(12))
        t.joint_vel.append(np.zeros(12))
        t.action.append(np.zeros(12))
        t.torque.append(np.zeros(12))
    m = evaluate(t, manifest)
    assert m["vx_rmse"] == m["vy_rmse"] == m["yaw_rate_rmse"] == 0.0
    assert m["fell"] == 0.0 and m["survived_s"] == pytest.approx(0.2)
    agg = aggregate([m, {**m, "fell": 1.0}])
    assert agg["fall_rate"] == 0.5 and agg["n_runs"] == 2
