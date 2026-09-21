import numpy as np
import pytest

from humanoid_loco.manifest import ObsTerm
from humanoid_loco.obs import RobotState, assemble_obs, projected_gravity, quat_rotate_inverse


def _quat(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    return np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * axis])


def test_projected_gravity_upright_and_tilted():
    np.testing.assert_allclose(projected_gravity(np.array([1, 0, 0, 0.0])), [0, 0, -1])
    # pitched nose-down by 90 deg about +y: gravity now points along the body +x axis
    np.testing.assert_allclose(
        projected_gravity(_quat([0, 1, 0], np.pi / 2)), [1, 0, 0], atol=1e-12
    )


def test_quat_rotate_inverse_matches_matrix():
    q = _quat([1, 2, 3], 0.7)
    w, x, y, z = q
    r = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    v = np.array([0.3, -1.2, 2.0])
    np.testing.assert_allclose(quat_rotate_inverse(q, v), r.T @ v, atol=1e-12)


def test_assemble_follows_manifest_order_scale_clip(manifest):
    n = manifest.num_actions
    state = RobotState(
        base_quat=np.array([1, 0, 0, 0.0]),
        base_ang_vel=np.array([0.1, 0.2, 0.3]),
        joint_pos=np.array(manifest.default_joint_pos) + 0.5,
        joint_vel=np.full(n, 2.0),
    )
    cmd, last = np.array([1.0, 0.0, 0.5]), np.arange(n, dtype=float)
    obs = assemble_obs(manifest, state, cmd, last)
    s = manifest.obs_slices()
    np.testing.assert_allclose(obs[s["base_ang_vel"]], [0.1, 0.2, 0.3], rtol=1e-6)
    np.testing.assert_allclose(obs[s["joint_pos"]], 0.5, rtol=1e-6)
    np.testing.assert_allclose(obs[s["actions"]], last)

    scaled = [ObsTerm("joint_vel", n, scale=0.05, clip=(-0.05, 0.05))]
    np.testing.assert_allclose(assemble_obs(manifest, state, cmd, last, group=scaled), 0.05)


def test_unknown_term_is_an_error(manifest):
    state = RobotState(np.array([1, 0, 0, 0.0]), np.zeros(3), np.zeros(12), np.zeros(12))
    with pytest.raises(KeyError, match="height_scan"):
        assemble_obs(
            manifest, state, np.zeros(3), np.zeros(12), group=[ObsTerm("height_scan", 187)]
        )
