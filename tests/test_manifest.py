import numpy as np
import pytest

from humanoid_loco.manifest import PolicyManifest


def test_roundtrip_preserves_everything(manifest, run_dir):
    again = PolicyManifest.load(manifest.save(run_dir / "copy.json"))
    assert again == manifest


def test_derived_quantities(manifest):
    assert manifest.num_actions == 12
    assert manifest.obs_dim == 3 + 3 + 3 + 12 * 3
    assert manifest.policy_dt == pytest.approx(0.02)
    slices = manifest.obs_slices()
    assert slices["actions"] == slice(manifest.obs_dim - 12, manifest.obs_dim)


def test_joint_targets_apply_scale_and_offset(manifest):
    a = np.ones(12)
    np.testing.assert_allclose(
        manifest.joint_targets(a),
        np.array(manifest.default_joint_pos) + np.array(manifest.action_scale),
    )


def test_dc_motor_bounds_shrink_with_speed(manifest):
    lo0, hi0 = manifest.torque_bounds(np.zeros(12))
    lo1, hi1 = manifest.torque_bounds(np.full(12, 15.0))
    assert np.all(hi1 <= hi0) and np.all(lo1 <= lo0) and np.all(hi0 <= manifest.torque_limit)
    knee = manifest.joint_names.index("left_knee_joint")  # vmax 20 rad/s: 15 rad/s saturates
    assert hi1[knee] < hi0[knee]


def test_validate_rejects_length_mismatch(manifest):
    bad = PolicyManifest(**{**manifest.__dict__, "kp": manifest.kp[:-1]})
    with pytest.raises(ValueError, match="kp"):
        bad.validate()
