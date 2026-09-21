import numpy as np
import pytest
from synthetic import g1_manifest, write_synthetic_policy

from humanoid_loco.obs import RobotState
from humanoid_loco.policy import OnnxPolicy


def _state(manifest):
    return RobotState(
        np.array([1, 0, 0, 0.0]), np.zeros(3), np.array(manifest.default_joint_pos), np.zeros(12)
    )


def test_zero_policy_returns_default_pose(manifest, run_dir):
    policy = OnnxPolicy(manifest, run_dir / "policy.onnx")
    np.testing.assert_allclose(policy(_state(manifest), np.zeros(3)), manifest.default_joint_pos)
    assert policy.last_action.shape == (12,) and policy.last_action.dtype == np.float32


def test_last_action_feeds_back_and_clip_applies(tmp_path):
    m = g1_manifest("policy.onnx")
    m.action_clip = 0.1
    write_synthetic_policy(tmp_path / "policy.onnx", m, gain=5.0)
    policy = OnnxPolicy(m, tmp_path / "policy.onnx")
    a1 = policy.act(_state(m), np.array([1.0, 0, 0]))
    a2 = policy.act(_state(m), np.array([1.0, 0, 0]))
    assert np.abs(a1).max() <= 0.1
    assert not np.allclose(a1, a2)  # same state, different last_action -> different output


def test_obs_dim_mismatch_is_rejected(manifest, run_dir):
    bad = g1_manifest()
    bad.obs_terms = bad.obs_terms[1:]
    with pytest.raises(ValueError, match="obs dim"):
        OnnxPolicy(bad, run_dir / "policy.onnx")
