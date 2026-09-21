import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))  # makes `synthetic` importable
from synthetic import g1_manifest, write_synthetic_policy  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
MENAGERIE = REPO / "third_party" / "mujoco_menagerie" / "unitree_g1"


@pytest.fixture(scope="session")
def run_dir(tmp_path_factory) -> Path:
    """A synthetic run directory: policy.json + policy.onnx that holds the default pose."""
    d = tmp_path_factory.mktemp("run")
    m = g1_manifest()
    write_synthetic_policy(d / "policy.onnx", m)
    m.save(d / "policy.json")
    return d


@pytest.fixture(scope="session")
def manifest(run_dir):
    from humanoid_loco.manifest import PolicyManifest

    return PolicyManifest.load(run_dir / "policy.json")


@pytest.fixture(scope="session")
def menagerie() -> Path:
    if not MENAGERIE.exists():
        pytest.skip("mujoco_menagerie not checked out (see README)")
    return MENAGERIE


@pytest.fixture
def env(manifest, menagerie):
    from humanoid_loco.mujoco.env import MujocoG1

    return MujocoG1(manifest, menagerie)
