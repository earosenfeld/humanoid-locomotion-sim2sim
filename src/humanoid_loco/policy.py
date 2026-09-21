"""ONNX policy runner driven entirely by the manifest."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

from humanoid_loco.manifest import PolicyManifest
from humanoid_loco.obs import RobotState, assemble_obs


class OnnxPolicy:
    """Stateful wrapper: keeps ``last_action`` so the ``actions`` obs term is exact."""

    def __init__(
        self,
        manifest: PolicyManifest,
        onnx_path: str | Path | None = None,
        providers: list[str] | None = None,
    ):
        self.manifest = manifest
        path = Path(onnx_path) if onnx_path else Path(manifest.onnx)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1  # deterministic single-thread latency, like on-robot
        self.session = ort.InferenceSession(
            str(path), opts, providers=providers or ["CPUExecutionProvider"]
        )
        (inp,) = self.session.get_inputs()
        self.input_name = inp.name
        if inp.shape[-1] != manifest.obs_dim:
            raise ValueError(
                f"ONNX expects obs dim {inp.shape[-1]}, manifest has {manifest.obs_dim}"
            )
        self.reset()

    def reset(self) -> None:
        self.last_action = np.zeros(self.manifest.num_actions, np.float32)

    def act(self, state: RobotState, command: np.ndarray) -> np.ndarray:
        """Raw policy action (pre-scale). Updates ``last_action``."""
        obs = assemble_obs(self.manifest, state, command, self.last_action)
        (out,) = self.session.run(None, {self.input_name: obs[None, :]})
        action = out[0].astype(np.float32)
        if self.manifest.action_clip is not None:
            action = np.clip(action, -self.manifest.action_clip, self.manifest.action_clip)
        self.last_action = action
        return self.last_action

    def __call__(self, state: RobotState, command: np.ndarray) -> np.ndarray:
        """Joint position targets for the PD layer."""
        return self.manifest.joint_targets(self.act(state, command))
