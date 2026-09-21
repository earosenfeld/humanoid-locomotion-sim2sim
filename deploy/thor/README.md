# Jetson AGX Thor timing

Same `policy.onnx` and manifest, measured on the edge target.

```bash
# on the Thor (JetPack 7, onnxruntime-gpu wheel for aarch64 / CUDA 13)
pip install onnxruntime-gpu numpy
python deploy/thor/bench.py --run runs/g1_flat_12dof/export --out deploy/thor/thor-timing.json

# optional: the C++ loop on aarch64 (system clang/gcc, prebuilt MuJoCo aarch64 + onnxruntime aarch64)
cmake -S cpp -B build -G Ninja -DMUJOCO_URL=... -DORT_URL=... && cmake --build build
```

Results live in `thor-timing.json` and `x86-timing.json`; the README table is generated from them.
