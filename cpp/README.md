# hls_loop — 1 kHz control loop

`manifest -> ONNX policy -> PD -> plant`, one thread, no allocations in the loop. The MuJoCo
plant is the sim-to-sim target; a robot is a second `Plant`.

## Build

```bash
cmake -S cpp -B build -G Ninja && cmake --build build          # any C++20 compiler
scripts/setup-cpp-toolchain-wsl.sh                              # no root? zig + cmake/ninja wheels
cmake -S cpp -B build -G Ninja --toolchain cpp/cmake/zig-toolchain.cmake
```

CMake fetches nlohmann/json, prebuilt MuJoCo and onnxruntime (Linux x86_64 pinned to the
Python versions, so `scene.mjb` from `hls prepare-cpp` loads unchanged).

## Run

```bash
hls prepare-cpp --run <export dir> --commands configs/commands/walk.yaml   # scene.mjb + schedule json
build/hls_loop --run <export dir> --schedule <export dir>/schedule-walk.json [--no-realtime]
hls score --run <export dir> trajectory.json                                # same metrics as Python
```

`--no-realtime` runs as fast as possible (parity tests); without it ticks are paced on the
monotonic clock and `timing.json` records compute time and wake jitter percentiles.

## Porting to a robot

Implement `hls::Plant` (`plant.hpp`) against the SDK: `read()` fills IMU quaternion, gyro and
joint state **in manifest order** (map by joint name once, at construction); `apply()` sends
torques, or `kp/kd/q_target` if the driver runs its own PD; `step()` waits for the next sample.
Everything else, including the DC-motor torque bounds, stays as is.
