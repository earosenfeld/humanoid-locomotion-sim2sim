// Observation terms, mirroring humanoid_loco/obs.py term for term.
#pragma once
#include <array>
#include <vector>

#include "hls/manifest.hpp"

namespace hls {

struct RobotState {
  std::array<double, 4> quat{1, 0, 0, 0};  // w x y z, world -> base
  std::array<double, 3> gyro{};            // base-frame angular velocity
  std::array<double, 3> lin_vel{};         // base-frame linear velocity (privileged / metrics)
  std::vector<double> q, qd;               // manifest (policy) order
};

void quat_rotate_inverse(const double q[4], const double v[3], double out[3]);

// Writes manifest.obs_dim() floats into `out`, in manifest order, with scale and clip applied.
void assemble_obs(const Manifest& m, const RobotState& s, const double cmd[3],
                  const float* last_action, float* out);

}  // namespace hls
