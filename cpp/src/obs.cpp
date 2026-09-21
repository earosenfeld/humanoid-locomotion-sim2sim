#include "hls/obs.hpp"

#include <algorithm>
#include <stdexcept>

namespace hls {

void quat_rotate_inverse(const double q[4], const double v[3], double out[3]) {
  const double w = q[0], x = q[1], y = q[2], z = q[3];
  const double dot = x * v[0] + y * v[1] + z * v[2];
  const double cx = y * v[2] - z * v[1], cy = z * v[0] - x * v[2], cz = x * v[1] - y * v[0];
  const double a = 2 * w * w - 1;
  out[0] = v[0] * a - 2 * w * cx + 2 * x * dot;
  out[1] = v[1] * a - 2 * w * cy + 2 * y * dot;
  out[2] = v[2] * a - 2 * w * cz + 2 * z * dot;
}

void assemble_obs(const Manifest& m, const RobotState& s, const double cmd[3],
                  const float* last_action, float* out) {
  const int n = m.num_actions();
  float* p = out;
  for (const auto& t : m.obs_terms) {
    float* start = p;
    if (t.name == "base_ang_vel") {
      for (double v : s.gyro) *p++ = static_cast<float>(v);
    } else if (t.name == "base_lin_vel") {
      for (double v : s.lin_vel) *p++ = static_cast<float>(v);
    } else if (t.name == "projected_gravity") {
      const double g[3] = {0, 0, -1};
      double r[3];
      quat_rotate_inverse(s.quat.data(), g, r);
      for (double v : r) *p++ = static_cast<float>(v);
    } else if (t.name == "velocity_commands") {
      for (int i = 0; i < 3; ++i) *p++ = static_cast<float>(cmd[i]);
    } else if (t.name == "joint_pos") {
      for (int i = 0; i < n; ++i) *p++ = static_cast<float>(s.q[i] - m.default_joint_pos[i]);
    } else if (t.name == "joint_vel") {
      for (int i = 0; i < n; ++i) *p++ = static_cast<float>(s.qd[i]);
    } else if (t.name == "actions") {
      for (int i = 0; i < n; ++i) *p++ = last_action[i];
    } else {
      throw std::runtime_error("unknown observation term " + t.name);
    }
    if (p - start != t.dim) throw std::runtime_error("obs term size mismatch: " + t.name);
    for (float* v = start; v != p; ++v) {
      *v *= t.scale;
      if (t.clip) *v = std::clamp(*v, t.clip->first, t.clip->second);
    }
  }
}

}  // namespace hls
