// Policy manifest: the same JSON the Python side writes; nothing here is typed by hand.
#pragma once
#include <filesystem>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace hls {

struct ObsTerm {
  std::string name;
  int dim;
  float scale;
  std::optional<std::pair<float, float>> clip;
};

struct HeldJoint {
  double pos, kp, kd;
};

struct Manifest {
  std::vector<std::string> joint_names;
  std::vector<double> default_joint_pos, kp, kd, torque_limit, action_scale;
  std::optional<std::vector<double>> saturation_effort, velocity_limit;
  std::optional<double> action_clip;
  std::vector<ObsTerm> obs_terms;
  double physics_dt = 0;
  int decimation = 0;
  std::vector<std::pair<std::string, HeldJoint>> held_joints;
  std::string onnx;

  static Manifest load(const std::filesystem::path& json_path);

  int num_actions() const { return static_cast<int>(joint_names.size()); }
  int obs_dim() const;
  double policy_dt() const { return physics_dt * decimation; }
  // Isaac Lab DCMotor clip: available torque shrinks linearly with joint speed.
  void torque_bounds(const double* qd, double* lo, double* hi) const;
};

}  // namespace hls
