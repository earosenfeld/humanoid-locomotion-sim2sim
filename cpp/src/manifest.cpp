#include "hls/manifest.hpp"

#include <algorithm>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace hls {

Manifest Manifest::load(const std::filesystem::path& json_path) {
  std::ifstream f(json_path);
  if (!f) throw std::runtime_error("cannot open manifest " + json_path.string());
  const auto j = nlohmann::json::parse(f);
  Manifest m;
  j.at("joint_names").get_to(m.joint_names);
  j.at("default_joint_pos").get_to(m.default_joint_pos);
  j.at("kp").get_to(m.kp);
  j.at("kd").get_to(m.kd);
  j.at("torque_limit").get_to(m.torque_limit);
  j.at("action_scale").get_to(m.action_scale);
  if (!j.at("saturation_effort").is_null()) {
    m.saturation_effort = j["saturation_effort"].get<std::vector<double>>();
    m.velocity_limit = j["velocity_limit"].get<std::vector<double>>();
  }
  if (!j.at("action_clip").is_null()) m.action_clip = j["action_clip"].get<double>();
  for (const auto& t : j.at("obs_terms")) {
    ObsTerm term{t.at("name"), t.at("dim"), t.at("scale"), {}};
    if (!t.at("clip").is_null()) term.clip = {t["clip"][0].get<float>(), t["clip"][1].get<float>()};
    m.obs_terms.push_back(std::move(term));
  }
  j.at("physics_dt").get_to(m.physics_dt);
  j.at("decimation").get_to(m.decimation);
  for (const auto& [name, h] : j.at("held_joints").items())
    m.held_joints.emplace_back(name, HeldJoint{h.at("pos"), h.at("kp"), h.at("kd")});
  j.at("onnx").get_to(m.onnx);

  const auto n = m.joint_names.size();
  for (const auto* v : {&m.default_joint_pos, &m.kp, &m.kd, &m.torque_limit, &m.action_scale})
    if (v->size() != n) throw std::runtime_error("manifest: per-joint array length mismatch");
  return m;
}

int Manifest::obs_dim() const {
  int d = 0;
  for (const auto& t : obs_terms) d += t.dim;
  return d;
}

void Manifest::torque_bounds(const double* qd, double* lo, double* hi) const {
  for (int i = 0; i < num_actions(); ++i) {
    const double lim = torque_limit[i];
    if (!saturation_effort) {
      lo[i] = -lim;
      hi[i] = lim;
      continue;
    }
    const double sat = (*saturation_effort)[i], vmax = (*velocity_limit)[i];
    hi[i] = std::clamp(sat * (1.0 - qd[i] / vmax), 0.0, lim);
    lo[i] = std::clamp(-sat * (1.0 + qd[i] / vmax), -lim, 0.0);
  }
}

}  // namespace hls
