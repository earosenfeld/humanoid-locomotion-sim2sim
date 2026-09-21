#include "hls/policy.hpp"

#include <algorithm>
#include <stdexcept>

namespace hls {

static Ort::SessionOptions make_opts() {
  Ort::SessionOptions o;
  o.SetIntraOpNumThreads(1);  // deterministic single-thread latency, like on-robot
  o.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  return o;
}

OnnxPolicy::OnnxPolicy(const Manifest& m, const std::filesystem::path& onnx_path)
    : m_(m),
      env_(ORT_LOGGING_LEVEL_WARNING, "hls"),
      opts_(make_opts()),
      session_(env_, onnx_path.c_str(), opts_),
      mem_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
      obs_(m.obs_dim()),
      action_(m.num_actions()) {
  Ort::AllocatorWithDefaultOptions alloc;
  in_name_ = session_.GetInputNameAllocated(0, alloc).get();
  out_name_ = session_.GetOutputNameAllocated(0, alloc).get();
  const auto shape = session_.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
  if (shape.back() != m.obs_dim())
    throw std::runtime_error("ONNX obs dim " + std::to_string(shape.back()) +
                             " != manifest " + std::to_string(m.obs_dim()));
}

void OnnxPolicy::reset() { std::fill(action_.begin(), action_.end(), 0.f); }

const std::vector<float>& OnnxPolicy::act(const RobotState& s, const double cmd[3]) {
  assemble_obs(m_, s, cmd, action_.data(), obs_.data());
  const int64_t in_shape[2] = {1, static_cast<int64_t>(obs_.size())};
  const int64_t out_shape[2] = {1, static_cast<int64_t>(action_.size())};
  auto in = Ort::Value::CreateTensor<float>(mem_, obs_.data(), obs_.size(), in_shape, 2);
  auto out = Ort::Value::CreateTensor<float>(mem_, action_.data(), action_.size(), out_shape, 2);
  const char* in_names[] = {in_name_.c_str()};
  const char* out_names[] = {out_name_.c_str()};
  session_.Run(Ort::RunOptions{nullptr}, in_names, &in, 1, out_names, &out, 1);
  if (m_.action_clip)
    for (auto& a : action_) a = std::clamp(a, -static_cast<float>(*m_.action_clip),
                                           static_cast<float>(*m_.action_clip));
  return action_;
}

void OnnxPolicy::targets(std::vector<double>& out) const {
  out.resize(action_.size());
  for (size_t i = 0; i < out.size(); ++i)
    out[i] = m_.default_joint_pos[i] + m_.action_scale[i] * action_[i];
}

}  // namespace hls
