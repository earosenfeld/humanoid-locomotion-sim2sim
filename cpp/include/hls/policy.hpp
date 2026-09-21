// onnxruntime policy with preallocated buffers: one Run() per policy step, no allocations.
#pragma once
#include <onnxruntime_cxx_api.h>

#include <filesystem>
#include <vector>

#include "hls/manifest.hpp"
#include "hls/obs.hpp"

namespace hls {

class OnnxPolicy {
 public:
  OnnxPolicy(const Manifest& m, const std::filesystem::path& onnx_path);
  void reset();
  // Raw (pre-scale) action; also stored as last_action for the next observation.
  const std::vector<float>& act(const RobotState& s, const double cmd[3]);
  // PD targets for the current last_action: default + scale * action.
  void targets(std::vector<double>& out) const;
  const std::vector<float>& last_action() const { return action_; }

 private:
  const Manifest& m_;
  Ort::Env env_;
  Ort::SessionOptions opts_;
  Ort::Session session_;
  Ort::MemoryInfo mem_;
  std::string in_name_, out_name_;
  std::vector<float> obs_, action_;
};

}  // namespace hls
