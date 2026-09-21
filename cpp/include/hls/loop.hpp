// Deterministic 1 kHz control loop: PD every tick, policy every `substeps` ticks, same thread.
#pragma once
#include <nlohmann/json.hpp>

#include <array>
#include <filesystem>
#include <optional>
#include <vector>

#include "hls/manifest.hpp"
#include "hls/plant.hpp"
#include "hls/policy.hpp"

namespace hls {

struct Segment {
  double duration;
  std::array<double, 3> command;
  std::optional<std::array<double, 3>> push;
};

struct Schedule {
  std::string name;
  std::vector<Segment> segments;
  static Schedule load(const std::filesystem::path& json_path);
  double duration() const;
  const Segment& at(double t) const;
};

struct LoopOptions {
  double control_dt = 0.001;
  bool realtime = true;  // pace ticks on the monotonic clock; false = run as fast as possible
  double fall_height = 0.4;
};

// Runs the schedule; returns the policy-rate trajectory (same keys as the Python Trajectory)
// plus per-tick timing so Python can score both with one evaluate().
class ControlLoop {
 public:
  ControlLoop(const Manifest& m, Plant& plant, OnnxPolicy& policy, LoopOptions opts);
  nlohmann::json run(const Schedule& schedule);
  nlohmann::json timing() const;  // compute-time and wake-jitter percentiles [us]

 private:
  const Manifest& m_;
  Plant& plant_;
  OnnxPolicy& policy_;
  LoopOptions opts_;
  std::vector<double> compute_us_, jitter_us_;
};

}  // namespace hls
