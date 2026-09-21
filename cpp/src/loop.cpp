#include "hls/loop.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <thread>

namespace hls {

using clock = std::chrono::steady_clock;
using json = nlohmann::json;

Schedule Schedule::load(const std::filesystem::path& p) {
  std::ifstream f(p);
  if (!f) throw std::runtime_error("cannot open schedule " + p.string());
  const auto j = json::parse(f);
  Schedule s{j.at("name"), {}};
  for (const auto& seg : j.at("segments")) {
    Segment out{seg.at("duration"), seg.at("command"), std::nullopt};
    if (seg.contains("push") && !seg["push"].is_null()) out.push = seg["push"];
    s.segments.push_back(out);
  }
  return s;
}

double Schedule::duration() const {
  double d = 0;
  for (const auto& s : segments) d += s.duration;
  return d;
}

const Segment& Schedule::at(double t) const {
  for (const auto& s : segments) {
    if (t < s.duration) return s;
    t -= s.duration;
  }
  return segments.back();
}

ControlLoop::ControlLoop(const Manifest& m, Plant& plant, OnnxPolicy& policy, LoopOptions opts)
    : m_(m), plant_(plant), policy_(policy), opts_(opts) {}

json ControlLoop::run(const Schedule& schedule) {
  const int n = m_.num_actions();
  const int substeps = static_cast<int>(std::lround(m_.policy_dt() / opts_.control_dt));
  if (std::abs(substeps * opts_.control_dt - m_.policy_dt()) > 1e-9)
    throw std::runtime_error("control_dt does not divide the policy period");
  const auto period = std::chrono::duration_cast<clock::duration>(
      std::chrono::duration<double>(opts_.control_dt));
  const long total_ticks = std::lround(schedule.duration() / opts_.control_dt);

  RobotState s;
  std::vector<double> targets(m_.default_joint_pos), tau(n), lo(n), hi(n);
  json traj = {{"time", json::array()},      {"command", json::array()},
               {"base_lin_vel", json::array()}, {"base_ang_vel", json::array()},
               {"base_height", json::array()}, {"joint_pos", json::array()},
               {"joint_vel", json::array()},   {"action", json::array()},
               {"torque", json::array()},      {"fell", false}};
  policy_.reset();
  plant_.read(s);
  policy_.act(s, schedule.segments.front().command.data());  // warm-up: JIT/allocator costs
  policy_.reset();
  compute_us_.clear(), jitter_us_.clear();
  compute_us_.reserve(total_ticks), jitter_us_.reserve(total_ticks);

  auto next = clock::now();
  double t = 0;
  for (long tick = 0; tick < total_ticks; ++tick) {
    if (opts_.realtime) {
      std::this_thread::sleep_until(next);
      jitter_us_.push_back(std::chrono::duration<double, std::micro>(clock::now() - next).count());
      next += period;
    }
    const auto t0 = clock::now();
    const Segment& seg = schedule.at(t);
    const double zero[3] = {0, 0, 0};
    plant_.push(seg.push ? seg.push->data() : zero);
    plant_.read(s);
    if (tick % substeps == 0) {
      policy_.act(s, seg.command.data());
      policy_.targets(targets);
    }
    m_.torque_bounds(s.qd.data(), lo.data(), hi.data());
    for (int i = 0; i < n; ++i)
      tau[i] = std::clamp(m_.kp[i] * (targets[i] - s.q[i]) - m_.kd[i] * s.qd[i], lo[i], hi[i]);
    plant_.apply(tau);
    plant_.step(opts_.control_dt);
    t += opts_.control_dt;
    compute_us_.push_back(std::chrono::duration<double, std::micro>(clock::now() - t0).count());

    if ((tick + 1) % substeps == 0) {  // log at policy rate, after the last tick of the period
      plant_.read(s);
      traj["time"].push_back(t);
      traj["command"].push_back(seg.command);
      traj["base_lin_vel"].push_back(s.lin_vel);
      traj["base_ang_vel"].push_back(s.gyro);
      traj["base_height"].push_back(plant_.base_height());
      traj["joint_pos"].push_back(s.q);
      traj["joint_vel"].push_back(s.qd);
      traj["action"].push_back(policy_.last_action());
      traj["torque"].push_back(tau);
      if (plant_.base_height() < opts_.fall_height) {
        traj["fell"] = true;
        break;
      }
    }
  }
  return traj;
}

static json percentiles(std::vector<double> v) {
  if (v.empty()) return nullptr;
  std::sort(v.begin(), v.end());
  auto q = [&](double p) { return v[std::min(v.size() - 1, static_cast<size_t>(p * v.size()))]; };
  return {{"p50", q(0.5)}, {"p99", q(0.99)}, {"max", v.back()}, {"n", v.size()}};
}

json ControlLoop::timing() const {
  return {{"control_dt", opts_.control_dt},
          {"realtime", opts_.realtime},
          {"compute_us", percentiles(compute_us_)},
          {"wake_jitter_us", percentiles(jitter_us_)}};
}

}  // namespace hls
