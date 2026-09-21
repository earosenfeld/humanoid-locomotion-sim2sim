// hls_loop --run <dir with policy.json, policy.onnx, scene.mjb> --schedule <json>
//          [--seconds-limit N] [--control-dt 0.001] [--no-realtime] [--out traj.json] [--timing t.json]
#include <fstream>
#include <iostream>
#include <string>

#include "hls/loop.hpp"

int main(int argc, char** argv) {
  std::filesystem::path run, schedule_path, out = "trajectory.json", timing = "timing.json";
  hls::LoopOptions opts;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&] { return std::string(argv[++i]); };
    if (a == "--run") run = next();
    else if (a == "--schedule") schedule_path = next();
    else if (a == "--control-dt") opts.control_dt = std::stod(next());
    else if (a == "--no-realtime") opts.realtime = false;
    else if (a == "--out") out = next();
    else if (a == "--timing") timing = next();
    else { std::cerr << "unknown arg " << a << "\n"; return 2; }
  }
  if (run.empty() || schedule_path.empty()) {
    std::cerr << "usage: hls_loop --run <dir> --schedule <json> [--no-realtime]\n";
    return 2;
  }
  const auto m = hls::Manifest::load(run / "policy.json");
  hls::OnnxPolicy policy(m, run / m.onnx);
  hls::MujocoPlant plant(m, run / "scene.mjb");
  hls::ControlLoop loop(m, plant, policy, opts);
  const auto schedule = hls::Schedule::load(schedule_path);

  const auto traj = loop.run(schedule);
  std::ofstream(out) << traj.dump() << "\n";
  const auto t = loop.timing();
  std::ofstream(timing) << t.dump(2) << "\n";
  std::cout << schedule.name << ": " << traj["time"].size() << " policy steps, fell="
            << traj["fell"] << ", compute p50/p99/max us = " << t["compute_us"]["p50"] << "/"
            << t["compute_us"]["p99"] << "/" << t["compute_us"]["max"] << "\n";
  return 0;
}
