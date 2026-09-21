// The plant behind the control loop. MujocoPlant is the sim-to-sim target; a hardware adapter
// implements the same five calls against the robot SDK.
#pragma once
#include <mujoco/mujoco.h>

#include <filesystem>
#include <vector>

#include "hls/manifest.hpp"
#include "hls/obs.hpp"

namespace hls {

class Plant {
 public:
  virtual ~Plant() = default;
  virtual void read(RobotState& s) = 0;                 // IMU + joints, manifest order
  virtual void apply(const std::vector<double>& tau) = 0;  // joint torques, manifest order
  virtual void step(double dt) = 0;                     // advance one control tick
  virtual double base_height() const = 0;
  virtual void push(const double force_world[3]) = 0;   // zero to clear
};

class MujocoPlant : public Plant {
 public:
  // Loads the binary model that `hls prepare-cpp` saved from the Python scene builder.
  MujocoPlant(const Manifest& m, const std::filesystem::path& mjb_path);
  ~MujocoPlant() override;
  void reset();
  void read(RobotState& s) override;
  void apply(const std::vector<double>& tau) override;
  void step(double dt) override;
  double base_height() const override { return d_->xpos[3 * pelvis_ + 2]; }
  void push(const double f[3]) override;

 private:
  const Manifest& m_;
  mjModel* mdl_ = nullptr;
  mjData* d_ = nullptr;
  std::vector<int> qadr_, vadr_, act_;
  int gyro_ = 0, quat_ = 0, vel_ = 0, pelvis_ = 0;
  std::vector<int> foot_geoms_;
};

}  // namespace hls
