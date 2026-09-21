#include "hls/plant.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

namespace hls {

static int id_or_throw(const mjModel* m, mjtObj type, const std::string& name) {
  const int id = mj_name2id(m, type, name.c_str());
  if (id < 0) throw std::runtime_error("model has no " + name);
  return id;
}

MujocoPlant::MujocoPlant(const Manifest& m, const std::filesystem::path& mjb_path) : m_(m) {
  mdl_ = mj_loadModel(mjb_path.c_str(), nullptr);
  if (!mdl_) throw std::runtime_error("cannot load " + mjb_path.string());
  d_ = mj_makeData(mdl_);
  for (const auto& name : m.joint_names) {
    const int j = id_or_throw(mdl_, mjOBJ_JOINT, name);
    qadr_.push_back(mdl_->jnt_qposadr[j]);
    vadr_.push_back(mdl_->jnt_dofadr[j]);
    act_.push_back(id_or_throw(mdl_, mjOBJ_ACTUATOR, name));
  }
  gyro_ = mdl_->sensor_adr[id_or_throw(mdl_, mjOBJ_SENSOR, "imu_gyro")];
  quat_ = mdl_->sensor_adr[id_or_throw(mdl_, mjOBJ_SENSOR, "imu_quat")];
  vel_ = mdl_->sensor_adr[id_or_throw(mdl_, mjOBJ_SENSOR, "imu_vel")];
  pelvis_ = id_or_throw(mdl_, mjOBJ_BODY, "pelvis");
  for (int g = 0; g < mdl_->ngeom; ++g)
    if (mdl_->geom_type[g] == mjGEOM_SPHERE &&
        std::strstr(mj_id2name(mdl_, mjOBJ_BODY, mdl_->geom_bodyid[g]), "ankle_roll"))
      foot_geoms_.push_back(g);
  reset();
}

MujocoPlant::~MujocoPlant() {
  mj_deleteData(d_);
  mj_deleteModel(mdl_);
}

void MujocoPlant::reset() {
  mj_resetDataKeyframe(mdl_, d_, id_or_throw(mdl_, mjOBJ_KEY, "stand"));
  for (int i = 0; i < m_.num_actions(); ++i) d_->qpos[qadr_[i]] = m_.default_joint_pos[i];
  for (const auto& [name, h] : m_.held_joints) {
    const int a = mj_name2id(mdl_, mjOBJ_ACTUATOR, name.c_str());
    if (a < 0) continue;  // training asset joints absent from this model (e.g. hands)
    d_->qpos[mdl_->jnt_qposadr[id_or_throw(mdl_, mjOBJ_JOINT, name)]] = h.pos;
    d_->ctrl[a] = h.pos;
  }
  std::fill(d_->qvel, d_->qvel + mdl_->nv, 0.0);
  mj_forward(mdl_, d_);
  double lowest = 1e9;  // settle the lowest foot contact sphere onto the floor
  for (int g : foot_geoms_)
    lowest = std::min(lowest, d_->geom_xpos[3 * g + 2] - mdl_->geom_size[3 * g]);
  d_->qpos[2] += 0.002 - lowest;
  mj_forward(mdl_, d_);
}

void MujocoPlant::read(RobotState& s) {
  const int n = m_.num_actions();
  s.q.resize(n), s.qd.resize(n);
  for (int i = 0; i < n; ++i) s.q[i] = d_->qpos[qadr_[i]], s.qd[i] = d_->qvel[vadr_[i]];
  std::copy_n(d_->sensordata + quat_, 4, s.quat.begin());
  std::copy_n(d_->sensordata + gyro_, 3, s.gyro.begin());
  std::copy_n(d_->sensordata + vel_, 3, s.lin_vel.begin());
}

void MujocoPlant::apply(const std::vector<double>& tau) {
  for (int i = 0; i < m_.num_actions(); ++i) d_->ctrl[act_[i]] = tau[i];
}

void MujocoPlant::step(double dt) {
  mdl_->opt.timestep = dt;
  mj_step(mdl_, d_);
}

void MujocoPlant::push(const double f[3]) { std::copy_n(f, 3, d_->xfrc_applied + 6 * pelvis_); }

}  // namespace hls
