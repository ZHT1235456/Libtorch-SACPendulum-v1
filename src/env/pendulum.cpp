#include "env/pendulum.h"

PendulumEnv::PendulumEnv()
: theta_(0.0), theta_dot_(0.0), step_count_(0),
  rng_(std::random_device{}()),
  uni_theta_(-M_PI, M_PI),
  uni_theta_dot_(-1.0, 1.0) // 有的实现会更大范围，你可按需改
{}

std::array<double, 3> PendulumEnv::reset(unsigned int seed) {
    rng_.seed(seed);
    theta_ = uni_theta_(rng_);
    theta_dot_ = uni_theta_dot_(rng_);
    step_count_ = 0;
    return obs();
}

StepResult PendulumEnv::step(double action) {
    // 1) 限幅
    action = clip(action, -max_torque_, max_torque_);

    // 2) 动力学（与常见实现对齐）
    // theta_ddot = -3*g/(2*l)*sin(theta + pi) + 3/(m*l^2) * u
    double theta_ddot = -3.0 * g_ / (2.0 * l_) * std::sin(theta_ + M_PI) + 3.0 / (m_ * l_ * l_) * action;

    // 3) 积分/限幅
    theta_dot_ += theta_ddot * dt_;
    theta_dot_ = clip(theta_dot_, -max_speed_, max_speed_);
    theta_ += theta_dot_ * dt_;
    theta_ = wrap_to_pi(theta_);

    step_count_++;

    // 4) 代价（负回报）
    // cost = theta^2 + 0.1*theta_dot^2 + 0.001*u^2
    double cost = theta_ * theta_ + 0.1 * theta_dot_ * theta_dot_ + 0.001 * action * action;
    double reward = -cost;

    // Pendulum-v1 通常没有终止条件，这里始终 false
    StepResult result{obs(), reward, false};
    return result;
}

std::array<double, 3> PendulumEnv::obs() const {
    return { std::cos(theta_), std::sin(theta_), theta_dot_ };
}

double PendulumEnv::wrap_to_pi(double x) {
    // 把角度 wrap 到 [-pi, pi]
    while (x <= -M_PI) x += 2.0 * M_PI;
    while (x >   M_PI) x -= 2.0 * M_PI;
    return x;
}

double PendulumEnv::clip(double x, double lo, double hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}
