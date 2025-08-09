#pragma once
#include <array>
#include <random>
#include <cmath>

struct StepResult {
    std::array<double, 3> state; // [cos(theta), sin(theta), theta_dot]
    double reward;
    bool done; // Pendulum-v1 通常没有 done，这里始终 false
};

class PendulumEnv {
public:
    PendulumEnv();

    // 重置并返回观测
    std::array<double, 3> reset(unsigned int seed = std::random_device{}());

    // 执行一步，action 为连续标量（不经缩放），范围 [-2, 2]
    StepResult step(double action);

    // 每个 episode 的最大步数（常用 200）
    int max_steps_per_episode() const { return 200; }

private:
    // 动力学常数（与 Gymnasium 对齐）
    const double g_ = 10.0;
    const double m_ = 1.0;
    const double l_ = 1.0;
    const double dt_ = 0.05;
    const double max_torque_ = 2.0;
    const double max_speed_ = 8.0;

    // 状态：使用角度/角速度内部表示
    double theta_;   // [-pi, pi]
    double theta_dot_;

    // 步内计数
    int step_count_;

    // 随机
    std::mt19937 rng_;
    std::uniform_real_distribution<double> uni_theta_;     // [-pi, pi]
    std::uniform_real_distribution<double> uni_theta_dot_; // [-1, 1] 可按需调

    // 工具
    static double wrap_to_pi(double x);
    static double clip(double x, double lo, double hi);

    // 观测
    std::array<double, 3> obs() const;
};
