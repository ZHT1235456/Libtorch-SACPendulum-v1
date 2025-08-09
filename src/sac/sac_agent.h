#pragma once
#include <torch/torch.h>
#include <string>
#include <filesystem>
#include "sac/actor.h"
#include "sac/critic.h"
#include "utils/replay_buffer.h"

struct SACConfig {
    int obs_dim = 3;
    int act_dim = 1;
    double act_limit = 2.0;   // Pendulum [-2,2]
    double gamma = 0.99;
    double tau = 0.005;
    int hidden = 256;
    int batch_size = 256;
    double lr = 3e-4;
    bool autotune_alpha = true;
    double target_entropy = -1.0; // for 1D action
    int updates_per_step = 1;
};

class SACAgent {
public:
    SACAgent(const SACConfig& cfg, torch::Device device);

    double select_action_train(const torch::Tensor& state_cpu);
    double select_action_eval(const torch::Tensor& state_cpu);

    void update(ReplayBuffer& buf);
    void soft_update(double tau);

    // --- Checkpoint I/O ---
    void save(const std::string& dir);                 // 保存网络+alpha（含target）
    bool load(const std::string& dir, torch::Device);  // 读取，返回是否成功

    double alpha() const { return alpha_value_.item<double>(); }

private:
    SACConfig cfg_;
    torch::Device device_;

    Actor  actor_;
    Critic q1_, q2_;
    Critic tq1_, tq2_; // target

    torch::optim::Adam optim_actor_;
    torch::optim::Adam optim_q1_;
    torch::optim::Adam optim_q2_;

    torch::Tensor log_alpha_;   // leaf, requires_grad
    torch::Tensor alpha_value_; // detached exp(log_alpha)
    torch::optim::Adam optim_alpha_;

    // 工具
    torch::Tensor scale_to_env_action(const torch::Tensor& a_minus1_1) {
        return a_minus1_1 * cfg_.act_limit; // [-1,1] -> [-2,2]
    }
};
    