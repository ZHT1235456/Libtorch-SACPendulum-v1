#pragma once
#include <torch/torch.h>
#include <cmath>

inline torch::Tensor normal_log_prob(const torch::Tensor& x, const torch::Tensor& mean, const torch::Tensor& log_std) {
    // log N(x; mean, std) = -0.5*((x-mean)^2 / std^2 + 2*log_std + log(2π))
    static const double LOG_TWO_PI = std::log(2.0 * M_PI);
    auto var = torch::exp(2.0 * log_std);
    auto log_prob = -0.5 * ((x - mean) * (x - mean) / var + 2.0 * log_std + LOG_TWO_PI);
    return log_prob; // same shape as x
}

struct ActorImpl : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, mean{nullptr}, log_std{nullptr};

    int obs_dim, act_dim;
    double log_std_min = -20.0, log_std_max = 2.0;

    ActorImpl(int obs_dim_=3, int hidden=256, int act_dim_=1)
    : obs_dim(obs_dim_), act_dim(act_dim_) {
        fc1 = register_module("fc1", torch::nn::Linear(obs_dim, hidden));
        fc2 = register_module("fc2", torch::nn::Linear(hidden, hidden));
        mean = register_module("mean", torch::nn::Linear(hidden, act_dim));
        log_std = register_module("log_std", torch::nn::Linear(hidden, act_dim));
    }

    std::pair<torch::Tensor, torch::Tensor> forward_impl(const torch::Tensor& x) {
        auto h = torch::relu(fc1->forward(x));
        h = torch::relu(fc2->forward(h));
        auto mu = mean->forward(h);
        auto ls = torch::clamp(log_std->forward(h), log_std_min, log_std_max);
        return {mu, ls};
    }

    // 训练用：重参数化采样 + tanh 压缩 + log_prob 修正（返回 a in [-1,1]）
    std::pair<torch::Tensor, torch::Tensor> sample_action_and_logp(const torch::Tensor& state) {
        auto pair = forward_impl(state);
        auto mu = pair.first;
        auto ls = pair.second;
        auto std = torch::exp(ls);

        auto eps = torch::randn_like(mu);
        auto u = mu + std * eps;            // pre-tanh
        auto a = torch::tanh(u);            // in [-1,1]

        // log π(a|s) = log N(u; mu, std) - sum log(1 - tanh(u)^2)
        auto logp_u = normal_log_prob(u, mu, ls);  // shape [B, act_dim]
        auto correction = torch::log(1.0 - a * a + 1e-6); // shape [B, act_dim]
        auto logp = (logp_u - correction).sum(1, true);   // [B,1]
        return {a, logp};
    }

    // 评估用：确定性动作（μ -> tanh）
    torch::Tensor act_deterministic(const torch::Tensor& state) {
        auto pair = forward_impl(state);
        auto mu = pair.first;
        return torch::tanh(mu);
    }
};
TORCH_MODULE(Actor);
