#pragma once
#include <torch/torch.h>

struct CriticImpl : torch::nn::Module {
    torch::nn::Linear c1{nullptr}, c2{nullptr}, c3{nullptr};
    int obs_dim, act_dim;

    CriticImpl(int obs_dim_=3, int act_dim_=1, int hidden=256)
    : obs_dim(obs_dim_), act_dim(act_dim_) {
        c1 = register_module("c1", torch::nn::Linear(obs_dim + act_dim, hidden));
        c2 = register_module("c2", torch::nn::Linear(hidden, hidden));
        c3 = register_module("c3", torch::nn::Linear(hidden, 1));
    }

    torch::Tensor forward(const torch::Tensor& s, const torch::Tensor& a) {
        auto x = torch::cat({s, a}, 1);
        x = torch::relu(c1->forward(x));
        x = torch::relu(c2->forward(x));
        return c3->forward(x); // [B,1]
    }
};
TORCH_MODULE(Critic);
