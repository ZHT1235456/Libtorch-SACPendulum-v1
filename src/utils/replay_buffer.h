#pragma once
#include <torch/torch.h>
#include <deque>
#include <random>

struct Transition {
    torch::Tensor s;   // [obs_dim]
    torch::Tensor a;   // [act_dim]  (真实动作范围 [-2,2])
    torch::Tensor r;   // [1]
    torch::Tensor s2;  // [obs_dim]
    torch::Tensor d;   // [1]  done (float 0/1)
};

class ReplayBuffer {
public:
    ReplayBuffer(size_t capacity, int obs_dim, int act_dim)
    : capacity_(capacity), obs_dim_(obs_dim), act_dim_(act_dim), rng_(123) {}

    void push(const torch::Tensor& s, const torch::Tensor& a,
              const torch::Tensor& r, const torch::Tensor& s2,
              const torch::Tensor& d) {
        if (data_.size() >= capacity_) data_.pop_front();
        data_.push_back(Transition{ s.detach().cpu(), a.detach().cpu(),
                                    r.detach().cpu(), s2.detach().cpu(),
                                    d.detach().cpu() });
    }

    size_t size() const { return data_.size(); }

    // 返回 batch 张量（在 device 上）
    std::tuple<torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor>
    sample(size_t batch_size, torch::Device device) {
        std::uniform_int_distribution<size_t> uni(0, data_.size()-1);

        auto S  = torch::empty({(long)batch_size, obs_dim_}, torch::kFloat32);
        auto A  = torch::empty({(long)batch_size, act_dim_}, torch::kFloat32);
        auto R  = torch::empty({(long)batch_size, 1},        torch::kFloat32);
        auto S2 = torch::empty({(long)batch_size, obs_dim_}, torch::kFloat32);
        auto D  = torch::empty({(long)batch_size, 1},        torch::kFloat32);

        for (size_t i=0; i<batch_size; ++i) {
            const auto& tr = data_[uni(rng_)];
            S[i]  = tr.s;
            A[i]  = tr.a;
            R[i]  = tr.r;
            S2[i] = tr.s2;
            D[i]  = tr.d;
        }
        return {S.to(device), A.to(device), R.to(device), S2.to(device), D.to(device)};
    }

private:
    size_t capacity_;
    int obs_dim_, act_dim_;
    std::deque<Transition> data_;
    std::mt19937 rng_;
};
