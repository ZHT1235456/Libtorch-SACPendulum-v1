#include "sac/sac_agent.h"
#include <iostream>
#include <filesystem>

SACAgent::SACAgent(const SACConfig& cfg, torch::Device device)
: cfg_(cfg), device_(device),
  actor_(Actor(cfg.obs_dim, cfg.hidden, cfg.act_dim)),
  q1_(Critic(cfg.obs_dim, cfg.act_dim, cfg.hidden)),
  q2_(Critic(cfg.obs_dim, cfg.act_dim, cfg.hidden)),
  tq1_(Critic(cfg.obs_dim, cfg.act_dim, cfg.hidden)),
  tq2_(Critic(cfg.obs_dim, cfg.act_dim, cfg.hidden)),
  optim_actor_(actor_->parameters(), torch::optim::AdamOptions(cfg.lr)),
  optim_q1_(q1_->parameters(), torch::optim::AdamOptions(cfg.lr)),
  optim_q2_(q2_->parameters(), torch::optim::AdamOptions(cfg.lr)),
  log_alpha_(torch::zeros({1}, torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true).device(device))),
  alpha_value_(torch::exp(log_alpha_.detach())),
  optim_alpha_({log_alpha_}, torch::optim::AdamOptions(cfg.lr))
{
    actor_->to(device_);
    q1_->to(device_);
    q2_->to(device_);
    tq1_->to(device_);
    tq2_->to(device_);

    // 初始化 target = online
    torch::NoGradGuard ng;
    for (auto& p : tq1_->named_parameters()) {
        p.value().copy_(q1_->named_parameters()[p.key()]);
    }
    for (auto& p : tq2_->named_parameters()) {
        p.value().copy_(q2_->named_parameters()[p.key()]);
    }
}

double SACAgent::select_action_train(const torch::Tensor& state_cpu) {
    auto s = state_cpu.unsqueeze(0).to(device_);
    auto pair = actor_->sample_action_and_logp(s);
    auto a01 = pair.first; // [-1,1]
    auto a = scale_to_env_action(a01);
    return a.item<double>();
}

double SACAgent::select_action_eval(const torch::Tensor& state_cpu) {
    auto s = state_cpu.unsqueeze(0).to(device_);
    auto a01 = actor_->act_deterministic(s);
    auto a = scale_to_env_action(a01);
    return a.item<double>();
}

void SACAgent::update(ReplayBuffer& buf) {
    if (buf.size() < (size_t)cfg_.batch_size) return;

    auto [S, A, R, S2, D] = buf.sample(cfg_.batch_size, device_);
    auto s  = S,  a = A,  r = R,  s2 = S2,  d = D;

    // ------- 1) target -------
    torch::Tensor target_q;
    {
        torch::NoGradGuard ng;
        auto [a2_01, logp2] = actor_->sample_action_and_logp(s2);
        auto a2 = scale_to_env_action(a2_01);
        auto q1_t = tq1_->forward(s2, a2);
        auto q2_t = tq2_->forward(s2, a2);
        auto min_q = torch::min(q1_t, q2_t);
        target_q = r + (1.0 - d) * cfg_.gamma * (min_q - alpha_value_ * logp2);
    }

    // ------- 2) update Qs -------
    {
        optim_q1_.zero_grad();
        auto q1v = q1_->forward(s, a);
        auto loss_q1 = torch::mse_loss(q1v, target_q);
        loss_q1.backward();
        optim_q1_.step();

        optim_q2_.zero_grad();
        auto q2v = q2_->forward(s, a);
        auto loss_q2 = torch::mse_loss(q2v, target_q);
        loss_q2.backward();
        optim_q2_.step();
    }

    // ------- 3) update Actor -------
    {
        optim_actor_.zero_grad();
        auto [a01, logp] = actor_->sample_action_and_logp(s);
        auto a_new = scale_to_env_action(a01);
        auto q1v = q1_->forward(s, a_new);
        auto q2v = q2_->forward(s, a_new);
        auto min_q = torch::min(q1v, q2v);
        auto loss_actor = (alpha_value_ * logp - min_q).mean();
        loss_actor.backward();
        optim_actor_.step();
    }

    // ------- 4) update alpha -------
    if (cfg_.autotune_alpha) {
        optim_alpha_.zero_grad();
        auto [a01, logp] = actor_->sample_action_and_logp(s);
        auto loss_alpha = (-log_alpha_ * (logp + cfg_.target_entropy).detach()).mean();
        loss_alpha.backward();
        optim_alpha_.step();
        alpha_value_ = torch::exp(log_alpha_.detach());
    }

    // ------- 5) soft update -------
    soft_update(cfg_.tau);
}

void SACAgent::soft_update(double tau) {
    torch::NoGradGuard ng;
    auto update = [tau](auto& target, auto& online) {
        for (const auto& it : online->named_parameters()) {
            auto& p_t = target->named_parameters()[it.key()];
            p_t.mul_(1.0 - tau);
            p_t.add_(tau * it.value());
        }
    };
    update(tq1_, q1_);
    update(tq2_, q2_);
}

// ----------------- Checkpoint -----------------
void SACAgent::save(const std::string& dir) {
    namespace fs = std::filesystem;
    fs::create_directories(dir);

    // 网络参数
    torch::save(actor_, fs::path(dir) / "actor.pt");
    torch::save(q1_,    fs::path(dir) / "q1.pt");
    torch::save(q2_,    fs::path(dir) / "q2.pt");
    torch::save(tq1_,   fs::path(dir) / "tq1.pt");
    torch::save(tq2_,   fs::path(dir) / "tq2.pt");

    // alpha 参数（用 tensor 存）
    torch::save(log_alpha_, fs::path(dir) / "log_alpha.pt");

    // （可选）也可以保存优化器状态，后续需要恢复学习率调度等再开启：
    // torch::save(optim_actor_, fs::path(dir) / "optim_actor.pt");
    // torch::save(optim_q1_,    fs::path(dir) / "optim_q1.pt");
    // torch::save(optim_q2_,    fs::path(dir) / "optim_q2.pt");
    // torch::save(optim_alpha_, fs::path(dir) / "optim_alpha.pt");
    std::cout << "[checkpoint] saved to " << dir << "\n";
}

bool SACAgent::load(const std::string& dir, torch::Device dev) {
    namespace fs = std::filesystem;
    auto ok = fs::exists(fs::path(dir) / "actor.pt") &&
              fs::exists(fs::path(dir) / "q1.pt")    &&
              fs::exists(fs::path(dir) / "q2.pt")    &&
              fs::exists(fs::path(dir) / "tq1.pt")   &&
              fs::exists(fs::path(dir) / "tq2.pt")   &&
              fs::exists(fs::path(dir) / "log_alpha.pt");
    if (!ok) {
        std::cerr << "[checkpoint] missing files in " << dir << ", skip load.\n";
        return false;
    }

    try {
        torch::load(actor_, fs::path(dir) / "actor.pt");
        torch::load(q1_,    fs::path(dir) / "q1.pt");
        torch::load(q2_,    fs::path(dir) / "q2.pt");
        torch::load(tq1_,   fs::path(dir) / "tq1.pt");
        torch::load(tq2_,   fs::path(dir) / "tq2.pt");
        torch::load(log_alpha_, fs::path(dir) / "log_alpha.pt");

        actor_->to(dev);
        q1_->to(dev);
        q2_->to(dev);
        tq1_->to(dev);
        tq2_->to(dev);
        log_alpha_ = log_alpha_.to(dev).set_requires_grad(true);
        alpha_value_ = torch::exp(log_alpha_.detach());

        std::cout << "[checkpoint] loaded from " << dir << "\n";
        return true;
    } catch (const c10::Error& e) {
        std::cerr << "[checkpoint] load failed: " << e.msg() << "\n";
        return false;
    }
}
