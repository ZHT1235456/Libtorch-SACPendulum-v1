#include <torch/torch.h>
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <random>
#include <string>
#include <filesystem>
#include <fstream>
#include <sstream>

#include "env/pendulum.h"
#include "sac/sac_agent.h"
#include "utils/replay_buffer.h"
#include "utils/logger.h"
#include "vis/renderer.h"
#include "utils/state_io.h"   // <-- 新增：state.json 读写

namespace fs = std::filesystem;

// ------------ 小工具 ------------
static std::string get_arg(int argc, char** argv, const std::string& key, const std::string& def = "") {
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == key && i + 1 < argc) return argv[i + 1];
    return def;
}
static bool has_flag(int argc, char** argv, const std::string& flag) {
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == flag) return true;
    return false;
}
static double angle_from_obs(const std::array<double,3>& s) {
    return std::atan2(s[1], s[0]); // [-pi, pi], 0=up
}

// ------------ 训练 ------------
void train_loop(const SACConfig& sac, const YAML::Node& y, bool resume) {
    struct TrainCfg {
        int total_steps=150000, start_steps=1000, max_ep_len=200;
        int eval_interval=5000, eval_episodes=5, seed=0, env_seed_base=123;
    } tr;
    tr.total_steps   = y["total_steps"]   ? y["total_steps"].as<int>()   : 150000;
    tr.start_steps   = y["start_steps"]   ? y["start_steps"].as<int>()   : 1000;
    tr.max_ep_len    = y["max_ep_len"]    ? y["max_ep_len"].as<int>()    : 200;
    tr.eval_interval = y["eval_interval"] ? y["eval_interval"].as<int>() : 5000;
    tr.eval_episodes = y["eval_episodes"] ? y["eval_episodes"].as<int>(): 5;
    tr.seed          = y["seed"]          ? y["seed"].as<int>()          : 0;
    tr.env_seed_base = y["env_seed_base"] ? y["env_seed_base"].as<int>() : 123;

    torch::manual_seed(tr.seed);
    torch::Device device(torch::kCPU);

    // 目录 / 状态文件
    const std::string ckpt_dir   = "checkpoints";
    const std::string state_path = ckpt_dir + "/state.json";
    fs::create_directories(ckpt_dir);

    PendulumEnv env;
    SACAgent agent(sac, device);

    // --- 断点续训：加载模型 + state.json ---
    long   steps     = 0;
    double best_eval = -1e9;
    if (resume) {
        agent.load(ckpt_dir, device);
        if (auto st = load_train_state(state_path)) {
            steps     = st->global_step;
            best_eval = st->best_eval;
            // 如需复用历史随机种子，可取消注释：
            // tr.seed = st->seed;
            // tr.env_seed_base = st->env_seed_base;
            std::cout << "[resume] state loaded: step=" << steps
                      << " best_eval=" << best_eval
                      << " last=" << st->last_update_iso << "\n";
        } else {
            std::cout << "[resume] no state.json, continue without it.\n";
        }
    }

    ReplayBuffer buf(1'000'000, sac.obs_dim, sac.act_dim);
    CSVLogger train_log("logs/train.csv", {"step", "episode_return"}, /*append=*/resume);
    CSVLogger eval_log("logs/eval.csv",   {"step", "avg_return", "alpha"}, /*append=*/resume);

    std::mt19937 rng(tr.seed);
    std::uniform_real_distribution<double> uni_action(-sac.act_limit, sac.act_limit);

    auto to_tensor = [](const std::array<double,3>& s) {
        return torch::tensor({(float)s[0], (float)s[1], (float)s[2]}, torch::kFloat32);
    };

    int ep_len = 0;
    double ep_ret = 0.0;
    auto s_arr = env.reset(tr.env_seed_base + (int)steps); // 接上步数播种更平滑
    auto s = to_tensor(s_arr);

    while (steps < tr.total_steps) {
        // 动作：warmup 随机 / SAC
        double a_scalar = (steps < tr.start_steps) ? uni_action(rng) : agent.select_action_train(s);

        // 环境一步
        auto out = env.step(a_scalar);
        auto s2 = to_tensor(out.state);

        // 存入 buffer（动作是真实尺度）
        buf.push(s,
                 torch::tensor({(float)a_scalar}, torch::kFloat32),
                 torch::tensor({(float)out.reward}, torch::kFloat32),
                 s2,
                 torch::tensor({0.0f}, torch::kFloat32));

        // 推进
        s = s2; ep_ret += out.reward; ep_len++; steps++;

        // 更新
        if (buf.size() >= (size_t)sac.batch_size)
            for (int u=0; u<sac.updates_per_step; ++u) agent.update(buf);

        // 回合截断（固定长度）
        if (ep_len >= tr.max_ep_len) {
            std::cout << "[train] step=" << steps << " ep_ret=" << ep_ret << "\n";
            train_log.write_row({(double)steps, ep_ret});
            s_arr = env.reset(tr.env_seed_base + (int)steps);
            s = to_tensor(s_arr);
            ep_len = 0; ep_ret = 0.0;
        }

        // 定期评估（不渲染）
        if (steps % tr.eval_interval == 0) {
            double avg = 0.0;
            for (int e=0; e<tr.eval_episodes; ++e) {
                auto st = to_tensor(env.reset(1000 + e));
                double er = 0.0;
                for (int t=0; t<tr.max_ep_len; ++t) {
                    double a_eval = agent.select_action_eval(st);
                    auto o2 = env.step(a_eval);
                    er += o2.reward;
                    st = to_tensor(o2.state);
                }
                avg += er;
            }
            avg /= tr.eval_episodes;
            std::cout << "[eval] step=" << steps << " avg_return=" << avg << " alpha=" << agent.alpha() << "\n";
            eval_log.write_row({(double)steps, avg, agent.alpha()});

            if (avg > best_eval) {
                best_eval = avg;
                std::cout << "[checkpoint] new best avg_return=" << best_eval << "\n";
                agent.save(ckpt_dir);
            }

            // 刷新运行状态
            TrainState st;
            st.global_step    = steps;
            st.best_eval      = best_eval;
            st.seed           = tr.seed;
            st.env_seed_base  = tr.env_seed_base;
            st.last_update_iso= iso8601_now();
            save_train_state(state_path, st);

            train_log.flush(); eval_log.flush();
        }
    }

    // 兜底再保存一次 state
    TrainState st_final;
    st_final.global_step    = steps;
    st_final.best_eval      = best_eval;
    st_final.seed           = tr.seed;
    st_final.env_seed_base  = tr.env_seed_base;
    st_final.last_update_iso= iso8601_now();
    save_train_state(state_path, st_final);

    std::cout << "Training finished.\n";
}

// ------------ 评估（默认渲染） ------------
void eval_loop(const SACConfig& sac, const YAML::Node& y) {
    int eval_episodes = y["eval_episodes"] ? y["eval_episodes"].as<int>() : 5;
    int max_ep_len    = y["max_ep_len"]    ? y["max_ep_len"].as<int>()    : 200;

    torch::Device device(torch::kCPU);
    SACAgent agent(sac, device);
    if (!agent.load("checkpoints", device)) {
        std::cerr << "[eval] No checkpoint found in ./checkpoints\n";
        return;
    }

    // 读取 state.json 仅用于提示
    if (auto st = load_train_state("checkpoints/state.json")) {
        std::cout << "[eval] model trained until step=" << st->global_step
                  << ", best_eval=" << st->best_eval
                  << ", last_update=" << st->last_update_iso << "\n";
    }

    PendulumEnv env;
    PendulumRenderer renderer(600, 600, 200);

    auto to_tensor = [](const std::array<double,3>& s) {
        return torch::tensor({(float)s[0], (float)s[1], (float)s[2]}, torch::kFloat32);
    };

    for (int e=0; e<eval_episodes; ++e) {
        auto s_arr = env.reset(2000 + e);
        auto s = to_tensor(s_arr);
        double ep_ret = 0.0;
        for (int t=0; t<max_ep_len; ++t) {
            double a_eval = agent.select_action_eval(s);
            auto out = env.step(a_eval);
            ep_ret += out.reward;
            s = to_tensor(out.state);

            // 始终渲染（按 q 可关闭窗口）
            if (!renderer.is_closed())
                renderer.render(angle_from_obs(out.state), a_eval, out.reward);
        }
        std::cout << "[eval] episode=" << e << " return=" << ep_ret << "\n";
    }
    std::cout << "[eval] Done. Press any key in the window to exit..." << std::endl;
    cv::waitKey(0);            // 按任意键继续
    cv::destroyAllWindows();   // 关闭窗口（可选）

}

// ------------ main ------------
int main(int argc, char** argv) {
    std::string mode = get_arg(argc, argv, "--mode", "");
    if (mode != "train" && mode != "eval") {
        std::cerr << "Usage: ./sac_pendulum --mode train|eval [--resume] [--config path]\n";
        return 1;
    }
    bool resume = has_flag(argc, argv, "--resume");
    // 默认从 build/ 的上一级读取配置
    std::string cfg_path = get_arg(argc, argv, "--config", "../config.yaml");

    YAML::Node y;
    try {
        y = YAML::LoadFile(cfg_path);
        std::cout << "[config] loaded: " << cfg_path << "\n";
    } catch (...) {
        std::cerr << "[config] cannot load " << cfg_path << "\n";
        return 1;
    }

    SACConfig sac;
    sac.obs_dim         = y["obs_dim"]        ? y["obs_dim"].as<int>()        : 3;
    sac.act_dim         = y["act_dim"]        ? y["act_dim"].as<int>()        : 1;
    sac.act_limit       = y["act_limit"]      ? y["act_limit"].as<double>()   : 2.0;
    sac.gamma           = y["gamma"]          ? y["gamma"].as<double>()       : 0.99;
    sac.tau             = y["tau"]            ? y["tau"].as<double>()         : 0.005;
    sac.hidden          = y["hidden"]         ? y["hidden"].as<int>()         : 256;
    sac.batch_size      = y["batch_size"]     ? y["batch_size"].as<int>()     : 256;
    sac.lr              = y["lr"]             ? y["lr"].as<double>()          : 3e-4;
    sac.autotune_alpha  = y["autotune_alpha"] ? y["autotune_alpha"].as<bool>(): true;
    sac.target_entropy  = y["target_entropy"] ? y["target_entropy"].as<double>(): -1.0;
    sac.updates_per_step= y["updates_per_step"] ? y["updates_per_step"].as<int>() : 1;

    if (mode == "train") train_loop(sac, y, resume);
    else                 eval_loop(sac, y);

    return 0;
}
