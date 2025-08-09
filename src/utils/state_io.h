#pragma once
#include <string>
#include <optional>

struct TrainState {
    long   global_step   = 0;
    double best_eval     = -1e9;
    int    seed          = 0;
    int    env_seed_base = 123;
    std::string last_update_iso;   // ISO8601 时间戳
    std::string config_digest;     // 可选：配置摘要（留空也行）
};

std::optional<TrainState> load_train_state(const std::string& path);
bool save_train_state(const std::string& path, const TrainState& st);
std::string iso8601_now();
