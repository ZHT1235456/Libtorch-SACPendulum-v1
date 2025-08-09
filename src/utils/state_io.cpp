#include "utils/state_io.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sstream>

using json = nlohmann::json;

std::string iso8601_now() {
    using clock = std::chrono::system_clock;
    auto now = clock::now();
    std::time_t t = clock::to_time_t(now);
    std::tm tm{};
    gmtime_r(&t, &tm);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
    return oss.str();
}

std::optional<TrainState> load_train_state(const std::string& path) {
    std::ifstream in(path);
    if (!in.is_open()) return std::nullopt;
    try {
        json j; in >> j;
        TrainState st;
        st.global_step   = j.value("global_step", 0L);
        st.best_eval     = j.value("best_eval", -1e9);
        st.seed          = j.value("seed", 0);
        st.env_seed_base = j.value("env_seed_base", 123);
        st.last_update_iso = j.value("last_update_iso", "");
        st.config_digest   = j.value("config_digest", "");
        return st;
    } catch (...) {
        return std::nullopt;
    }
}

bool save_train_state(const std::string& path, const TrainState& st) {
    try {
        json j = {
            {"global_step",    st.global_step},
            {"best_eval",      st.best_eval},
            {"seed",           st.seed},
            {"env_seed_base",  st.env_seed_base},
            {"last_update_iso",st.last_update_iso},
            {"config_digest",  st.config_digest}
        };
        std::ofstream out(path);
        if (!out.is_open()) return false;
        out << j.dump(2) << std::endl;
        return true;
    } catch (...) {
        return false;
    }
}
