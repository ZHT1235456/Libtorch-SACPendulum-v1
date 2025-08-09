cmake_minimum_required(VERSION 3.18)
project(SACPendulum LANGUAGES CXX)

# 按你的实际路径
set(CMAKE_PREFIX_PATH "/home/z/libtorch")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(OpenCV REQUIRED)
find_package(nlohmann_json REQUIRED)

# 先定义可执行目标（确保这些源文件真实存在）
add_executable(sac_pendulum
    src/main.cpp
    src/env/pendulum.cpp
    src/sac/sac_agent.cpp
    src/vis/renderer.cpp
    src/utils/state_io.cpp
)

# 再设置包含目录、链接库
target_include_directories(sac_pendulum PRIVATE src)

target_link_libraries(sac_pendulum
    "${TORCH_LIBRARIES}"
    yaml-cpp
    ${OpenCV_LIBS}
    nlohmann_json::nlohmann_json
)

# Torch 推荐编译旗标
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
