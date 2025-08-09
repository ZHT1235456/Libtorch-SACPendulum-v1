# Libtorch-SACPendulum-v1

基于 **LibTorch**（PyTorch C++ API）实现的 **Soft Actor-Critic (SAC)** 算法，并应用于经典控制任务 **Pendulum-v1**。  
支持训练与评估模式，评估时提供 OpenCV 渲染效果，可观察摆杆动态。

---

## 功能特性
- **SAC 算法 C++ 实现**（无 Python 推理依赖）
- **LibTorch 2.8.0 + CPU 版本**
- **Pendulum-v1 环境模拟**
- **命令行参数控制**：训练 / 评估 / 断点续训 / 自定义配置
- **CSV 日志输出**：记录训练过程的回合奖励，可直接用 Python 绘图
- **OpenCV 可视化**：评估时动态显示摆杆状态

---

## 环境配置

### 1. 安装依赖
- **操作系统**：Ubuntu 20.04 / 22.04 / 24.04
- **必备工具**：
  ```bash
  sudo apt update
  sudo apt install cmake g++ libopencv-dev


* **YAML 解析库**（nlohmann/json 已集成）

  ```bash
  sudo apt install nlohmann-json3-dev
  ```
* **LibTorch (CPU)**：
  从 [PyTorch 官网](https://pytorch.org/get-started/locally/) 下载
  `libtorch-shared-with-deps-2.8.0+cpu.zip`，解压到 `/home/用户名/libtorch`

### 2. 编译项目

```bash
git clone git@github.com:ZHT1235456/Libtorch-SACPendulum-v1.git
cd Libtorch-SACPendulum-v1
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=/home/用户名/libtorch
make -j
```

---

## 使用方法

### 训练

```bash
./sac_pendulum --mode train
```

### 断点续训

```bash
./sac_pendulum --mode train --resume
```

### 评估（可视化）

```bash
./sac_pendulum --mode eval
```

---

## 日志与可视化

训练过程会在 `build/logs/train.csv` 中记录每回合奖励。
可用 `plot_train.py` 绘制训练曲线：

```bash
python plot_train.py
```

评估时会使用 OpenCV 渲染摆杆状态窗口。

---

## 项目结构

```
Libtorch-SACPendulum-v1/
│── CMakeLists.txt
│── config.yaml
│── plot_train.py
│── figures/
│   ├── train_curve.png
│   └── render_example.png
└── src/
    ├── main.cpp
    ├── env/
    │   ├── pendulum.h
    │   └── pendulum.cpp
    ├── sac/
    │   ├── actor.h
    │   ├── critic.h
    │   ├── sac_agent.h
    │   └── sac_agent.cpp
    ├── utils/
    │   ├── replay_buffer.h
    │   ├── logger.h
    │   ├── state_io.h
    │   └── state_io.cpp
    └── vis/
        ├── renderer.h
        └── renderer.cpp


