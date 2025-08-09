import csv, argparse
import matplotlib.pyplot as plt
from collections import deque

def moving_avg(xs, k):
    if k <= 1: return xs
    out, q, s = [], deque(), 0.0
    for v in xs:
        q.append(v); s += v
        if len(q) > k: s -= q.popleft()
        out.append(s / len(q))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="logs/train.csv", help="路径到 train.csv")
    ap.add_argument("--smooth", type=int, default=1, help="滑动平均窗口(步数)，默认不平滑")
    args = ap.parse_args()

    steps, rets = [], []
    try:
        with open(args.file, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                steps.append(float(row["step"]))
                rets.append(float(row["episode_return"]))
    except FileNotFoundError:
        print(f"找不到文件: {args.file}\n请先运行训练，确保生成 logs/train.csv。")
        return

    if not steps:
        print("train.csv 为空，先跑点训练数据再来画图吧～")
        return

    rets_s = moving_avg(rets, args.smooth)

    plt.figure()
    plt.plot(steps, rets, label="Episode Return", alpha=0.35)
    if args.smooth > 1:
        plt.plot(steps, rets_s, label=f"Smoothed (k={args.smooth})")
    plt.xlabel("Global Step")
    plt.ylabel("Episode Return")
    plt.title("SAC on Pendulum-v1 — Training Episode Reward")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
