import csv
import matplotlib.pyplot as plt

steps = []
avg_returns = []
alphas = []

with open('build/logs/train.csv', 'r', newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            steps.append(float(row['step']))
            avg_returns.append(float(row['avg_return']))
            alphas.append(float(row['alpha']))
        except Exception:
            continue

if not steps:
    print("No data in logs/eval.csv. Run training first.")
    exit(0)

fig, ax1 = plt.subplots()
ax1.plot(steps, avg_returns, label='Avg Return')
ax1.set_xlabel('Steps')
ax1.set_ylabel('Avg Return')
ax1.grid(True)

# 可选：次轴画 alpha（如果不想要就注释掉这段）
ax2 = ax1.twinx()
ax2.plot(steps, alphas, linestyle='--', label='Alpha')
ax2.set_ylabel('Alpha')

# 合并图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

plt.title('SAC on Pendulum-v1 (Eval)')
plt.tight_layout()
plt.show()
