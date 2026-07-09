import pandas as pd
import matplotlib.pyplot as plt

# 1. Plot the Convergence Line Chart
df = pd.read_csv('neural_training_log.csv')

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(df['Episode'], df['MovingAverageWait'], label='Neural Agent (Double DQN)', color='purple', linewidth=2)
# Hardcoded baselines from your Assignment 3 universal test set
plt.axhline(y=136.5, color='green', linestyle='--', label='SJF Baseline')
plt.axhline(y=275.3, color='orange', linestyle='--', label='RR Baseline')
plt.title("Neural Agent: Wait Time Convergence")
plt.xlabel("Training Episodes")
plt.ylabel("Mean Wait Time")
plt.legend()
plt.grid(True)

# 2. Plot the Tradeoff Scatter Plot (Pareto Frontier)
plt.subplot(1, 2, 2)
# Replace the Agent coordinates with the final evaluation numbers your C++ prints out
schedulers = {
    'FCFS': (387.2, 0.69, 'red'),
    'RR': (275.3, 0.72, 'orange'),
    'SJF': (136.5, 0.42, 'green'),
    'Neural Agent': (160.0, 0.85, 'purple') # <--- Update these post-training
}

for name, (wait, fairness, color) in schedulers.items():
    plt.scatter(wait, fairness, color=color, s=100, label=name)
    plt.text(wait + 5, fairness, name, fontsize=10, weight='bold')

plt.title("The Pareto Frontier: Wait Time vs. Fairness")
plt.xlabel("Mean Wait Time (Lower is Better)")
plt.ylabel("Jain's Fairness Index (Higher is Better)")
plt.grid(True, linestyle=':')

plt.tight_layout()
plt.show()
