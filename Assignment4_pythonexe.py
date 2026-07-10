
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
from collections import deque
import matplotlib.pyplot as plt
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

class Process:
    """Extended Assignment-3 Process representation.

    virtual_runtime is the cumulative number of ticks this process has
    actually been executed for (NOT the same as wait time). This is the
    quantity the CFS-style fairness reward is built around.
    """
    def __init__(self, pid, arrival_time, burst_time, priority=1.0):
        self.pid = pid
        self.arrival_time = arrival_time
        self.burst_time = burst_time
        self.remaining_time = burst_time
        self.start_time = -1.0
        self.finish_time = -1.0
        self.virtual_runtime = 0.0
        self.priority = priority

    def clone(self):
        p = Process(self.pid, self.arrival_time, self.burst_time, self.priority)
        return p


class WorkloadGenerator:
    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)

    def generate_episode(self, num_processes):
        processes = []
        current_time = 0.0
        for i in range(num_processes):
            current_time += self.rng.poisson(4.0)
            burst = max(1.0, self.rng.exponential(1.0 / 0.2) * 10.0)
            prio = self.rng.uniform(1.0, 10.0)
            processes.append(Process(i + 1, current_time, burst, prio))
        return processes

    def generate_fixed_test_set(self, num_episodes, num_processes):
        """A reproducible, held-out set of workloads used to evaluate
        baselines AND the trained agent identically (Part 5.1/5.4)."""
        return [self.generate_episode(num_processes) for _ in range(num_episodes)]

def calculate_metrics(processes):
    waits = []
    fairness_terms = []
    for p in processes:
        turnaround = p.finish_time - p.arrival_time
        wait = max(0.0, turnaround - p.burst_time)
        waits.append(wait)
        ratio = p.burst_time / max(turnaround, 1e-6)
        fairness_terms.append(ratio)

    waits = np.array(waits, dtype=np.float64)
    mean_wait = float(waits.mean()) if len(waits) else 0.0
    p90_wait = float(np.percentile(waits, 90)) if len(waits) else 0.0

    x = np.array(fairness_terms, dtype=np.float64)
    n = len(x)
    jains = float((x.sum() ** 2) / (n * (x ** 2).sum())) if n and (x ** 2).sum() > 0 else 0.0

    return {"mean_wait": mean_wait, "p90_wait": p90_wait, "jains_fairness": jains}

def _run_baseline(processes, policy):
    procs = [p.clone() for p in processes]
    procs.sort(key=lambda p: p.arrival_time)
    n = len(procs)
    ready_queue = []
    current_time = 0.0
    completed = 0
    rr_pointer = 0

    def load_arrived():
        nonlocal ready_queue
        for p in procs:
            if p.arrival_time <= current_time and p.finish_time == -1.0 and p not in ready_queue:
                ready_queue.append(p)

    load_arrived()
    if not ready_queue and procs:
        current_time = procs[0].arrival_time
        load_arrived()

    while completed < n:
        if not ready_queue:
            future = [p.arrival_time for p in procs if p.finish_time == -1.0]
            if not future:
                break
            current_time = min(future)
            load_arrived()
            continue

        idx = policy(ready_queue, current_time, rr_pointer)
        idx = idx % len(ready_queue)
        active = ready_queue[idx]
        if active.start_time == -1.0:
            active.start_time = current_time
        active.remaining_time -= 1.0
        active.virtual_runtime += 1.0
        current_time += 1.0

        if active.remaining_time <= 1e-9:
            active.finish_time = current_time
            active.remaining_time = 0.0
            completed += 1
            ready_queue.pop(idx)
            if policy is _rr_policy:
                rr_pointer = idx % max(len(ready_queue), 1)
        else:
            if policy is _rr_policy:
                rr_pointer = (idx + 1) % len(ready_queue)

        load_arrived()

    return procs


def _fcfs_policy(ready_queue, current_time, rr_pointer):
    # Earliest arrival first
    return min(range(len(ready_queue)), key=lambda i: ready_queue[i].arrival_time)


def _sjf_policy(ready_queue, current_time, rr_pointer):
    # Shortest remaining time first
    return min(range(len(ready_queue)), key=lambda i: ready_queue[i].remaining_time)


def _rr_policy(ready_queue, current_time, rr_pointer):
    return rr_pointer


def _random_policy(ready_queue, current_time, rr_pointer):
    return random.randrange(len(ready_queue))


BASELINE_POLICIES = {
    "FCFS": _fcfs_policy,
    "SJF": _sjf_policy,
    "RR": _rr_policy,
    "Random": _random_policy,
}

def evaluate_baselines(test_set):
    results = {}
    for name, policy in BASELINE_POLICIES.items():
        agg = {"mean_wait": [], "p90_wait": [], "jains_fairness": []}
        for workload in test_set:
            finished = _run_baseline(workload, policy)
            m = calculate_metrics(finished)
            for k in agg:
                agg[k].append(m[k])
        results[name] = {k: float(np.mean(v)) for k, v in agg.items()}
    return results

class DirectRLEnvironment:
    FEATURES = 5  

    def __init__(self, max_queue=10):
        self.max_queue = max_queue
        self.alpha = 0.5          # weight on virtual-runtime variance
        self.beta = 1.0           # weight on starvation penalty
        self.starvation_threshold = 100.0

    def load_arrived_processes(self):
        for p in self.processes:
            if p.arrival_time <= self.current_time and p.finish_time == -1.0:
                if p not in self.ready_queue:
                    self.ready_queue.append(p)

    def extract_state_tensor(self):
        tensor = np.zeros((self.max_queue, self.FEATURES), dtype=np.float32)
        mask = np.zeros(self.max_queue, dtype=bool)

        for i in range(self.max_queue):
            if i < len(self.ready_queue):
                p = self.ready_queue[i]
                wait_time = self.current_time - p.arrival_time
                tensor[i][0] = p.remaining_time / 100.0
                tensor[i][1] = min(wait_time / 500.0, 1.0)
                tensor[i][2] = p.priority / 10.0
                tensor[i][3] = min(p.virtual_runtime / 100.0, 1.0)
                starve_gap = (wait_time - p.virtual_runtime) - self.starvation_threshold
                tensor[i][4] = np.clip(starve_gap / 100.0, -1.0, 1.0)
                mask[i] = True
        return tensor, mask

    def reset(self, processes):
        self.processes = processes
        self.ready_queue = []
        self.current_time = 0.0
        self.completed_processes = 0
        self.load_arrived_processes()

        if not self.ready_queue and self.processes:
            self.current_time = self.processes[0].arrival_time
            self.load_arrived_processes()

        return self.extract_state_tensor()

    def step(self, action_index):
        if not self.ready_queue:
            next_arrival = min(
                [p.arrival_time for p in self.processes if p.finish_time == -1.0],
                default=1e9,
            )
            if next_arrival != 1e9:
                self.current_time = next_arrival
                self.load_arrived_processes()
            done = self.completed_processes >= len(self.processes)
            return self.extract_state_tensor(), 0.0, done

        invalid_action_penalty = 0.0
        # Part 3.2: validate the action is actually marked valid in the mask.
        if action_index >= len(self.ready_queue) or action_index < 0:
            invalid_action_penalty = -50.0
            action_index = 0

        active_p = self.ready_queue[action_index]
        if active_p.start_time == -1.0:
            active_p.start_time = self.current_time

        active_p.remaining_time -= 1.0
        active_p.virtual_runtime += 1.0
        self.current_time += 1.0

        if active_p.remaining_time <= 0:
            active_p.finish_time = self.current_time
            active_p.remaining_time = 0
            self.completed_processes += 1
            self.ready_queue.pop(action_index)

        self.load_arrived_processes()

        while not self.ready_queue and self.completed_processes < len(self.processes):
            future_arrivals = [p.arrival_time for p in self.processes if p.finish_time == -1.0]
            if not future_arrivals:
                break
            self.current_time = min(future_arrivals)
            self.load_arrived_processes()
          
        variance, max_wait = 0.0, 0.0
        if self.ready_queue:
            mean_v = sum(p.virtual_runtime for p in self.ready_queue) / len(self.ready_queue)
            variance = sum((p.virtual_runtime - mean_v) ** 2 for p in self.ready_queue) / len(self.ready_queue)
            max_wait = max(
                (self.current_time - p.arrival_time) - p.virtual_runtime for p in self.ready_queue
            )

        # Part 3.4 - starvation cap (escalating, bounded penalty)
        starvation_penalty = 0.0
        if max_wait > self.starvation_threshold:
            excess = max_wait - self.starvation_threshold
            starvation_penalty = min(excess * excess * 0.1, 200.0)  # Part 3.5: bounded

        variance_term = min(variance, 100.0)  # Part 3.5: bounded
        reward = -(self.alpha * variance_term + self.beta * starvation_penalty) + invalid_action_penalty
        done = self.completed_processes >= len(self.processes)

        return self.extract_state_tensor(), reward, done

    def get_finished_processes(self):
        return [p for p in self.processes if p.finish_time != -1.0]

class DirectSchedulerNet(nn.Module):
    def __init__(self, in_features=5, embed_dim=64, num_heads=4, max_queue=10):
        super().__init__()
        self.embedding = nn.Linear(in_features, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_queue, embed_dim) * 0.02)

        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

        self.head = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, state_tensor, valid_mask):
        x = self.embedding(state_tensor) + self.pos_embedding[:, : state_tensor.shape[1], :]

        key_padding_mask = ~valid_mask  # MultiheadAttention: True = IGNORE

        # Pre-norm residual attention block
        normed = self.norm1(x)
        attn_out, _ = self.attention(normed, normed, normed, key_padding_mask=key_padding_mask)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))

        q_values = self.head(x).squeeze(-1)  # [batch, N]
        q_values = q_values.masked_fill(~valid_mask, float("-1e9"))
        return q_values
      
class DoubleDQNAgent:
    def __init__(self, max_queue=10, in_features=5, lr=1e-3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DirectSchedulerNet(in_features=in_features, max_queue=max_queue).to(self.device)
        self.target_net = DirectSchedulerNet(in_features=in_features, max_queue=max_queue).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        self.memory = deque(maxlen=20000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995
        self.reward_scale = 0.01 

        self.last_q_magnitude = 0.0
        self.grad_steps = 0
        self.target_sync_steps = 150

    def choose_action(self, state_tensor, mask):
        if np.random.rand() < self.epsilon:
            valid_indices = [i for i, m in enumerate(mask) if m]
            return random.choice(valid_indices) if valid_indices else 0

        with torch.no_grad():
            t_state = torch.tensor(state_tensor, dtype=torch.float32).unsqueeze(0).to(self.device)
            t_mask = torch.tensor(mask, dtype=torch.bool).unsqueeze(0).to(self.device)
            q_values = self.policy_net(t_state, t_mask)
            return q_values.argmax(dim=1).item()

    def push_memory(self, state, mask, action, reward, next_state, next_mask, done):
        self.memory.append((state, mask, action, reward, next_state, next_mask, done))

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(self.memory, self.batch_size)

        states = torch.tensor(np.array([x[0] for x in batch]), dtype=torch.float32).to(self.device)
        masks = torch.tensor(np.array([x[1] for x in batch]), dtype=torch.bool).to(self.device)
        actions = torch.tensor([x[2] for x in batch], dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.tensor([x[3] * self.reward_scale for x in batch], dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array([x[4] for x in batch]), dtype=torch.float32).to(self.device)
        next_masks = torch.tensor(np.array([x[5] for x in batch]), dtype=torch.bool).to(self.device)
        dones = torch.tensor([x[6] for x in batch], dtype=torch.float32).to(self.device)

        with torch.no_grad():
            next_actions = self.policy_net(next_states, next_masks).argmax(dim=1).unsqueeze(1)
            next_q_values = self.target_net(next_states, next_masks).gather(1, next_actions).squeeze(1)
            expected_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
            expected_q_values = torch.clamp(expected_q_values, min=-400.0, max=50.0)

        current_q_values = self.policy_net(states, masks).gather(1, actions).squeeze(1)

        loss = nn.SmoothL1Loss()(current_q_values, expected_q_values)
        if not torch.isfinite(loss):
            print(f"[warning] non-finite loss ({loss.item()}) -- skipping this gradient step")
            return None

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.grad_steps += 1
        if self.grad_steps % self.target_sync_steps == 0:
            self.update_target_network()
        self.last_q_magnitude = current_q_values.detach().abs().mean().item()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def act_greedy(self, state_tensor, mask):
        with torch.no_grad():
            t_state = torch.tensor(state_tensor, dtype=torch.float32).unsqueeze(0).to(self.device)
            t_mask = torch.tensor(mask, dtype=torch.bool).unsqueeze(0).to(self.device)
            q_values = self.policy_net(t_state, t_mask)
            return q_values.argmax(dim=1).item()


def run_episode(env, workload, agent=None, greedy=False):
    state, mask = env.reset(workload)
    done = False
    while not done:
        if greedy:
            action = agent.act_greedy(state, mask)
        else:
            action = agent.choose_action(state, mask)
        (next_state, next_mask), reward, done = env.step(action)
        if not greedy:
            agent.push_memory(state, mask, action, reward, next_state, next_mask, done)
            agent.train_step()
        state, mask = next_state, next_mask
    return env.get_finished_processes()


def evaluate_agent(agent, env, test_set):
    agg = {"mean_wait": [], "p90_wait": [], "jains_fairness": []}
    for workload in test_set:
        finished = run_episode(env, [p.clone() for p in workload], agent=agent, greedy=True)
        m = calculate_metrics(finished)
        for k in agg:
            agg[k].append(m[k])
    return {k: float(np.mean(v)) for k, v in agg.items()}


if __name__ == "__main__":
    NUM_EPISODES = 800        
    PROCESSES_PER_EPISODE = 15
    MAX_QUEUE = 10
    TEST_SET_SIZE = 30
    TARGET_UPDATE_EVERY = 10

    train_generator = WorkloadGenerator(seed=42)
    test_generator = WorkloadGenerator(seed=1234)  

    env = DirectRLEnvironment(max_queue=MAX_QUEUE)
    agent = DoubleDQNAgent(max_queue=MAX_QUEUE, in_features=DirectRLEnvironment.FEATURES)

    test_set = test_generator.generate_fixed_test_set(TEST_SET_SIZE, PROCESSES_PER_EPISODE)

    print(f"Training on device: {agent.device}")
    print("Simulating baselines on the fixed test set...")
    baseline_results = evaluate_baselines(test_set)
    for name, m in baseline_results.items():
        print(f"  {name:8s} | MeanWait: {m['mean_wait']:7.2f} | P90: {m['p90_wait']:7.2f} | Jain: {m['jains_fairness']:.3f}")

    recent_waits = deque(maxlen=50)
    recent_p90 = deque(maxlen=50)
    log_data = []

    print("\nIgniting PyTorch Direct Neural Scheduler Training...")
    for episode in range(NUM_EPISODES):
        workload = train_generator.generate_episode(PROCESSES_PER_EPISODE)
        finished = run_episode(env, workload, agent=agent, greedy=False)
        m = calculate_metrics(finished)
        recent_waits.append(m["mean_wait"])
        recent_p90.append(m["p90_wait"])

        if episode % 10 == 0:
            avg_wait = sum(recent_waits) / len(recent_waits)
            avg_p90 = sum(recent_p90) / len(recent_p90)
            log_data.append({
                "Episode": episode,
                "MovingAverageWait": avg_wait,
                "MovingAverageP90": avg_p90,
                "Epsilon": agent.epsilon,
                "QMagnitude": agent.last_q_magnitude,
            })
            print(
                f"Episode {episode:4d} | Epsilon: {agent.epsilon:.3f} | "
                f"Moving Avg Wait: {avg_wait:7.2f} | P90: {avg_p90:7.2f} | "
                f"MeanQ: {agent.last_q_magnitude:6.3f}"
            )

    print("\nEvaluating trained agent (greedy) on the fixed test set...")
    agent_eval = evaluate_agent(agent, env, test_set)
    print(
        f"  Neural Agent | MeanWait: {agent_eval['mean_wait']:7.2f} | "
        f"P90: {agent_eval['p90_wait']:7.2f} | Jain: {agent_eval['jains_fairness']:.3f}"
    )

    all_results = dict(baseline_results)
    all_results["Neural Agent"] = agent_eval
    comparison_df = pd.DataFrame(all_results).T[["mean_wait", "p90_wait", "jains_fairness"]]
    comparison_df.columns = ["Mean Wait Time", "P90 Wait Time", "Jain's Fairness Index"]
    print("\n=== Final Comparison Table ===")
    print(comparison_df.round(3).to_string())
    comparison_df.round(3).to_csv("/mnt/user-data/outputs/final_comparison_table.csv")

    df = pd.DataFrame(log_data)

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(df["Episode"], df["MovingAverageWait"], label="Neural Agent (Double DQN)", color="purple", linewidth=2)
    colors = {"FCFS": "red", "SJF": "green", "RR": "orange", "Random": "gray"}
    for name, m in baseline_results.items():
        plt.axhline(y=m["mean_wait"], color=colors[name], linestyle="--", label=f"{name} Baseline")
    plt.title("Neural Agent: Wait Time Convergence")
    plt.xlabel("Training Episodes")
    plt.ylabel("Mean Wait Time")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    schedulers = {name: (m["mean_wait"], m["jains_fairness"], colors[name]) for name, m in baseline_results.items()}
    schedulers["Neural Agent"] = (agent_eval["mean_wait"], agent_eval["jains_fairness"], "purple")

    for name, (wait, fairness, color) in schedulers.items():
        plt.scatter(wait, fairness, color=color, s=100, label=name)
        plt.text(wait + 5, fairness, name, fontsize=10, weight="bold")

    plt.title("The Pareto Frontier: Wait Time vs. Fairness")
    plt.xlabel("Mean Wait Time (Lower is Better)")
    plt.ylabel("Jain's Fairness Index (Higher is Better)")
    plt.grid(True, linestyle=":")

    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/convergence_and_pareto.png", dpi=150)
    print("\nSaved plots to /mnt/user-data/outputs/convergence_and_pareto.png")
    plt.figure(figsize=(7, 4))
    plt.plot(df["Episode"], df["QMagnitude"], color="teal")
    plt.title("Mean |Q-value| Over Training (divergence check)")
    plt.xlabel("Training Episodes")
    plt.ylabel("Mean |Q|")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/q_value_diagnostic.png", dpi=150)
    print("Saved Q-value diagnostic to /mnt/user-data/outputs/q_value_diagnostic.png")

    df.to_csv("/mnt/user-data/outputs/training_log.csv", index=False)
