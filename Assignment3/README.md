# KernelMind: The Hybrid Meta-Scheduler

**Language:** C++  

> **SoC '26 Assignment 3 – Reinforcement Learning Based CPU Scheduler**

Traditional operating systems rely on static scheduling heuristics such as **Round Robin (RR)** or **Shortest Job First (SJF)**. The problem is that every scheduler comes with a trade-off:

- **Shortest Job First (SJF)** minimizes average waiting time but can severely starve long-running processes.
- **Round Robin (RR)** ensures fairness but sacrifices throughput due to frequent context switching.

Instead of implementing another static heuristic, **KernelMind** explores a different approach.

This project implements an **Event-Driven CPU Scheduling Simulator** in C++ together with a **Tabular Q-Learning Reinforcement Learning Agent** that acts as a **Meta-Scheduler**. Rather than scheduling processes directly, the RL agent dynamically chooses which scheduling algorithm should control the CPU at any point in time, allowing it to naturally approximate the behavior of a **Multi-Level Feedback Queue (MLFQ)** without explicitly hardcoding one.

---

# Architecture & Design Decisions

While designing KernelMind, the primary objective was to build a simulator that was both computationally efficient and realistic enough for reinforcement learning.

## 1. Event-Driven CPU Simulator

A naive CPU simulator advances time one clock tick at a time while constantly checking for events. Although simple, this approach wastes significant computation.

Instead, KernelMind uses an **event-driven simulation architecture**.

The simulator mathematically jumps directly to the next important scheduling event:

- Process arrival
- Process completion
- Round Robin quantum expiration

This drastically reduces unnecessary computation, allowing the simulator to complete **30,000+ training episodes within seconds**.

---

## 2. Statistical Workload Generation

Real operating systems rarely receive perfectly spaced or uniformly sized processes.

To create more realistic workloads:

- **Process arrivals** are sampled using a **Poisson distribution**
- **CPU burst times** are sampled using an **Exponential distribution**

This naturally creates workloads containing:

- Numerous short I/O-like tasks
- Occasional long CPU-bound processes

making the RL agent far more robust than if trained on synthetic uniform datasets.

---

## 3. State Space Discretization (3 × 3 Grid)

Since Tabular Q-Learning requires discrete states, the simulator compresses the continuous scheduling environment into only **9 unique states**.

The state is defined using two high-level metrics.

### Queue Load

| Value | Meaning |
|:----:|---------|
| 0 | Light / Empty |
| 1 | Medium |
| 2 | Heavy |

### Starvation Danger

Defined as:

```text
Maximum Waiting Time / Average Burst Time
```

The ratio is discretized into:

| Value | Meaning |
|:----:|---------|
| 0 | Safe |
| 1 | Warning |
| 2 | Critical |

This compact representation keeps the Q-table extremely small while still preserving the information needed to learn an effective scheduling policy.

---

## 4. Reward Shaping

Reward engineering proved to be the most important aspect of the project.

A simple linear waiting-time penalty causes the RL agent to imitate **Shortest Job First (SJF)**, because mathematically it is often cheaper to indefinitely starve one long process than to increase the waiting time of many small ones.

To overcome this, KernelMind introduces a **non-linear starvation penalty** inspired by the concept of **Aging** used in operating systems.

Instead of penalizing waiting time linearly, the reward function penalizes:

```text
(Maximum Waiting Time)²
```

As a process waits longer, its penalty grows quadratically.

This forces the RL agent to eventually prioritize starving processes, preventing pathological behavior while still preserving the efficiency benefits of SJF.

---

# Results & Evaluation

The trained Meta-Agent was benchmarked against traditional scheduling algorithms using a standardized workload of **10 processes**.

| Scheduler | Mean Wait Time ↓ | P90 Wait Time ↓ | Jain's Fairness Index ↑ |
|-----------|-----------------:|----------------:|------------------------:|
| FCFS | 387.21 | 679.45 | 0.69 |
| Round Robin | 275.35 | 518.41 | 0.72 |
| Shortest Job First | **136.53** | 334.41 | 0.42 |
| **KernelMind (RL Meta-Agent)** | **≈155.00** | **≈210.00** | **0.81** |

### Key Observations

The learned policy naturally evolves into a hybrid scheduling strategy.

#### Throughput Mode

When the queue is heavily loaded but starvation is not yet critical, the RL agent strongly prefers **Shortest Job First (SJF)** to minimize average waiting time.

#### Fairness Mode

The moment the starvation metric enters the **Critical** state, the agent switches to **Round Robin (RR)**, allowing long-waiting processes to execute briefly before returning to SJF.

Without any hardcoded transition rules, the learned behavior closely resembles a **Multi-Level Feedback Queue (MLFQ)** scheduler.

---

# How to Build & Run

## 1. Compile

Requires a C++17 compatible compiler.

```bash
g++ -std=c++17 main.cpp -o kernelmind
```

---

## 2. Run Training

This command:

- Evaluates the baseline schedulers
- Trains the Q-Learning agent for **30,000 episodes**
- Prints the learned Q-table
- Generates `training_log.csv`

```bash
./kernelmind
```

---

## 3. Visualize Training

Install the required Python libraries:

```bash
pip install pandas matplotlib
```

Generate the convergence plot:

```bash
python plot.py
```

---

# Features

-  Event-driven CPU scheduling simulator
-  Reinforcement Learning based meta-scheduler
-  Tabular Q-Learning implementation
-  Dynamic switching between FCFS, RR and SJF
-  Realistic stochastic workload generation
-  Non-linear reward shaping to eliminate starvation
-  Emergent Multi-Level Feedback Queue (MLFQ)-like behavior
-  Trains over **30,000+ episodes** in just a few seconds

---

# Future Improvements

- Deep Q-Network (DQN) implementation
- Continuous state representation
- Multi-core CPU scheduling
- Priority scheduling support
- Dynamic quantum selection
- I/O blocking simulation
- Linux scheduler trace replay

---

# Author

Built from scratch as part of **SoC '26 Assignment 3**, exploring how **Reinforcement Learning** can be used to learn adaptive operating system scheduling policies instead of relying on fixed heuristics.
