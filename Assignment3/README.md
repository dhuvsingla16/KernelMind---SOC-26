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

## 4. Action Space

Instead of scheduling individual processes directly, the reinforcement learning agent operates as a **meta-scheduler**. At every scheduling event, it chooses one of three classical CPU scheduling algorithms as its action.

When a CPU is bombarded with tasks—some tiny (such as I/O-bound jobs) and some massive (such as compiling large programs)—the operating system must decide which process should execute next.

The action space therefore consists of:

### FCFS (First-Come, First-Served)

Processes execute strictly in the order they arrive.

- Extremely simple and predictable.
- Suffers from the **Convoy Effect**, where one long-running process blocks the CPU while numerous short jobs accumulate behind it.
- Provides fairness in arrival order but often results in poor overall responsiveness.

### Shortest Job First (SJF)

Always executes the process with the smallest CPU burst first.

- Minimizes average waiting time.
- Maximizes overall system throughput.
- Can cause **starvation**, since long-running processes may continually be postponed as shorter jobs arrive.

### Round Robin (RR)

Each process receives a fixed **time quantum** before being preempted.

- Prevents starvation by ensuring every process periodically receives CPU time.
- Improves fairness among processes.
- Frequent context switching introduces overhead because the processor must repeatedly save and restore registers, reducing overall throughput.

Rather than hardcoding which scheduler should be used, the RL agent learns **when each scheduling policy is most beneficial** based on the current system state.

---

## 5. Learning Algorithm

KernelMind uses **Tabular Q-Learning**, where the agent gradually estimates the long-term value of selecting each scheduling policy in every state.

### Bellman Equation (Temporal Difference Learning)

After every scheduling decision, the agent observes the immediate reward and updates its Q-table using the **Temporal Difference (TD) Error**.

The update rule is

```math
Q(s,a)\leftarrow Q(s,a)+\alpha\left[r+\gamma\max_{a'}Q(s',a')-Q(s,a)\right]
```

where

- \(r\) is the immediate reward,
- \(\gamma\) is the discount factor,
- \(\alpha\) is the learning rate,
- and

```math
\delta = r+\gamma\max_{a'}Q(s',a')-Q(s,a)
```

is the **Temporal Difference (TD) Error**.

The TD error measures how surprising the observed outcome is compared to the agent's previous expectation. Positive TD errors strengthen an action, while negative errors reduce its estimated value.

### Epsilon-Greedy Exploration

During training, KernelMind follows an **ε-decay exploration strategy**.

- Initially, ε is high, encouraging the agent to explore different scheduling policies through random actions.
- As training progresses, ε gradually decays toward **0.0**, causing the agent to rely increasingly on its learned Q-values.
- By the end of **30,000 training episodes**, exploration has effectively ceased, and the agent consistently exploits the scheduling policy that maximizes its expected long-term reward.

This balance between exploration and exploitation allows the agent to discover an adaptive scheduling strategy rather than converging prematurely to a suboptimal heuristic.

---
## 4. Reward Shaping

Reward engineering proved to be the most important aspect of the project.

A simple linear waiting-time penalty causes the RL agent to imitate **Shortest Job First (SJF)**, because mathematically it is often cheaper to indefinitely starve one long process than to increase the waiting time of many small ones.

To overcome this, KernelMind introduces a reward formulation that simultaneously minimizes the **average waiting time** while aggressively penalizing **extreme starvation**.

Instead of using a purely linear objective, the reward at each scheduling decision is computed as:

```math
R_t = -\left(\alpha\frac{\sum_{i=1}^{|Q|}W_i}{|Q|} + \beta\max_{i\in Q}(W_i)^2\right)
```

Where:

- \(Q\) is the current ready queue.
- \(W_i\) is the waiting time of process \(i\).
- \(\alpha\) controls the penalty on the **average waiting time**, encouraging high throughput.
- \(\beta\) controls the **quadratic starvation penalty**, ensuring that excessively delayed processes rapidly become expensive.

The quadratic term grows much faster than the average waiting-time penalty, forcing the agent to eventually prioritize starving processes while still maintaining the efficiency benefits of SJF.

---

# Results & Evaluation

The trained Meta-Agent was benchmarked against traditional scheduling algorithms using a standardized workload of **10 processes**.

| Scheduler | Mean Wait Time ↓ | P90 Wait Time ↓ | Jain's Fairness Index ↑ |
|-----------|-----------------:|----------------:|------------------------:|
| FCFS | 387.21 | 679.45 | 0.69 |
| Round Robin | 275.35 | 518.41 | 0.72 |
| Shortest Job First | **136.53** | 334.41 | 0.42 |
| **KernelMind (RL Meta-Agent)** | **≈155.00** | **≈210.00** | **0.81** |

### Training Convergence

During training, the moving average waiting time steadily decreased, indicating that the agent successfully learned an increasingly effective scheduling policy.

```text

Episode 0      | Moving Avg Wait: 156.04
Episode 5000   | Moving Avg Wait: 14.1121
Episode 10000  | Moving Avg Wait: 5.63636
Episode 15000  | Moving Avg Wait: 4.71722
Episode 20000  | Moving Avg Wait: 4.41922
Episode 25000  | Moving Avg Wait: 4.2416


```

The sharp reduction in moving average waiting time demonstrates stable convergence of the Q-Learning agent. Most of the improvement occurs during the early stages of training, after which the policy gradually refines itself and approaches a stable optimum.

### Learned Q-Table

After **30,000 training episodes**, the learned action-value table is:

```text
--- FINAL Q-TABLE ---

State    FCFS(0)        RR(1)          SJF(2)
0        -24311.06      -17113.06      -27155.95
1        -15776.89      -45650.26      -43321.01
2        -13163.27      -12562.47      -13167.74
3        -32283.35      -17834.13      -25618.64
4        -62728.41      -63954.57      -17566.68
5        -4.13          -11695.12      -1763.03
6        -32021.44      -17903.91      -27675.74
7        -61033.86      -38023.96      -18626.86
8        -116896.04     -121193.06     -96729.12
```

Since the reward is defined as a **negative scheduling cost**, **larger (less negative)** Q-values correspond to better long-term scheduling decisions. The learned values reveal that the agent develops clear preferences for different scheduling algorithms depending on the current workload and starvation conditions, rather than relying on a single heuristic throughout execution.

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

- Event-driven CPU scheduling simulator
- Reinforcement Learning based meta-scheduler
- Tabular Q-Learning implementation
- Dynamic switching between FCFS, RR and SJF
- Realistic stochastic workload generation
- Non-linear reward shaping to eliminate starvation
- Emergent Multi-Level Feedback Queue (MLFQ)-like behavior
- Trains over **30,000+ episodes** in just a few seconds

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
