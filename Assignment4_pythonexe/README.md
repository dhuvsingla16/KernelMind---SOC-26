# KernelMind: The Direct Neural Scheduler

**Repository:** `dhuvsingla16/KernelMind---SOC-26`

Welcome to my implementation for **SoC '26 Assignment 4: The Direct Neural Scheduler**.

In **Assignment 3**, KernelMind functioned as a **Meta-Scheduler**. Rather than making scheduling decisions itself, it learned to dynamically select between classical operating system scheduling algorithms such as **First-Come First-Served (FCFS)**, **Shortest Job First (SJF)**, and **Round Robin (RR)** depending on the current workload.

This assignment removes that abstraction completely.

Instead of selecting from a predefined set of heuristics, the scheduler now performs **direct process dispatching**. At every scheduling decision, the reinforcement learning agent observes the contents of the ready queue and directly predicts **which specific process should receive the CPU next**.

This dramatically increases the complexity of the scheduling problem. The agent must now discover scheduling behavior completely from experience instead of relying on handcrafted heuristics.

To enable this transition, the entire system was redesigned:

- Transitioned from **Tabular Q-Learning** to a **PyTorch-based Double Deep Q-Network (DDQN)**
- Replaced handcrafted state tables with learned neural representations
- Introduced **Transformer-style Multi-Head Self-Attention** for queue modeling
- Designed a fully event-driven simulator capable of generating thousands of scheduling episodes efficiently
- Implemented advanced stabilization techniques including experience replay, target networks, Huber loss, gradient clipping, and action masking

---

# Table of Contents

- [System Architecture](#system-architecture)
- [Event-Driven CPU Simulator](#1-event-driven-cpu-simulator)
- [State Space Representation](#2-state-space-representation)
- [Neural Network Architecture](#3-neural-network-architecture)
- [Double Deep Q-Network](#4-double-deep-q-network-ddqn)
- [Reward Function Design](#5-reward-function-design)
- [Training Pipeline](#training-pipeline)
- [Design Questions](#design-question-answers)
- [Experimental Results](#final-evaluation--results)
- [Installation](#installation)
- [Running the Project](#running-the-project)

---

# System Architecture

The complete scheduling pipeline is illustrated below.

```
Process Generator
        │
        ▼
 Ready Queue (Variable Length)
        │
        ▼
 Fixed Window Representation (10 Slots)
        │
        ▼
 Feature Normalization
        │
        ▼
 Positional Encoding
        │
        ▼
 Multi-Head Self Attention
        │
        ▼
 Feed Forward Network
        │
        ▼
 Q-Values for Every Queue Slot
        │
        ▼
 Action Masking
        │
        ▼
 Selected Process Index
        │
        ▼
 CPU Execution
        │
        ▼
 Reward + Next State
        │
        ▼
 Replay Buffer
        │
        ▼
 Double DQN Update
```

Unlike traditional schedulers that rely on explicit heuristics, every scheduling decision emerges purely from reinforcement learning.

---

# 1. Event-Driven CPU Simulator

The environment is implemented as a custom **event-driven CPU scheduling simulator**.

Unlike a conventional simulator that advances one CPU tick at a time, the simulator mathematically jumps directly to the next scheduling event.

Critical events include:

- Process arrival
- Process completion
- CPU becoming idle

Whenever no meaningful work occurs, the simulator fast-forwards time instead of wasting computation.

This dramatically reduces training time since millions of unnecessary idle iterations are avoided.

## Workload Generation

Each episode contains randomly generated workloads.

Arrival times follow a **Poisson process**, producing realistic random task arrivals.

CPU burst lengths follow an **Exponential distribution**, naturally producing many short jobs and a few extremely long jobs.

This creates workloads containing a healthy mixture of

- CPU-bound processes
- I/O-bound processes
- Short interactive jobs
- Long computational jobs

which closely resembles real operating system behavior.

---

# 2. State Space Representation

The ready queue has variable length.

Neural networks require fixed-size tensors.

Therefore, the queue is represented using a **fixed observation window of N = 10 processes**.

If fewer than ten processes are available, remaining slots are padded with zeros.

Each process is encoded using **five normalized features**.

| Feature | Description |
|----------|-------------|
| Remaining Burst | Remaining execution time divided by 100 |
| Wait Time | Current waiting time divided by 500 |
| Priority | Process priority divided by 10 |
| Virtual Runtime | Total CPU time already received divided by 100 |
| Starvation Gap | `(wait − virtual_runtime − threshold)` clamped to `[-1,1]` |

The normalization ensures every feature lies inside either

```
[0,1]
```

or

```
[-1,1]
```

which greatly improves gradient stability during training.

## Why Virtual Runtime?

Virtual Runtime measures **how much CPU service a process has already received**, rather than how long it has waited.

This allows the scheduler to reason about fairness similarly to the Linux Completely Fair Scheduler (CFS).

---

# 3. Neural Network Architecture

A standard Multi-Layer Perceptron (MLP) would require flattening the queue into one long vector.

Doing so introduces undesirable positional biases such as

- "slot 3 is inherently different from slot 7"

even when those slots simply contain different processes.

Instead, the scheduler models the queue as a sequence.

## Positional Encoding

Learnable positional embeddings are added to every queue slot.

This enables the network to understand queue ordering while still learning flexible representations.

---

## Multi-Head Self-Attention

The core of the scheduler is a Transformer-inspired attention layer.

Every process is allowed to attend to every other process simultaneously.

Rather than examining processes independently, the network learns relationships such as

- Which process has waited the longest?
- Which process is almost complete?
- Which processes are starving?
- Which scheduling decision minimizes future penalties?

This relational reasoning is significantly more expressive than handcrafted scheduling heuristics.

---

## Residual Connections

The attention block uses

- Layer Normalization
- Residual Skip Connections

which improve optimization stability and allow deeper feature learning.

---

## Double Masking Strategy

Since many queue slots may simply be padding, two masking mechanisms are used.

### Attention Key Padding Mask

The Transformer attention layer completely ignores padded entries.

Therefore, empty queue slots never influence attention scores.

---

### Action Space Masking

The final linear layer predicts one Q-value for every queue slot.

Before selecting an action,

all padded slots are forced to

```
-1 × 10^9
```

ensuring that the agent can never select a nonexistent process.

---

# 4. Double Deep Q-Network (DDQN)

KernelMind uses **Double Deep Q-Learning** instead of standard DQN.

Double DQN significantly reduces the overestimation bias commonly observed in classical Q-learning.

---

## Q-Value Estimation

The network outputs

\
$Q(S,a)\$


for every valid process.

Since the reward is formulated as a penalty,

higher Q-values correspond to

**lower expected cumulative future penalties.**

The scheduler therefore selects

\$\arg\max_a Q(S,a)\$

at inference time.

---

## Epsilon-Greedy Exploration

Training uses an ε-greedy exploration policy.

Initially,

```
ε = 1.0
```

meaning actions are almost completely random.

As training progresses,

ε gradually decays until

```
ε = 0.05
```

allowing the network to shift from exploration toward exploitation.

During evaluation,

```
ε = 0
```

so the learned policy is executed deterministically.

---

## Experience Replay

Every interaction is stored as

\
$(S_t,A_t,R_{t+1},S_{t+1})$


inside a replay memory.

Training samples random mini-batches of

```
64 transitions
```

which breaks temporal correlation and stabilizes learning.

---

## Bellman Target

The Double DQN target is computed as

\$Y_t =
R_{t+1}
+
\gamma
Q_{target}
\left(
S_{t+1},
\arg\max_a
Q_{policy}(S_{t+1},a)
\right)\$

Using separate policy and target networks prevents unstable bootstrap updates.

---

## Huber Loss

Training minimizes

```
SmoothL1Loss
```

(Huber Loss)

instead of Mean Squared Error.

Huber Loss behaves quadratically for small errors and linearly for large errors, naturally limiting extremely large temporal-difference updates.

Gradient clipping is additionally applied using

```
clip_grad_norm_(1.0)
```

to prevent exploding gradients.

---

# 5. Reward Function Design

The reward function determines the scheduling behavior learned by the agent.

Using only average waiting time would encourage the scheduler to endlessly prioritize short jobs, causing severe starvation.

Instead, the reward is a weighted combination of multiple objectives.

---

## CFS Fairness Penalty

The scheduler penalizes the variance of

```
Virtual Runtime
```

across all runnable processes.

This encourages CPU time to be distributed fairly among competing jobs.

---

## Starvation Penalty

Once a process exceeds a predefined waiting threshold,

a bounded quadratic penalty activates.

This aggressively discourages starvation while avoiding unbounded gradients.

---

## Queue Length Penalty

A penalty proportional to queue length encourages the scheduler to complete jobs quickly.

Interestingly, this naturally pushes the learned policy toward SJF-like behavior without explicitly programming SJF.

---

# Training Pipeline

Each training episode follows the sequence below.

1. Generate a random scheduling workload.
2. Construct the normalized queue representation.
3. Pass the queue through the Transformer-based DDQN.
4. Select an action using ε-greedy exploration.
5. Execute the selected process.
6. Observe reward and next state.
7. Store the transition in replay memory.
8. Sample a random mini-batch.
9. Compute the Double DQN Bellman target.
10. Update network parameters.
11. Synchronize the target network periodically.

This loop is repeated for **3000 training episodes**.

---

# Design Question Answers

## Design Question 1 — The Cost of Direct Control

### Why does direct per-tick scheduling make credit assignment harder?

In Assignment 3, choosing a scheduling heuristic such as Round Robin automatically determined many future scheduling decisions.

Here, the agent decides **every single scheduling tick**.

Suppose the scheduler makes a poor decision at tick 5.

The resulting performance degradation may not become visible until tick 65 after dozens of subsequent scheduling decisions.

The Bellman update must therefore propagate blame through a very long sequence of actions.

This makes temporal credit assignment substantially harder than in heuristic selection.

---

## Design Question 2 — Reward Shaping

### Why use Virtual Runtime variance instead of Wait Time variance?

Raw waiting time creates a misleading optimization signal.

Suppose a heavily starved process finally begins execution.

While that process runs,

every other process continues accumulating waiting time.

Consequently,

the overall variance of waiting time may actually increase,

despite the scheduler making the correct decision.

Virtual Runtime behaves differently.

As soon as a neglected process receives CPU time,

its virtual runtime immediately moves toward the average,

reducing the fairness penalty.

This provides a clean and immediate learning signal.

---

## Design Question 3 — Diagnosing Instability

### How was training instability resolved?

Early experiments consistently plateaued around

```
Mean Wait Time ≈ 330
```

which performed worse than Round Robin.

Logging the average absolute Q-value revealed numerical divergence,

eventually producing NaN values.

Two major issues were identified.

### 1. Stale Target Network

Initially,

the target network synchronized every

```
10 episodes
```

which corresponded to thousands of optimization steps.

The policy network therefore bootstrapped from severely outdated estimates.

The solution was to synchronize

**every 150 gradient updates**

instead.

---

### 2. Gradient Explosion

The quadratic starvation penalty occasionally generated extremely large temporal-difference errors.

Training was stabilized using

- Huber Loss
- Gradient Clipping
- Better target synchronization

After these changes,

training converged consistently.

---

## Design Question 4 — Pareto Frontier Analysis

### Does the neural scheduler outperform classical heuristics?

Yes.

The learned scheduler occupies a favorable point on the throughput–fairness tradeoff.

Compared with SJF,

it sacrifices only a small amount of average waiting time,

while substantially improving fairness and reducing worst-case waiting time.

Compared with Round Robin,

it preserves fairness while dramatically improving throughput.

This demonstrates that reinforcement learning can discover scheduling strategies beyond traditional handcrafted heuristics.

---

# Final Evaluation & Results

The final policy was evaluated after

```
3000 training episodes
```

using a fixed unseen benchmark consisting of

```
30 workloads
```

with

```
ε = 0
```

for deterministic evaluation.

| Scheduler | Mean Wait ↓ | P90 Wait ↓ | Jain Fairness ↑ |
|------------|------------:|-----------:|----------------:|
| FCFS | 323.39 | 585.98 | 0.413 |
| RR | 330.02 | 549.46 | 0.823 |
| SJF | **159.86** | 413.72 | 0.713 |
| **Neural DDQN Scheduler** | **168.45** | **380.15** | **0.852** |

---

# Analysis

## Throughput

The neural scheduler learns behavior remarkably close to Shortest Job First.

Average waiting time differs by less than ten time units while remaining significantly more balanced.

---

## Worst-Case Waiting Time

The 90th percentile waiting time improves from

```
413.72
```

under SJF

to

```
380.15
```

under the learned scheduler.

This indicates that the network successfully identifies starvation before it becomes severe.

---

## Fairness

The learned scheduler achieves the highest Jain Fairness Index among all evaluated schedulers.

This demonstrates that the reward shaping strategy successfully balanced efficiency and fairness simultaneously.

---

# Installation

Install the required Python packages.

```bash
pip install torch numpy pandas matplotlib
```

---

# Running the Project

Execute the training script.

```bash
python kernelmind.py
```

The script will automatically

- generate randomized scheduling workloads
- simulate FCFS, RR, and SJF baselines
- train the Double Deep Q-Network scheduler
- periodically synchronize the target network
- evaluate the learned greedy policy
- generate convergence plots
- generate Pareto frontier visualizations
- report quantitative benchmarking statistics

---

# Key Technologies

- Python
- PyTorch
- Double Deep Q-Network (DDQN)
- Transformer Self-Attention
- Reinforcement Learning
- Event-Driven Simulation
- Experience Replay
- Gradient Clipping
- Huber Loss
- CPU Scheduling
- Operating Systems

