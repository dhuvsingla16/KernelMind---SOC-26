# KernelMind — Reinforcement Learning for Operating System Design

> **Summer of Code (SoC) 2026**  
> Department of Electrical Engineering  
> Indian Institute of Technology Bombay

---

## Author

**Dhruv Singla**  
B.Tech, Electronics Engineering  
Indian Institute of Technology Bombay

**Roll Number:** **24B1211**

---

# About the Project

KernelMind is my **Summer of Code (SoC) 2026** project exploring how **Reinforcement Learning (RL)** can be used to design intelligent Operating System components.

Modern operating systems traditionally rely on carefully engineered scheduling heuristics and handcrafted policies to manage CPU resources. While these algorithms perform well under many workloads, they are static by design and cannot adapt intelligently to changing execution patterns.

KernelMind investigates whether Reinforcement Learning can replace or augment these handcrafted scheduling decisions by allowing an agent to learn optimal behaviour directly through interaction with its environment.

The project begins with the fundamentals of Reinforcement Learning—building environments from scratch, understanding Markov Decision Processes, designing reward functions and implementing Tabular Q-Learning—and gradually progresses towards increasingly sophisticated RL-driven operating system schedulers.

Each assignment builds upon the previous one, making this repository a complete learning journey from classical Reinforcement Learning to intelligent kernel scheduling.

---

# Learning Roadmap

```
Assignment 1
      │
      ▼
Reinforcement Learning Fundamentals

      │
      ▼
Assignment 2
Physics-based RL Environment
(Project Hail Mary)

      │
      ▼
Assignment 3
RL Meta-Scheduler
(OS Scheduling)

      │
      ▼
Assignment 4
Direct Neural Scheduler
```

---

# Repository Structure

```
KernelMind---SOC-26/

│
├── Assignment-1/
│
├── Assignment-2/
│
├── Assignment-3/
│
├── Assignment-4/
│
├── Assignment4_PythonExe/
│
└── README.md
```

Each assignment contains its own

- Source Code
- Report
- Documentation
- Training Results
- Figures
- Assignment-specific README

making every module self-contained and independently executable.

---

# Assignments

## Assignment 1 — Reinforcement Learning Foundations

Introduction to the mathematical foundations of Reinforcement Learning.

Topics covered include

- Markov Decision Processes (MDPs)
- Policies
- State Value Functions
- Action Value Functions
- Bellman Equations
- Dynamic Programming
- Monte Carlo Learning
- Temporal Difference Learning

This assignment establishes the theoretical background required for the rest of the project.

---

## Assignment 2 — The Adrian Descent

A complete physics-based Reinforcement Learning environment inspired by *Project Hail Mary*.

The objective is to autonomously land a probe on the hostile planet Adrian while handling

- Strong gravity
- Atmospheric drag
- Stochastic wind disturbances
- Limited engine thrust
- Reward optimization

Major concepts explored

- Newtonian Physics
- Euler Integration
- Markov Decision Processes
- Reward Shaping
- State Discretization
- Tabular Q-Learning
- ε-Greedy Exploration
- Policy Evaluation

---

## Assignment 3 — KernelMind Meta-Scheduler

Instead of directly scheduling processes, the Reinforcement Learning agent learns **which scheduling algorithm should be selected** based on the current workload characteristics.

Scheduling algorithms available to the agent include

- First Come First Serve (FCFS)
- Shortest Job First (SJF)
- Round Robin (RR)

The scheduler dynamically switches between these classical algorithms to optimize

- Throughput
- Average Waiting Time
- Turnaround Time
- CPU Utilization
- Fairness

Topics explored

- Reinforcement Learning
- Operating Systems
- CPU Scheduling
- Meta Decision Making
- Reward Engineering
- Dynamic Scheduling Policies

---

## Assignment 4 — Direct Neural Scheduler

Assignment 4 removes handcrafted scheduling heuristics entirely.

Rather than selecting among existing scheduling algorithms, the Reinforcement Learning agent directly learns **which process should execute next**, allowing scheduling behaviour to emerge purely through experience.

Topics explored include

- Direct Policy Learning
- Neural Scheduling
- Reinforcement Learning
- Scheduling Optimization
- Intelligent Kernel Design

### Assignment 4 Implementation Note

Assignment 4 was initially implemented in **C++**, where the neural network and training pipeline were built completely from scratch without relying on external machine learning libraries.

While the implementation was functional, the handcrafted neural network struggled to converge to a satisfactory scheduling policy, making experimentation and hyperparameter tuning increasingly difficult.

To focus on the reinforcement learning aspects of the assignment rather than low-level neural network implementation, the project was subsequently migrated to **Python**, leveraging Python's scientific computing ecosystem for faster experimentation, debugging, and model training.

Both implementations are preserved in this repository.

- **Assignment-4/** — Initial C++ implementation with a handcrafted neural network.
- **Assignment4_PythonExe/** — Final and primary implementation used for experimentation, training, evaluation, and reported results.

> **Note:** The Python implementation in **Assignment4_PythonExe/** should be considered the official and recommended version of Assignment 4.

---

# Technologies Used

Throughout the project, the following technologies and concepts have been explored.

### Programming Languages

- Python
- C++

### Libraries

- NumPy
- Matplotlib

### Reinforcement Learning

- Markov Decision Processes
- Tabular Q-Learning
- Reward Shaping
- ε-Greedy Exploration
- Bellman Updates
- Policy Evaluation

### Operating Systems

- CPU Scheduling
- Meta Scheduling
- Dynamic Scheduling Policies
- Reinforcement Learning-based Scheduling

---

# Repository Goals

This repository documents the complete progression from

- Reinforcement Learning Fundamentals
- Physics-based Environment Design
- Markov Decision Processes
- Reward Engineering
- Tabular Reinforcement Learning
- Intelligent Meta Scheduling
- Direct Neural Scheduling
- Reinforcement Learning for Operating Systems

Each assignment increases the complexity of both the environment and the learning algorithm while maintaining clean implementations, detailed documentation, and reproducible experiments.

---

# Future Scope

KernelMind is intended to evolve beyond tabular reinforcement learning toward more advanced learning paradigms.

Future directions include

- Deep Q Networks (DQN)
- Policy Gradient Methods
- Actor-Critic Algorithms
- Function Approximation
- Continuous State Spaces
- Adaptive Kernel Scheduling
- Multi-Core CPU Scheduling
- Learned Resource Allocation Policies

---

# Acknowledgements

This repository is part of the **Summer of Code (SoC) 2026**.

The assignments progressively introduce Reinforcement Learning through carefully designed practical systems that bridge classical RL concepts with Operating System kernel design, ultimately demonstrating how intelligent scheduling policies can be learned rather than manually engineered.

---

## License

This repository is intended for educational and research purposes.
