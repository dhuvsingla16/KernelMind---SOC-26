#include <iostream>
#include <vector>
#include <queue>
#include <random>
#include <algorithm>
#include <memory>
#include <iomanip>
#include <fstream>
#include <tuple>
#include <cmath>

struct Process {
    int pid;
    double arrival_time;
    double burst_time;
    double remaining_time;
    double start_time;
    double finish_time;
    bool is_io_bound; 
    
    double virtual_runtime; // Cumulative ticks actually executed
    double priority;        // Priority weight for the state tensor

    Process(int id, double arr, double burst, double prio = 1.0, bool io = false)
        : pid(id), arrival_time(arr), burst_time(burst),
          remaining_time(burst), start_time(-1.0), finish_time(-1.0), 
          is_io_bound(io), virtual_runtime(0.0), priority(prio) {}
};

//  WORKLOAD GENERATOR (Reused & Tweaked) 
class WorkloadGenerator {
private:
    std::mt19937 rng;
    std::poisson_distribution<int> arrival_dist;
    std::exponential_distribution<double> burst_dist;
    std::uniform_real_distribution<double> prio_dist; // For priority

public:
    WorkloadGenerator(int seed = 42) 
        : rng(seed), arrival_dist(4.0), burst_dist(0.2), prio_dist(1.0, 10.0) {}

    std::vector<Process> generate_episode(int num_processes) {
        std::vector<Process> processes;
        double current_time = 0.0;

        for (int i = 0; i < num_processes; ++i) {
            current_time += arrival_dist(rng); 
            double burst = std::max(1.0, burst_dist(rng) * 10.0); 
            double prio = prio_dist(rng);
            processes.emplace_back(i + 1, current_time, burst, prio);
        }
        return processes;
    }
};

struct DirectState {
    std::vector<std::vector<double>> tensor; // Shape: [N_slots][3_features]
    std::vector<bool> mask;                  // True = valid process, False = padding
};

DirectState extract_state_tensor(const std::vector<Process*>& ready_queue, double current_time, int N = 10) {
    DirectState state;
    state.tensor.resize(N, std::vector<double>(3, 0.0));
    state.mask.resize(N, false);

    // Normalization constants to keep NN inputs roughly between 0 and 1
    const double MAX_BURST = 100.0; 
    const double MAX_WAIT = 500.0;
    const double MAX_PRIO = 10.0;

    for (int i = 0; i < N; ++i) {
        if (i < ready_queue.size()) {
            Process* p = ready_queue[i];
            double wait_time = current_time - p->arrival_time;

            // Feature 1: Normalized remaining burst
            state.tensor[i][0] = p->remaining_time / MAX_BURST;
            // Feature 2: Normalized wait time
            state.tensor[i][1] = std::min(wait_time / MAX_WAIT, 1.0); 
            // Feature 3: Normalized priority
            state.tensor[i][2] = p->priority / MAX_PRIO;
            
            state.mask[i] = true; // Mark as a valid target for the neural network
        } else {
            // Slot is zero-padded automatically, just ensure mask is false
            state.mask[i] = false; 
        }
    }
    return state;
}
