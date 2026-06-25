#include <iostream>
#include <vector>
#include <queue>
#include <random>
#include <algorithm>
#include <memory>
#include <iomanip>

struct Process {
    int pid;
    double arrival_time;
    double burst_time;
    double remaining_time;
    double start_time;
    double finish_time;
    bool is_io_bound; // Flag for the future "I/O Storm" objective

    Process(int id, double arr, double burst, bool io = false)
        : pid(id), arrival_time(arr), burst_time(burst),
          remaining_time(burst), start_time(-1.0), finish_time(-1.0), is_io_bound(io) {}
};

class WorkloadGenerator {
private:
    std::mt19937 rng;
    std::poisson_distribution<int> arrival_dist;
    std::exponential_distribution<double> burst_dist;

public:
    WorkloadGenerator(int seed = 42) : rng(seed), arrival_dist(4.0), burst_dist(0.2) {}

    std::vector<Process> generate_episode(int num_processes) {
        std::vector<Process> processes;
        double current_time = 0.0;

        for (int i = 0; i < num_processes; ++i) {
            // Step time forward using Poisson distribution for arrivals
            current_time += arrival_dist(rng); 
            // Generate burst time using Exponential distribution (more short jobs, fewer massive ones)
            double burst = std::max(1.0, burst_dist(rng) * 10.0); 
            
            processes.emplace_back(i + 1, current_time, burst);
        }
        return processes;
    }
};

class Scheduler {
protected:
    std::vector<Process> processes;      // The master list of processes
    std::queue<Process*> ready_queue;    // Pointers to processes ready to run
    double current_time;
    int completed_processes;

public:
    Scheduler() : current_time(0.0), completed_processes(0) {}
    virtual ~Scheduler() = default;

    void load_processes(const std::vector<Process>& incoming) {
        processes = incoming;
        current_time = 0.0;
        completed_processes = 0;
        // Ensure processes are sorted by arrival time
        std::sort(processes.begin(), processes.end(), 
                  [](const Process& a, const Process& b) { return a.arrival_time < b.arrival_time; });
    }

    // Pure virtual function: Derived classes must implement their specific logic
    virtual void run() = 0;

    void print_metrics() {
        double total_wait = 0.0;
        std::cout << "\nPID\tArrival\tBurst\tStart\tFinish\tWait Time\n";
        std::cout << "---------------------------------------------------------\n";
        for (const auto& p : processes) {
            double wait_time = (p.finish_time - p.arrival_time) - p.burst_time;
            total_wait += wait_time;
            std::cout << p.pid << "\t" << p.arrival_time << "\t" 
                      << p.burst_time << "\t" << p.start_time << "\t" 
                      << p.finish_time << "\t" << wait_time << "\n";
        }
        std::cout << "Mean Wait Time: " << (total_wait / processes.size()) << "\n";
    }
};

// ---------------- FCFS Scheduler ----------------
class FCFSScheduler : public Scheduler {
public:
    void run() override {
        for (auto& p : processes) {
            // Fast-forward time if the CPU is idle waiting for the next arrival
            if (current_time < p.arrival_time) {
                current_time = p.arrival_time;
            }

            p.start_time = current_time;
            current_time += p.burst_time; // Non-preemptive: executes to completion
            p.finish_time = current_time;
            p.remaining_time = 0;
        }
    }
};

// ---------------- Round Robin Scheduler ----------------
class RoundRobinScheduler : public Scheduler {
private:
    double time_quantum;

public:
    RoundRobinScheduler(double quantum = 2.0) : time_quantum(quantum) {}

    void run() override {
        int index = 0;
        int n = processes.size();

        while (completed_processes < n) {
            // Load all processes that have arrived up to current_time into the ready queue
            while (index < n && processes[index].arrival_time <= current_time) {
                ready_queue.push(&processes[index]);
                index++;
            }

            if (ready_queue.empty()) {
                // CPU is idle, jump time forward to the next arrival
                if (index < n) {
                    current_time = processes[index].arrival_time;
                }
                continue;
            }

            Process* active_p = ready_queue.front();
            ready_queue.pop();

            // Record start time if this is the first time the process gets the CPU
            if (active_p->start_time == -1.0) {
                active_p->start_time = current_time;
            }

            // Execute for either the time quantum or whatever time is remaining
            double time_spent = std::min(active_p->remaining_time, time_quantum);
            current_time += time_spent;
            active_p->remaining_time -= time_spent;

            // Load any NEW arrivals that occurred during this execution slice 
            // BEFORE putting the preempted process back in the queue
            while (index < n && processes[index].arrival_time <= current_time) {
                ready_queue.push(&processes[index]);
                index++;
            }

            if (active_p->remaining_time > 0) {
                ready_queue.push(active_p); // Context switch: back to the queue
            } else {
                active_p->finish_time = current_time;
                completed_processes++;
            }
        }
    }
};

// Custom comparator to sort processes by burst time (ascending)
struct CompareBurstTime {
    bool operator()(Process* const& p1, Process* const& p2) {
        return p1->burst_time > p2->burst_time;
    }
};

// ---------------- SJF Scheduler (Non-Preemptive) ----------------
class SJFScheduler : public Scheduler {
private:
    std::priority_queue<Process*, std::vector<Process*>, CompareBurstTime> ready_queue;

public:
    void run() override {
        int index = 0;
        int n = processes.size();

        while (completed_processes < n) {
            // Load all processes that have arrived up to current_time
            while (index < n && processes[index].arrival_time <= current_time) {
                ready_queue.push(&processes[index]);
                index++;
            }

            if (ready_queue.empty()) {
                // CPU is idle; jump to the next arrival time
                if (index < n) {
                    current_time = processes[index].arrival_time;
                }
                continue;
            }

            // Extract the process with the shortest burst time
            Process* active_p = ready_queue.top();
            ready_queue.pop();

            // Execute non-preemptively to completion
            active_p->start_time = current_time;
            current_time += active_p->burst_time;
            active_p->finish_time = current_time;
            active_p->remaining_time = 0;
            
            completed_processes++;
        }
    }
};

int main() {
    // 1. Generate a testing episode
    WorkloadGenerator generator;
    std::vector<Process> workload = generator.generate_episode(10); // Generate 10 processes

    // Print the raw workload to verify generation
    std::cout << "--- GENERATED WORKLOAD ---\n";
    std::cout << "PID\tArrival\tBurst\n";
    for(const auto& p : workload) {
        std::cout << p.pid << "\t" << p.arrival_time << "\t" << p.burst_time << "\n";
    }

    // 2. Instantiate the Schedulers
    FCFSScheduler fcfs;
    RoundRobinScheduler rr(2.0); // Quantum of 2 ticks
    SJFScheduler sjf;

    // 3. Test FCFS
    std::cout << "\n\n=== FCFS RESULTS ===";
    fcfs.load_processes(workload);
    fcfs.run();
    fcfs.print_metrics();

    // 4. Test Round Robin
    std::cout << "\n\n=== ROUND ROBIN RESULTS ===";
    rr.load_processes(workload);
    rr.run();
    rr.print_metrics();

    // 5. Test Shortest Job First
    std::cout << "\n\n=== SJF RESULTS ===";
    sjf.load_processes(workload);
    sjf.run();
    sjf.print_metrics();

    return 0;
}
