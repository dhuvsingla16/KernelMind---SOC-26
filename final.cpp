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
            current_time += arrival_dist(rng); 
            double burst = std::max(1.0, burst_dist(rng) * 10.0); 
            processes.emplace_back(i + 1, current_time, burst);
        }
        return processes;
    }
};

class Scheduler {
protected:
    std::vector<Process> processes;      
    std::queue<Process*> ready_queue;    
    double current_time;
    int completed_processes;

public:
    Scheduler() : current_time(0.0), completed_processes(0) {}
    virtual ~Scheduler() = default;

    void load_processes(const std::vector<Process>& incoming) {
        processes = incoming;
        current_time = 0.0;
        completed_processes = 0;
        std::sort(processes.begin(), processes.end(), 
                  [](const Process& a, const Process& b) { return a.arrival_time < b.arrival_time; });
    }

    virtual void run() = 0;

    void print_metrics() {
        std::vector<double> wait_times;
        double total_wait = 0.0;
        double sum_of_squares = 0.0;

        std::cout << "\nPID\tArrival\tBurst\tStart\tFinish\tWait Time\n";
        std::cout << "---------------------------------------------------------\n";
        
        for (const auto& p : processes) {
            double wait_time = std::max(0.0, (p.finish_time - p.arrival_time) - p.burst_time);
            wait_times.push_back(wait_time);
            total_wait += wait_time;
            sum_of_squares += (wait_time * wait_time);

            std::cout << p.pid << "\t" << p.arrival_time << "\t" 
                      << p.burst_time << "\t" << p.start_time << "\t" 
                      << p.finish_time << "\t" << wait_time << "\n";
        }

        int n = processes.size();
        double mean_wait = total_wait / n;

        std::sort(wait_times.begin(), wait_times.end());
        int p90_index = std::ceil(0.90 * n) - 1;
        double p90_wait = wait_times[std::max(0, p90_index)];

        double jains_index = 0.0;
        double denominator = n * sum_of_squares + 1e-9; 
        jains_index = (total_wait * total_wait) / denominator;

        std::cout << "---------------------------------------------------------\n";
        std::cout << "Mean Wait Time:      " << mean_wait << "\n";
        std::cout << "P90 Wait Time:       " << p90_wait << "\n";
        std::cout << "Jain's Fairness:     " << jains_index << "\n";
    }
};

class FCFSScheduler : public Scheduler {
public:
    void run() override {
        for (auto& p : processes) {
            if (current_time < p.arrival_time) {
                current_time = p.arrival_time;
            }
            p.start_time = current_time;
            current_time += p.burst_time; 
            p.finish_time = current_time;
            p.remaining_time = 0;
        }
    }
};

class RoundRobinScheduler : public Scheduler {
private:
    double time_quantum;

public:
    RoundRobinScheduler(double quantum = 2.0) : time_quantum(quantum) {}

    void run() override {
        int index = 0;
        int n = processes.size();

        while (completed_processes < n) {
            while (index < n && processes[index].arrival_time <= current_time) {
                ready_queue.push(&processes[index]);
                index++;
            }

            if (ready_queue.empty()) {
                if (index < n) current_time = processes[index].arrival_time;
                continue;
            }

            Process* active_p = ready_queue.front();
            ready_queue.pop();

            if (active_p->start_time == -1.0) active_p->start_time = current_time;

            double time_spent = std::min(active_p->remaining_time, time_quantum);
            current_time += time_spent;
            active_p->remaining_time -= time_spent;

            while (index < n && processes[index].arrival_time <= current_time) {
                ready_queue.push(&processes[index]);
                index++;
            }

            if (active_p->remaining_time > 0) {
                ready_queue.push(active_p); 
            } else {
                active_p->finish_time = current_time;
                completed_processes++;
            }
        }
    }
};

struct CompareBurstTime {
    bool operator()(Process* const& p1, Process* const& p2) {
        return p1->burst_time > p2->burst_time;
    }
};

class SJFScheduler : public Scheduler {
private:
    std::priority_queue<Process*, std::vector<Process*>, CompareBurstTime> ready_queue;

public:
    void run() override {
        int index = 0;
        int n = processes.size();

        while (completed_processes < n) {
            while (index < n && processes[index].arrival_time <= current_time) {
                ready_queue.push(&processes[index]);
                index++;
            }

            if (ready_queue.empty()) {
                if (index < n) current_time = processes[index].arrival_time;
                continue;
            }

            Process* active_p = ready_queue.top();
            ready_queue.pop();

            active_p->start_time = current_time;
            current_time += active_p->burst_time;
            active_p->finish_time = current_time;
            active_p->remaining_time = 0;
            
            completed_processes++;
        }
    }
};

class RLEnvironment {
private:
    std::vector<Process> processes;
    std::vector<Process*> ready_queue; 
    double current_time;
    int completed_processes;
    double alpha = 1.0;  
    double beta = 0.05;  

    void load_arrived_processes() {
        for (auto& p : processes) {
            if (p.arrival_time <= current_time && p.finish_time == -1.0 && p.start_time == -1.0) {
                ready_queue.push_back(&p);
            }
        }
    }

public:
    RLEnvironment() : current_time(0.0), completed_processes(0) {}

    void reset(const std::vector<Process>& incoming) {
        processes = incoming;
        ready_queue.clear();
        current_time = 0.0;
        completed_processes = 0;
        load_arrived_processes();
    }

    bool is_done() {
        return completed_processes >= processes.size();
    }

    int get_state() {
        if (ready_queue.empty()) return 0;

        int load_state = 0;
        if (ready_queue.size() > 5) load_state = 2;
        else if (ready_queue.size() > 2) load_state = 1;

        double max_wait = 0.0;
        double total_burst = 0.0;
        for (auto* p : ready_queue) {
            double current_wait = current_time - p->arrival_time;
            if (current_wait > max_wait) max_wait = current_wait;
            total_burst += p->burst_time;
        }
        double avg_burst = total_burst / ready_queue.size();
        
        int starvation_state = 0;
        if (max_wait > avg_burst * 3) starvation_state = 2; 
        else if (max_wait > avg_burst) starvation_state = 1; 

        return (load_state * 3) + starvation_state;
    }

    double get_reward() {
        if (ready_queue.empty()) return 0.0;

        double total_wait = 0.0;
        double max_wait = 0.0;

        for (auto* p : ready_queue) {
            double current_wait = current_time - p->arrival_time;
            total_wait += current_wait;
            if (current_wait > max_wait) max_wait = current_wait;
        }

        double avg_wait = total_wait / ready_queue.size();
        return -(alpha * avg_wait + beta * (max_wait * max_wait)); 
    }

    double get_mean_wait() {
        double total_wait = 0.0;
        for (const auto& p : processes) {
            total_wait += std::max(0.0, (p.finish_time - p.arrival_time) - p.burst_time);
        }
        return total_wait / processes.size();
    }

    std::tuple<int, double, bool> step(int action) {
        if (ready_queue.empty()) {
            double next_arrival = 1e9;
            for (const auto& p : processes) {
                if (p.finish_time == -1.0 && p.arrival_time > current_time) {
                    next_arrival = std::min(next_arrival, p.arrival_time);
                }
            }
            if (next_arrival != 1e9) current_time = next_arrival;
            load_arrived_processes();
            return std::make_tuple(get_state(), 0.0, is_done());
        }

        int selected_idx = 0;

        if (action == 0) {
            selected_idx = 0; 
        } 
        else if (action == 2) {
            double min_burst = 1e9;
            for (size_t i = 0; i < ready_queue.size(); ++i) {
                if (ready_queue[i]->burst_time < min_burst) {
                    min_burst = ready_queue[i]->burst_time;
                    selected_idx = i;
                }
            }
        }

        Process* active_p = ready_queue[selected_idx];
        ready_queue.erase(ready_queue.begin() + selected_idx);

        if (active_p->start_time == -1.0) active_p->start_time = current_time;

        if (action == 1) { 
            double time_spent = std::min(active_p->remaining_time, 2.0);
            current_time += time_spent;
            active_p->remaining_time -= time_spent;

            if (active_p->remaining_time <= 0) {
                active_p->finish_time = current_time;
                completed_processes++;
            } else {
                ready_queue.push_back(active_p); 
            }
        } 
        else { 
            current_time += active_p->remaining_time;
            active_p->remaining_time = 0;
            active_p->finish_time = current_time;
            completed_processes++;
        }

        load_arrived_processes();
        return std::make_tuple(get_state(), get_reward(), is_done());
    }
};

class QLearningAgent {
private:
    double learning_rate;   
    double discount_factor; 
    double epsilon;         
    double epsilon_decay;   
    double epsilon_min;     
    int num_actions;
    
    std::vector<std::vector<double>> q_table; 
    
    std::mt19937 rng;
    std::uniform_real_distribution<double> float_dist;

public:
    QLearningAgent(int states = 9, int actions = 3, int seed = 42)
        : learning_rate(0.1), discount_factor(0.99), epsilon(1.0),
          epsilon_decay(0.9995), epsilon_min(0.0), num_actions(actions),
          q_table(states, std::vector<double>(actions, 0.0)),
          rng(seed), float_dist(0.0, 1.0) {}

    int choose_action(int state, bool training = true) {
        if (!training || float_dist(rng) > epsilon) {
            int best_action = 0;
            double max_q = q_table[state]; 
            
            for (int a = 1; a < num_actions; ++a) {
                if (q_table[state][a] > max_q) {
                    max_q = q_table[state][a];
                    best_action = a;
                }
            }
            return best_action;
        } 
        return rng() % num_actions;
    }

    void learn(int state, int action, double reward, int next_state) {
        double max_next_q = q_table[next_state]; 
        for (int a = 1; a < num_actions; ++a) {
            if (q_table[next_state][a] > max_next_q) {
                max_next_q = q_table[next_state][a];
            }
        }

        double td_target = reward + (discount_factor * max_next_q);
        double td_error = td_target - q_table[state][action];
        q_table[state][action] += learning_rate * td_error;
    }

    void decay_epsilon() {
        if (epsilon > epsilon_min) {
            epsilon *= epsilon_decay;
        } else {
            epsilon = 0.0; 
        }
    }
    
    void print_q_table() {
        std::cout << "\n--- FINAL Q-TABLE ---\n";
        std::cout << "State\tFCFS(0)\t\tRR(1)\t\tSJF(2)\n";
        for (size_t s = 0; s < q_table.size(); ++s) {
            std::cout << s << "\t";
            for (size_t a = 0; a < q_table[s].size(); ++a) {
                std::cout << std::fixed << std::setprecision(2) << q_table[s][a] << "\t\t";
            }
            std::cout << "\n";
        }
    }
};

int main() {
    int num_episodes = 30000;
    int processes_per_episode = 15;
    
    WorkloadGenerator generator(42);
    QLearningAgent agent;
    RLEnvironment env;

    std::ofstream outfile("training_log.csv");
    outfile << "Episode,MovingAverageWait\n";

    double moving_average = 0.0;
    std::vector<double> recent_waits;

    std::cout << "Starting Meta-Scheduler Training (" << num_episodes << " episodes)...\n";

    for (int episode = 0; episode < num_episodes; ++episode) {
        std::vector<Process> workload = generator.generate_episode(processes_per_episode);
        env.reset(workload);

        int state = env.get_state();
        bool done = false;

        while (!done) {
            int action = agent.choose_action(state, true); 
            
            // Universally compatible C++11 tuple extraction
            std::tuple<int, double, bool> step_result = env.step(action);
            int next_state = std::get<0>(step_result);
            double reward = std::get<1>(step_result);
            bool is_done = std::get<2>(step_result);
            
            agent.learn(state, action, reward, next_state);
            
            state = next_state;
            done = is_done;
        }

        agent.decay_epsilon();

        double ep_wait = env.get_mean_wait();
        recent_waits.push_back(ep_wait);
        if (recent_waits.size() > 100) recent_waits.erase(recent_waits.begin());

        if (episode % 100 == 0) {
            double sum = 0;
            for (double w : recent_waits) sum += w;
            moving_average = sum / recent_waits.size();
            outfile << episode << "," << moving_average << "\n";
        }

        if (episode % 5000 == 0) {
            std::cout << "Episode " << episode << " | Moving Avg Wait: " << moving_average << "\n";
        }
    }

    outfile.close();
    std::cout << "Training Complete! Data written to training_log.csv\n";
    
    agent.print_q_table();

    return 0;
}
