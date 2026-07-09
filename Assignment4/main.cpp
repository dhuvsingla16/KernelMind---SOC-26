#include <iostream>
#include <vector>
#include <queue>
#include <random>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <tuple>
#include <cmath>

// --- 1. PROCESS & WORKLOAD ---
struct Process {
    int pid;
    double arrival_time;
    double burst_time;
    double remaining_time;
    double start_time;
    double finish_time;
    double virtual_runtime; 
    double priority;        

    Process(int id, double arr, double burst, double prio = 1.0)
        : pid(id), arrival_time(arr), burst_time(burst),
          remaining_time(burst), start_time(-1.0), finish_time(-1.0), 
          virtual_runtime(0.0), priority(prio) {}
};

class WorkloadGenerator {
private:
    std::mt19937 rng;
    std::poisson_distribution<int> arrival_dist;
    std::exponential_distribution<double> burst_dist;
    std::uniform_real_distribution<double> prio_dist;

public:
    WorkloadGenerator(int seed = 42) 
        : rng(seed), arrival_dist(4.0), burst_dist(0.2), prio_dist(1.0, 10.0) {}

    std::vector<Process> generate_episode(int num_processes) {
        std::vector<Process> processes;
        double current_time = 0.0;
        for (int i = 0; i < num_processes; ++i) {
            current_time += arrival_dist(rng); 
            double burst = std::max(1.0, burst_dist(rng) * 10.0); 
            processes.emplace_back(i + 1, current_time, burst, prio_dist(rng));
        }
        return processes;
    }
};

// --- 2. STATE TENSOR ---
struct DirectState {
    std::vector<std::vector<double>> tensor; 
    std::vector<bool> mask;                  
};

DirectState extract_state_tensor(const std::vector<Process*>& ready_queue, double current_time, int N = 10) {
    DirectState state;
    state.tensor.resize(N, std::vector<double>(3, 0.0));
    state.mask.resize(N, false);

    for (int i = 0; i < N; ++i) {
        if (i < ready_queue.size()) {
            Process* p = ready_queue[i];
            state.tensor[i][0] = p->remaining_time / 100.0;
            state.tensor[i][1] = std::min((current_time - p->arrival_time) / 500.0, 1.0); 
            state.tensor[i][2] = p->priority / 10.0;
            state.mask[i] = true; 
        }
    }
    return state;
}

// --- 3. ENVIRONMENT ---
class DirectRLEnvironment {
private:
    std::vector<Process> processes;
    std::vector<Process*> ready_queue; 
    double current_time;
    int completed_processes;
    int N_SLOTS;

    double alpha = 0.5;   
    double beta = 1.0;    
    double starvation_threshold = 100.0; 

    void load_arrived_processes() {
        for (auto& p : processes) {
            if (p.arrival_time <= current_time && p.finish_time == -1.0) {
                auto it = std::find(ready_queue.begin(), ready_queue.end(), &p);
                if (it == ready_queue.end()) ready_queue.push_back(&p);
            }
        }
    }

public:
    DirectRLEnvironment(int max_queue = 10) : current_time(0.0), completed_processes(0), N_SLOTS(max_queue) {}

    DirectState reset(const std::vector<Process>& incoming) {
        processes = incoming;
        ready_queue.clear();
        current_time = 0.0;
        completed_processes = 0;
        load_arrived_processes();
        if (ready_queue.empty() && !processes.empty()) {
            current_time = processes[0].arrival_time;
            load_arrived_processes();
        }
        return extract_state_tensor(ready_queue, current_time, N_SLOTS);
    }

    bool is_done() { return completed_processes >= processes.size(); }

    std::tuple<DirectState, double, bool> step(int action_index) {
        if (ready_queue.empty()) {
            double next_arrival = 1e9;
            for (const auto& p : processes) {
                if (p.finish_time == -1.0 && p.arrival_time > current_time)
                    next_arrival = std::min(next_arrival, p.arrival_time);
            }
            if (next_arrival != 1e9) {
                current_time = next_arrival;
                load_arrived_processes();
            }
            return std::make_tuple(extract_state_tensor(ready_queue, current_time, N_SLOTS), 0.0, is_done());
        }

        double invalid_action_penalty = 0.0;
        if (action_index >= ready_queue.size() || action_index < 0) {
            invalid_action_penalty = -50.0; 
            action_index = 0; 
        }

        Process* active_p = ready_queue[action_index];
        if (active_p->start_time == -1.0) active_p->start_time = current_time;

        active_p->remaining_time -= 1.0;
        active_p->virtual_runtime += 1.0;
        current_time += 1.0;

        if (active_p->remaining_time <= 0) {
            active_p->finish_time = current_time;
            active_p->remaining_time = 0;
            completed_processes++;
            ready_queue.erase(ready_queue.begin() + action_index);
        }
        load_arrived_processes();

        double total_v_runtime = 0.0, variance = 0.0, max_wait = 0.0;
        for (auto* p : ready_queue) total_v_runtime += p->virtual_runtime;
        if (!ready_queue.empty()) {
            double mean_v_runtime = total_v_runtime / ready_queue.size();
            for (auto* p : ready_queue) {
                double diff = p->virtual_runtime - mean_v_runtime;
                variance += (diff * diff);
                double wait = (current_time - p->arrival_time) - p->virtual_runtime;
                if (wait > max_wait) max_wait = wait;
            }
            variance /= ready_queue.size();
        }
        
        double starvation_penalty = 0.0;
        if (max_wait > starvation_threshold) {
            double excess = max_wait - starvation_threshold;
            starvation_penalty = std::min(excess * excess * 0.1, 200.0); 
        }
        
        double reward = -(alpha * std::min(variance, 100.0) + beta * starvation_penalty) + invalid_action_penalty;
        return std::make_tuple(extract_state_tensor(ready_queue, current_time, N_SLOTS), reward, is_done());
    }

    double get_mean_wait() {
        double total_wait = 0.0;
        for (const auto& p : processes) total_wait += std::max(0.0, (p.finish_time - p.arrival_time) - p.burst_time);
        return total_wait / processes.size();
    }
};

// --- 4. LIGHTWEIGHT NEURAL NETWORK ---
class SimpleNN {
public:
    std::vector<double> W1, b1, W2, b2;
    int in_size = 30, hidden_size = 64, out_size = 10;
    
    SimpleNN() {
        std::mt19937 rng(42);
        std::normal_distribution<double> dist(0.0, 0.1);
        W1.resize(in_size * hidden_size); b1.resize(hidden_size, 0.0);
        W2.resize(hidden_size * out_size); b2.resize(out_size, 0.0);
        for(auto& w : W1) w = dist(rng);
        for(auto& w : W2) w = dist(rng);
    }

    std::vector<double> forward(const std::vector<std::vector<double>>& tensor, std::vector<double>& h_out) {
        std::vector<double> input(in_size, 0.0);
        for(int i=0; i<10; ++i) for(int j=0; j<3; ++j) input[i*3+j] = tensor[i][j];
        
        h_out.assign(hidden_size, 0.0);
        for(int h=0; h<hidden_size; ++h) {
            for(int i=0; i<in_size; ++i) h_out[h] += input[i] * W1[i*hidden_size + h];
            h_out[h] = std::max(0.0, h_out[h] + b1[h]); // ReLU
        }

        std::vector<double> output(out_size, 0.0);
        for(int o=0; o<out_size; ++o) {
            for(int h=0; h<hidden_size; ++h) output[o] += h_out[h] * W2[h*out_size + o];
            output[o] += b2[o];
        }
        return output;
    }

void backprop(const std::vector<std::vector<double>>& tensor, int action, double target_q, double current_q, double lr) {
        std::vector<double> h_out;
        forward(tensor, h_out);
        
        std::vector<double> input(in_size, 0.0);
        for(int i=0; i<10; ++i) for(int j=0; j<3; ++j) input[i*3+j] = tensor[i][j];

        // 1. Calculate the raw error
        double error = target_q - current_q;
        
        // 2. GRADIENT CLIPPING (The Secret Sauce to prevent divergence)
        // We cap the error between -1.0 and 1.0 (Huber Loss approximation)
        if (error > 1.0) error = 1.0;
        if (error < -1.0) error = -1.0;

        double dL_dy = -2.0 * error; 
        
        // Update W2 and b2 for the specific action
        std::vector<double> dL_dh(hidden_size, 0.0);
        for(int h=0; h<hidden_size; ++h) {
            dL_dh[h] = dL_dy * W2[h*out_size + action];
            W2[h*out_size + action] -= lr * dL_dy * h_out[h];
        }
        b2[action] -= lr * dL_dy;

        // Update W1 and b1
        for(int h=0; h<hidden_size; ++h) {
            if(h_out[h] > 0) { // ReLU derivative
                for(int i=0; i<in_size; ++i) W1[i*hidden_size + h] -= lr * dL_dh[h] * input[i];
                b1[h] -= lr * dL_dh[h];
            }
        }
    }
};

// --- 5. DOUBLE DQN AGENT ---
class DoubleDQNAgent {
public:
    SimpleNN policy_net;
    SimpleNN target_net;
    double epsilon = 1.0, epsilon_decay = 0.995;
    std::mt19937 rng;

    DoubleDQNAgent(int seed = 42) : rng(seed) { target_net = policy_net; }

    int choose_action(const DirectState& state) {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        if (dist(rng) < epsilon) {
            std::vector<int> valids;
            for (int i = 0; i < state.mask.size(); ++i) if (state.mask[i]) valids.push_back(i);
            if(valids.empty()) return 0;
            return valids[rng() % valids.size()];
        }
        std::vector<double> h;
        std::vector<double> q = policy_net.forward(state.tensor, h);
        int best = 0; double max_q = -1e9;
        for (int i = 0; i < state.mask.size(); ++i) {
            if (!state.mask[i]) q[i] = -1e9;
            if (q[i] > max_q) { max_q = q[i]; best = i; }
        }
        return best;
    }

    void learn(const DirectState& state, int action, double reward, const DirectState& next) {
        std::vector<double> h1, h2;
        std::vector<double> next_q_pol = policy_net.forward(next.tensor, h1);
        int best_next = 0; double max_val = -1e9;
        for (int i = 0; i < next.mask.size(); ++i) {
            if (next.mask[i] && next_q_pol[i] > max_val) { max_val = next_q_pol[i]; best_next = i; }
        }
        double target_val = target_net.forward(next.tensor, h2)[best_next];
        double expected = reward + 0.99 * target_val;
        double current = policy_net.forward(state.tensor, h1)[action];
        policy_net.backprop(state.tensor, action, expected, current, 1e-4);
    }
};

// --- 6. MISSION CONTROL ---
int main() {
    int num_episodes = 500; 
    WorkloadGenerator generator(42);
    DoubleDQNAgent agent;
    DirectRLEnvironment env(10);
    std::ofstream outfile("neural_training_log.csv");
    outfile << "Episode,MovingAverageWait\n";
    std::vector<double> recent_waits;
    std::cout << "Igniting Direct Neural Scheduler Training...\n";

    for (int ep = 0; ep < num_episodes; ++ep) {
        auto workload = generator.generate_episode(15);
        DirectState state = env.reset(workload);
        while (!env.is_done()) {
            int action = agent.choose_action(state);
            auto [next_state, reward, is_done] = env.step(action);
            agent.learn(state, action, reward, next_state);
            state = next_state;
        }
        if (ep % 10 == 0) agent.target_net = agent.policy_net;
        if (agent.epsilon > 0.05) agent.epsilon *= agent.epsilon_decay;

        recent_waits.push_back(env.get_mean_wait());
        if (recent_waits.size() > 50) recent_waits.erase(recent_waits.begin());
        if (ep % 10 == 0) {
            double sum = 0; for (double w : recent_waits) sum += w;
            double avg = sum / recent_waits.size();
            outfile << ep << "," << avg << "\n";
            std::cout << "Episode " << ep << " | Moving Avg Wait: " << avg << "\n";
        }
    }
    outfile.close();
    std::cout << "Training Complete! Data written to neural_training_log.csv\n";
    return 0;
}
