class DirectRLEnvironment {
private:
    std::vector<Process> processes;
    std::vector<Process*> ready_queue; 
    double current_time;
    int completed_processes;
    int N_SLOTS;


    double alpha = 0.5;   // Weight for virtual runtime variance (Fairness)
    double beta = 1.0;    // Weight for starvation threshold penalty
    double starvation_threshold = 100.0; 

    void load_arrived_processes() {
        for (auto& p : processes) {
            // Load processes that have arrived but haven't finished
            if (p.arrival_time <= current_time && p.finish_time == -1.0) {
                // Check if it's already in the queue to avoid duplicates
                auto it = std::find(ready_queue.begin(), ready_queue.end(), &p);
                if (it == ready_queue.end()) {
                    ready_queue.push_back(&p);
                }
            }
        }
    }

public:
    DirectRLEnvironment(int max_queue_size = 10) : current_time(0.0), completed_processes(0), N_SLOTS(max_queue_size) {}

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

    bool is_done() {
        return completed_processes >= processes.size();
    }

    // The Step Function
    std::tuple<DirectState, double, bool> step(int action_index) {
       
        if (ready_queue.empty()) {
            double next_arrival = 1e9;
            for (const auto& p : processes) {
                if (p.finish_time == -1.0 && p.arrival_time > current_time) {
                    next_arrival = std::min(next_arrival, p.arrival_time);
                }
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
            action_index = 0; // Fallback to safely execute something
        }
      
        Process* active_p = ready_queue[action_index];
        if (active_p->start_time == -1.0) {
            active_p->start_time = current_time;
        }

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

        double reward = calculate_reward() + invalid_action_penalty;

        return std::make_tuple(extract_state_tensor(ready_queue, current_time, N_SLOTS), reward, is_done());
    }

private:
    double calculate_reward() {
        if (ready_queue.empty()) return 0.0;

        double total_v_runtime = 0.0;
        for (auto* p : ready_queue) {
            total_v_runtime += p->virtual_runtime;
        }
        double mean_v_runtime = total_v_runtime / ready_queue.size();
        
        double variance = 0.0;
        double max_wait = 0.0;
        
        for (auto* p : ready_queue) {
            double diff = p->virtual_runtime - mean_v_runtime;
            variance += (diff * diff);
            
            double wait_time = (current_time - p->arrival_time) - p->virtual_runtime;
            if (wait_time > max_wait) {
                max_wait = wait_time;
            }
        }
        variance /= ready_queue.size();

        // BOUNDING: Cap the variance penalty so it doesn't explode to infinity
        double bounded_variance_penalty = std::min(variance, 100.0);

        double starvation_penalty = 0.0;
        if (max_wait > starvation_threshold) {
            double excess = max_wait - starvation_threshold;
            starvation_penalty = std::min(excess * excess * 0.1, 200.0); 
        }

        return -(alpha * bounded_variance_penalty + beta * starvation_penalty);
    }
};
