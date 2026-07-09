class DoubleDQNAgent {
private:
    NeuralNetwork policy_net;
    NeuralNetwork target_net;
    
    double learning_rate = 1e-4;
    double discount_factor = 0.99;
    double epsilon = 1.0;
    double epsilon_min = 0.05;
    double epsilon_decay = 0.999;
    
    std::mt19937 rng;
    std::uniform_real_distribution<double> float_dist;

public:
    DoubleDQNAgent(int seed = 42) : rng(seed), float_dist(0.0, 1.0) {
        target_net.copy_weights_from(policy_net);
    }

    int choose_action(const DirectState& state, bool training = true) {
        if (training && float_dist(rng) < epsilon) {
            std::vector<int> valid_indices;
            for (int i = 0; i < state.mask.size(); ++i) {
                if (state.mask[i]) valid_indices.push_back(i);
            }
            if (valid_indices.empty()) return 0; // Failsafe
            return valid_indices[rng() % valid_indices.size()];
        }

        std::vector<double> q_values = policy_net.forward(state.tensor);
        double max_q = -1e9;
        int best_action = 0;

        for (int i = 0; i < state.mask.size(); ++i) {
            if (!state.mask[i]) {
                q_values[i] = -1e9; // Mask out padding
            }
            if (q_values[i] > max_q) {
                max_q = q_values[i];
                best_action = i;
            }
        }
        return best_action;
    }

    void learn(const DirectState& state, int action, double reward, const DirectState& next_state, bool done) {
        std::vector<double> next_q_policy = policy_net.forward(next_state.tensor);
        int best_next_action = 0;
        double max_val = -1e9;
        for (int i = 0; i < next_state.mask.size(); ++i) {
            if (next_state.mask[i] && next_q_policy[i] > max_val) {
                max_val = next_q_policy[i];
                best_next_action = i;
            }
        }

        std::vector<double> next_q_target = target_net.forward(next_state.tensor);
        double target_q_value = next_q_target[best_next_action];

        double expected_q = reward;
        if (!done) {
            expected_q += discount_factor * target_q_value;
        }

        double current_q = policy_net.forward(state.tensor)[action];
        double loss = std::pow(expected_q - current_q, 2.0); // Mean Squared Error
        policy_net.backpropagate(state.tensor, action, expected_q);
    }
    
    void update_target_network() {
        target_net.copy_weights_from(policy_net);
    }
};
