#include <fstream>
#include <iostream>
#include <vector>

int main() {
    int num_episodes = 500; 
    int processes_per_episode = 15;
    
    WorkloadGenerator generator(42); 
    DoubleDQNAgent agent;
    DirectRLEnvironment env(10); 

    std::ofstream outfile("neural_training_log.csv");
    outfile << "Episode,MovingAverageWait\n";

    double moving_average = 0.0;
    std::vector<double> recent_waits;

    std::cout << "Igniting Direct Neural Scheduler Training...\n";

    for (int episode = 0; episode < num_episodes; ++episode) {
        // Generate workload
        std::vector<Process> workload = generator.generate_episode(processes_per_episode);
        DirectState state = env.reset(workload);
        bool done = false;
        double episode_reward = 0.0;
        while (!done) {
            int action = agent.choose_action(state, true); 
            
            auto [next_state, reward, is_done] = env.step(action);
            
            // Double DQN Update
            agent.learn(state, action, reward, next_state, is_done);
            
            state = next_state;
            done = is_done;
            episode_reward += reward;
        }
      
        if (episode % 10 == 0) {
            agent.update_target_network();
        }
        agent.decay_epsilon();

        // Logging
        double ep_wait = env.get_mean_wait();
        recent_waits.push_back(ep_wait);
        if (recent_waits.size() > 50) recent_waits.erase(recent_waits.begin());

        if (episode % 10 == 0) {
            double sum = 0;
            for (double w : recent_waits) sum += w;
            moving_average = sum / recent_waits.size();
            outfile << episode << "," << moving_average << "\n";
            std::cout << "Episode " << episode << " | Moving Avg Wait: " << moving_average << "\n";
        }
    }

    outfile.close();
    std::cout << "Training Complete! Data written to neural_training_log.csv\n";
    return 0;
}
