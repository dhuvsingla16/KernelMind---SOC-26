void print_metrics() {
    std::vector<double> wait_times;
    double total_wait = 0.0;
    double sum_of_squares = 0.0;

    std::cout << "\nPID\tArrival\tBurst\tStart\tFinish\tWait Time\n";
    std::cout << "---------------------------------------------------------\n";
    
    for (const auto& p : processes) {
        // Ensure wait time doesn't drop below 0 due to floating point inaccuracies
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

    // --- P90 Calculation ---
    std::sort(wait_times.begin(), wait_times.end());
    int p90_index = std::ceil(0.90 * n) - 1;
    double p90_wait = wait_times[std::max(0, p90_index)];

    // --- Jain's Fairness Index ---
    double jains_index = 0.0;
    // Add a tiny epsilon to prevent divide-by-zero if all wait times are perfectly 0
    double denominator = n * sum_of_squares + 1e-9; 
    jains_index = (total_wait * total_wait) / denominator;

    std::cout << "---------------------------------------------------------\n";
    std::cout << "Mean Wait Time:      " << mean_wait << "\n";
    std::cout << "P90 Wait Time:       " << p90_wait << "\n";
    std::cout << "Jain's Fairness:     " << jains_index << "\n";
}
