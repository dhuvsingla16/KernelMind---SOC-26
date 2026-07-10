import numpy as np
import matplotlib.pyplot as plt
import os
import time
from IPython.display import clear_output

class ProbeEnv:
    def __init__(self):
        self.m = 1000.0             # kg
        self.g = 13.7               # m/s^2
        self.R_adrian = 10700000.0  # m
        self.k_drag = 2.0           # Drag coefficient
        self.F_thrust = 25000.0     # N
        self.dt = 0.1               # seconds

        self.drop_altitude = 1000.0
        self.max_altitude = 1200.0
        self.max_steps = 500       # 100 seconds max

        self.wind_multipliers = [1.0, 1.5, 2.5]
        self.wind_trans = [
            [0.8, 0.2, 0.0],  # From Calm
            [0.1, 0.7, 0.2],  # From Gusty
            [0.0, 0.3, 0.7]   # From Gale
        ]
        
        self.reset()

    def reset(self):
        self.h = self.drop_altitude
        self.v = 0.0
        self.wind = 0  # Start calm
        self.step_count = 0
        return (self.h, self.v, self.wind)

    def step(self, action):
        self.step_count += 1

        self.wind = np.random.choice([0, 1, 2], p=self.wind_trans[self.wind])
        wind_mult = self.wind_multipliers[self.wind]
        
        F_gravity = -self.m * self.g * (1 - self.h / self.R_adrian)
        F_thrust_actual = action * self.F_thrust
        F_drag = wind_mult * self.k_drag * (self.v ** 2) * np.sign(-self.v)
        
        F_net = F_gravity + F_thrust_actual + F_drag
        a = F_net / self.m
        
        self.v += a * self.dt
        self.h += self.v * self.dt
        
        reward = 0.0
        done = False

        if action == 1:
            reward -= 1.0  # Fuel cost
        reward -= 1.1      # Time penalty

        if self.h <= 0:
            done = True
            self.h = 0.0
            if self.v >= -3.0:
                # Soft Catch
                reward += 1500.0 
            else:
                reward -= 100.0 * abs(self.v) 
                
        elif self.h > self.max_altitude:
            # Runaway Probe
            done = True
            reward -= 500.0
            
        elif self.step_count >= self.max_steps:
            # Hovering indefinitely
            done = True
            reward -= 500.0
            
        return (self.h, self.v, self.wind), reward, done

class ProbeAgent:
    def __init__(self):
        self.h_bins = np.linspace(0, 1200, 50)     
        self.v_bins = np.linspace(-150, 50, 40)      
        self.wind_bins = 3                          

        self.num_actions = 2

        self.q_table = np.zeros((len(self.h_bins)+1, len(self.v_bins)+1, self.wind_bins, self.num_actions))

        self.alpha = 0.13       # Learning rate
        self.gamma = 0.99      # Discount factor
        self.epsilon = 1.0     # Starting exploration
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999 

    def discretize_state(self, h, v, wind):
        h_idx = np.digitize(h, self.h_bins)
        v_idx = np.digitize(v, self.v_bins)
        return (h_idx, v_idx, wind)

    def choose_action(self, state, exploit_only=False):
        h_idx, v_idx, wind = self.discretize_state(*state)
        
        if not exploit_only and np.random.rand() < self.epsilon:
            return np.random.choice([0, 1]) # Explore
        
        # Exploit
        return np.argmax(self.q_table[h_idx, v_idx, wind])

    def learn(self, state, action, reward, next_state, done):
        h_idx, v_idx, w_idx = self.discretize_state(*state)
        nh_idx, nv_idx, nw_idx = self.discretize_state(*next_state)
        
        # Bellman Equation
        current_q = self.q_table[h_idx, v_idx, w_idx, action]
        if done:
            max_future_q = 0.0
        else:
            max_future_q = np.max(self.q_table[nh_idx, nv_idx, nw_idx])
            
        new_q = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)
        self.q_table[h_idx, v_idx, w_idx, action] = new_q

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def render_probe_ascii(h, max_h, v, action, wind, step_count, is_jupyter=True):
    if is_jupyter:
        clear_output(wait=True)
    else:
        os.system('clear' if os.name == 'posix' else 'cls')
        
    term_lines = 40
    if h > 150.0:
        display_max = max_h
        zoom_str = "[ CAMERA: WIDE ANGLE (1000m) ]"
    else:
        display_max = 150.0
        zoom_str = "[ CAMERA: TARGET APPROACH (150m) ]"

    pos = int((h / display_max) * term_lines)
    pos = max(0, min(term_lines, pos))
    
    print("-" * 75)
    wind_strs = ["Calm", "Gusty", "Adrian Gale"]
    thrust_str = "[####] ON" if action == 1 else "[    ] OFF"
    print(f" T+{step_count:03d}s | ALT: {h:6.1f}m | VEL: {v:7.1f}m/s | THRUST: {thrust_str} | WIND: {wind_strs[wind]}")
    print(f" {zoom_str}")
    
    for i in range(term_lines, -1, -1):
        if i == pos:
            if action == 1:
                print("      /\\")
                print("     /ww\\")
                print("     /||\\  <-- Spin-Drive ON")
            else:
                print("      /\\")
                print("     /--\\")
                print("     /  \\")
        else:
            if i % 10 == 0:
                print(f" {int((i/term_lines)*display_max):4d}m |")
            else:
                print("       |")
    
    print("=======[ TAUMOEBA TARGET ]============================================")
    time.sleep(0.04)

def main():
    env = ProbeEnv()
    agent = ProbeAgent()
    
    episodes = 15000
    rewards_history = []
    success_history = []
    
    print("Initiating Astrophage Spin-Drive Training Sequence...")
    
    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
        agent.decay_epsilon()
        rewards_history.append(total_reward)
        
        # Check if landing was successful
        success = 1 if (env.h <= 0 and env.v >= -3.0) else 0
        success_history.append(success)
        
        if ep % 1000 == 0:
            avg_reward = np.mean(rewards_history[-1000:])
            avg_success = np.mean(success_history[-1000:]) * 100
            print(f"Episode: {ep:5d} | Epsilon: {agent.epsilon:.3f} | Avg Reward: {avg_reward:7.1f} | Success Rate: {avg_success:5.1f}%")

    print("\nTraining Complete. Preparing for Live Drop...")
    time.sleep(2)
    
    state = env.reset()
    done = False
    
    while not done:
        action = agent.choose_action(state, exploit_only=True)
        next_state, reward, done = env.step(action)
        render_probe_ascii(env.h, env.max_altitude, env.v, action, env.wind, env.step_count, is_jupyter=True)
        state = next_state

    # Plotting
    window = 500
    moving_avg_rewards = np.convolve(rewards_history, np.ones(window)/window, mode='valid')
    moving_avg_success = np.convolve(success_history, np.ones(window)/window, mode='valid') * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(moving_avg_rewards, color='teal')
    ax1.set_title("Learning Curve (Moving Avg Reward)")
    ax1.set_xlabel("Episodes")
    ax1.set_ylabel("Reward")
    ax1.grid(True)
    
    ax2.plot(moving_avg_success, color='purple')
    ax2.set_title("Landing Success Rate (%)")
    ax2.set_xlabel("Episodes")
    ax2.set_ylabel("Success %")
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
