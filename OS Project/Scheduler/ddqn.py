import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import psutil
import os
import threading
from collections import deque

# ─────────────────────────────────────────
# Solution 2: Lightweight Neural Network
# 2 hidden layers, 64 neurons each
# ─────────────────────────────────────────
class DDQNNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DDQNNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, x):
        return self.network(x)


# ─────────────────────────────────────────
# Experience Replay Buffer
# ─────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# ─────────────────────────────────────────
# Double DQN Scheduler
# ─────────────────────────────────────────
class DDQNScheduler:

    STATE_SIZE  = 3
    ACTION_SIZE = 4
    ACTIONS = [
        psutil.IDLE_PRIORITY_CLASS,
        psutil.BELOW_NORMAL_PRIORITY_CLASS,
        psutil.NORMAL_PRIORITY_CLASS,
        psutil.ABOVE_NORMAL_PRIORITY_CLASS
    ]
    ACTION_NAMES = [
        'IDLE',
        'BELOW_NORMAL',
        'NORMAL',
        'ABOVE_NORMAL'
    ]

    def __init__(self):
        self.behavior_net = DDQNNetwork(self.STATE_SIZE, self.ACTION_SIZE)
        self.target_net   = DDQNNetwork(self.STATE_SIZE, self.ACTION_SIZE)
        self.target_net.load_state_dict(self.behavior_net.state_dict())
        self.target_net.eval()

        self.optimizer     = optim.Adam(self.behavior_net.parameters(), lr=0.001)
        self.loss_fn       = nn.MSELoss()
        self.gamma         = 0.99
        self.batch_size    = 32
        self.min_buffer    = 1000

        # epsilon greedy exploration
        self.epsilon       = 1.0
        self.epsilon_min   = 0.01
        self.epsilon_decay = 0.995

        self.replay_buffer = ReplayBuffer(capacity=10000)
        self.step_count    = 0
        self.loss_history  = []
        self.last_state    = None
        self.last_action   = None

        self.model_path = 'C:/Users/adity/OneDrive/Desktop/OS Project/dataset/ddqn_model.pth'
        self.load_model()

    def get_real_state(self):
        """s_t = (queue_length, cpu_load, memory_available) from real Windows"""
        nginx_workers = []
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                if 'nginx' in proc.info['name'].lower():
                    nginx_workers.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        queue_length     = len(nginx_workers)
        cpu_load         = psutil.cpu_percent(interval=0.1) / 100.0
        memory_available = min(psutil.virtual_memory().available / (1024**3) / 16.0, 1.0)
        return np.array([queue_length, cpu_load, memory_available], dtype=np.float32)

    def calculate_reward(self, completion_time, response_time):
        """
        Fix 2: read psutil once and reuse
        instead of calling it 3 separate times per reward calculation
        reduces CPU overhead under heavy load
        """
        cpu  = psutil.cpu_percent(interval=0.05)
        freq = psutil.cpu_freq()

        a1, a2, a3, a4, a5 = 0.3, 0.3, 0.2, 0.1, 0.1
        U        = 1 - (cpu / 100.0)
        T        = min(completion_time / 1000.0, 1.0)
        R        = min(response_time / 1000.0, 1.0)
        fairness = max(0, 1 - (T * 0.5))
        energy   = min((freq.current * cpu) / 100000.0, 1.0) if freq else 0.5
        return (a1 * U) - (a2 * T) + (a3 * (1-R)) + (a4 * fairness) - (a5 * energy)

    def select_action(self, state):
        """Epsilon greedy: random during exploration, network during exploitation"""
        if random.random() < self.epsilon:
            return random.randint(0, self.ACTION_SIZE - 1)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            return self.behavior_net(state_tensor).argmax().item()

    def train_step(self):
        """
        Double DQN update rule
        step_count is incremented here only - not in set_process_priority
        """
        if len(self.replay_buffer) < self.min_buffer:
            return None

        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states = zip(*batch)
        states      = torch.FloatTensor(np.array(states))
        actions     = torch.LongTensor(actions)
        rewards     = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))

        current_q = self.behavior_net(states).gather(1, actions.unsqueeze(1))
        with torch.no_grad():
            next_actions = self.behavior_net(next_states).argmax(1)
            next_q       = self.target_net(next_states).gather(1, next_actions.unsqueeze(1))
            target_q     = rewards.unsqueeze(1) + self.gamma * next_q

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # step_count incremented here only (removed from set_process_priority)
        self.step_count += 1

        # sync target network every 100 steps
        if self.step_count % 100 == 0:
            self.target_net.load_state_dict(self.behavior_net.state_dict())

        # decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        loss_val = loss.item()
        self.loss_history.append(loss_val)

        # save model every 500 steps
        if self.step_count > 0 and self.step_count % 500 == 0:
            self.save_model()

        return loss_val

    def set_process_priority(self, task_type, completion_time=100, response_time=100):
        """
        Main scheduling function called for every real request.

        Fix 3: train every 3rd request instead of every request
        reduces CPU overhead under heavy load significantly
        step_count is no longer incremented here to avoid double counting
        """
        current_state = self.get_real_state()
        action        = self.select_action(current_state)

        # apply real Windows priority
        try:
            proc = psutil.Process(os.getpid())
            proc.nice(self.ACTIONS[action])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

        # store experience and train
        if self.last_state is not None:
            reward = self.calculate_reward(completion_time, response_time)
            self.replay_buffer.push(self.last_state, self.last_action, reward, current_state)

            # Fix 3: only train every 3rd request to reduce heavy load overhead
            if len(self.replay_buffer) >= self.min_buffer and len(self.replay_buffer) % 3 == 0:
                loss = self.train_step()
                if loss and self.step_count % 50 == 0:
                    print(f"Step: {self.step_count} | Loss: {loss:.4f} | Epsilon: {self.epsilon:.3f} | Buffer: {len(self.replay_buffer)}")

        self.last_state  = current_state
        self.last_action = action

    def get_priority_name(self, task_type):
        if self.last_action is not None:
            return self.ACTION_NAMES[self.last_action]
        return 'NORMAL'

    def save_model(self):
        torch.save({
            'behavior_net': self.behavior_net.state_dict(),
            'target_net':   self.target_net.state_dict(),
            'epsilon':      self.epsilon,
            'step_count':   self.step_count,
            'loss_history': self.loss_history
        }, self.model_path)
        print(f"Model saved at step {self.step_count}")

    def load_model(self):
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path)
            self.behavior_net.load_state_dict(checkpoint['behavior_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.epsilon      = checkpoint['epsilon']
            self.step_count   = checkpoint['step_count']
            self.loss_history = checkpoint['loss_history']
            print(f"Model loaded. Step: {self.step_count}, Epsilon: {self.epsilon:.3f}")
        else:
            print("No saved model found. Starting fresh.")


# ─────────────────────────────────────────
# Module level shared instance and lock
# imported directly by app.py and scheduler_manager.py
# ensures only ONE agent exists across all Flask threads
# ─────────────────────────────────────────
ddqn_lock      = threading.Lock()
_ddqn_instance = DDQNScheduler()