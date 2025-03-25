
# First apply the patch before importing gym
from patch_gym import patch_gym
patch_gym()

import gym
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Set up device for GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Experience replay memory
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

# Q-Network architecture
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # Hyperparameters
        self.gamma = 0.99  # discount factor
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.update_target_frequency = 10  # update target network every N episodes
        self.batch_size = 64
        
        # Networks
        self.policy_net = DQN(state_size, action_size).to(device)
        self.target_net = DQN(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Set target network to evaluation mode
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.memory = ReplayMemory(10000)
        
        # Tracking metrics
        self.loss_history = []
    
    def remember(self, state, action, next_state, reward, done):
        self.memory.push(state, action, next_state, reward, done)
    
    def act(self, state, evaluate=False):
        if not evaluate and random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action_values = self.policy_net(state)
        return torch.argmax(action_values).item()
    
    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        # Convert to tensors
        state_batch = torch.FloatTensor(np.array(batch.state)).to(device)
        action_batch = torch.LongTensor(np.array(batch.action)).unsqueeze(1).to(device)
        reward_batch = torch.FloatTensor(np.array(batch.reward)).unsqueeze(1).to(device)
        
        # Handle terminal states
        non_final_mask = torch.tensor([not d for d in batch.done], dtype=torch.bool).to(device)
        non_final_next_states = torch.FloatTensor(np.array([s for s, d in zip(batch.next_state, batch.done) if not d])).to(device)
        
        # Compute Q(s_t, a)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Compute V(s_{t+1}) for all next states
        next_state_values = torch.zeros(self.batch_size, 1).to(device)
        if sum(non_final_mask) > 0:
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].unsqueeze(1).detach()
        
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        
        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
        self.loss_history.append(loss.item())
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to stabilize training
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        # Update epsilon for exploration-exploitation trade-off
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save(self, filename):
        torch.save({
            'policy_model_state_dict': self.policy_net.state_dict(),
            'target_model_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_history': self.loss_history,
            'epsilon': self.epsilon
        }, filename)
    
    def load(self, filename):
        checkpoint = torch.load(filename)
        self.policy_net.load_state_dict(checkpoint['policy_model_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.loss_history = checkpoint['loss_history']
        self.epsilon = checkpoint['epsilon']

# Training function
def train_dqn(env_name, num_episodes=500, max_steps=500, render=False):
    env = gym.make(env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = DQNAgent(state_size, action_size)
    scores = []
    
    # Training loop
    for episode in range(num_episodes):
        state, _ = env.reset()  # Assumes new Gym API
        score = 0
        
        for t in range(max_steps):
            if render:
                env.render()
            
            # Select action
            action = agent.act(state)
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store in replay memory
            agent.remember(state, action, next_state, reward, done)
            
            # Update state and score
            state = next_state
            score += reward
            
            # Learn from experiences
            agent.learn()
            
            if done:
                break
        
        # Update target network periodically
        if episode % agent.update_target_frequency == 0:
            agent.update_target_network()
        
        scores.append(score)
        avg_score = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
        
        # Print progress
        print(f"Episode: {episode+1}/{num_episodes}, Score: {score}, Average Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.2f}")
        
        # Check if environment is solved
        reward_threshold = 475.0  # CartPole-v1 is considered solved at avg score of 475+
        if avg_score >= reward_threshold:
            print(f"\nEnvironment solved in {episode+1} episodes! Average Score: {avg_score:.2f}")
            agent.save(f"{env_name}_dqn_solved.pth")
            break
    
    env.close()
    return agent, scores

# Evaluation function
def evaluate_agent(agent, env_name, num_episodes=10, render=True):
    env = gym.make(env_name)
    scores = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        score = 0
        done = False
        
        while not done:
            if render:
                env.render()
            
            action = agent.act(state, evaluate=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            score += reward
        
        scores.append(score)
        print(f"Evaluation Episode {episode+1}: Score = {score}")
    
    env.close()
    print(f"Average Evaluation Score: {np.mean(scores):.2f}")
    return scores

# Plotting function
def plot_results(scores, loss_history, title="DQN Training Results"):
    plt.figure(figsize=(15, 5))
    
    # Plot scores
    plt.subplot(1, 2, 1)
    plt.plot(scores)
    plt.plot(np.convolve(scores, np.ones(100)/100, mode='valid') if len(scores) >= 100 else scores, 'r')  # Moving average
    plt.title('DQN Score during Training')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(loss_history)
    plt.title('DQN Loss during Training')
    plt.xlabel('Learning Step')
    plt.ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.show()

# Main execution
if __name__ == "__main__":
    # Training
    env_name = "CartPole-v1"
    print(f"Training DQN agent for {env_name}...")
    agent, scores = train_dqn(env_name, num_episodes=100, max_steps=100, render=False)
    
    # Save model
    agent.save(f"{env_name}_dqn_final.pth")
    
    # Plot results
    plot_results(scores, agent.loss_history, title=f"DQN Training on {env_name}")
    
    # Evaluate the trained agent
    print("\nEvaluating the trained agent...")
    eval_scores = evaluate_agent(agent, env_name, num_episodes=5, render=True)