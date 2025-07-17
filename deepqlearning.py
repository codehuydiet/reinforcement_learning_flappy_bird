import torch
import torch.nn as nn
from torch.optim import Adam
import random
from collections import deque

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

class Agent:
    def __init__(self, state_dim, action_dim, load_model_path=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = Adam(self.policy_net.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

        if load_model_path:
            self.load_state(load_model_path)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                return torch.argmax(self.policy_net(state_tensor)).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        q_values = self.policy_net(states).gather(1, actions).squeeze()
        next_q_values = self.target_net(next_states).max(1)[0]
        targets = rewards + self.gamma * next_q_values * (1 - dones)

        loss = self.criterion(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_state(self, save_path):
        torch.save({
            'model_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'memory': self.memory
        }, save_path)

    def load_state(self, load_path):
        checkpoint = torch.load(load_path)
        self.policy_net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.memory = checkpoint['memory']

def train_dqn(env, episodes=500, save_path="dqn_model.pth", load_model_path=None):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = Agent(state_dim, action_dim, load_model_path)

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.train()

            state = next_state
            total_reward += reward

        agent.update_target_network()
        agent.epsilon = max(agent.epsilon * agent.epsilon_decay, agent.epsilon_min)

        print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

        agent.save_state(save_path)

    torch.save(agent.policy_net.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    return agent

if __name__ == "__main__":
    #Huấn luyện từ đầu
    from flappy import FlappyBirdEnv
    env = FlappyBirdEnv()
    print("Training...")
    train_dqn(env, episodes=500, save_path="flappybird_dqn.pth")
    env.close()

    # #Huấn luyện tiếp từ mô hình đã lưu
    # from flappy import FlappyBirdEnv
    # env = FlappyBirdEnv()
    # load_model_path = "flappybird_dqn.pth"
    # print("Training continuing...")
    # train_dqn(env, episodes=500, save_path="flappybird_dqn.pth", load_model_path=load_model_path)
    # env.close()
