import torch
import torch.nn as nn

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

def test_dqn_with_saved_model(env, model_path="dqn_model.pth", episodes=10):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = DQN(state_dim, action_dim)
    policy_net.load_state_dict(torch.load(model_path))
    policy_net.eval()

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            with torch.no_grad():
                action = torch.argmax(policy_net(torch.FloatTensor(state).unsqueeze(0))).item()
            next_state, reward, done, _, _ = env.step(action)
            state = next_state
            total_reward += reward
            env.render()
        print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}")

if __name__ == "__main__":
    from flappy import FlappyBirdEnv

    env = FlappyBirdEnv(render_mode="human")
    print("\nTesting with saved model...\n")
    test_dqn_with_saved_model(env, model_path="flappybird_dqn.pth", episodes=1)

    env.close()