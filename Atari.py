from __future__ import annotations

import random
import torch
import torch.nn.functional as func
import collections
from collections import defaultdict

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

env = gym.make("ALE/MsPacman-v5", render_mode="rgb_array")


class ReplayBuffer:
    """经验回放"""

    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, s0: np.ndarray, a0, r, t, s1: np.ndarray):
        s0 = np.asarray(s0)
        a0 = int(a0)
        r = float(r)
        t = bool(t)
        s1 = np.asarray(s1)
        self.buffer.append((s0, a0, r, t, s1))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, terminated, next_state = zip(*transitions)
        return np.array(state), action, reward, terminated, np.array(next_state)

    # warning

    def size(self):
        return len(self.buffer)


class QNet(torch.nn.Module):
    """仅有一个hidden layer的FCN"""

    def __init__(self, state_dim):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=210, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.FC1 = torch.nn.Linear(state_dim[2] * state_dim[1] * 32, 256)
        self.FC2 = torch.nn.Linear(256, 9)
        self.relu = torch.nn.ReLU(inplace=True)
        self.__initWeights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x).view(x.size(0), -1)
        x = self.FC1(x)
        x = self.relu(x)
        x = self.FC2(x)
        x = self.relu(x)
        return x

    def __initWeights(self):
        torch.nn.init.normal_(self.conv1.weight, std=0.01)
        torch.nn.init.normal_(self.conv2.weight, std=0.01)
        torch.nn.init.normal_(self.FC1.weight, std=0.01)
        torch.nn.init.normal_(self.FC2.weight, std=0.01)
        torch.nn.init.constant_(self.conv1.bias, 0.1)
        torch.nn.init.constant_(self.conv2.bias, 0.1)
        torch.nn.init.constant_(self.FC1.bias, 0.1)
        torch.nn.init.constant_(self.FC2.bias, 0.1)


class QLearning:
    """Q-Learning"""

    def __init__(self, alpha, initial_epsilon, epsilon_decay, final_epsilon, gamma):  # 时间衰减的epsilon-Greedy
        self.Q_table = defaultdict(lambda: np.zeros(env.action_space.n))  # 初始化Q表
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        self.epsilon = initial_epsilon  # epsilon-Greedy 's parameter
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def take_action(self, state: np.ndarray) -> int:
        if np.random.random() < self.epsilon:
            action = env.action_space.sample()
        else:
            action = int(np.argmax(self.Q_table[str(state)]))  # 获取价值评估最大的值的动作下标
        return action

    def update(self, s0: np.ndarray, a0, r, t, s1: np.ndarray):
        future_Q = (not t) * np.max(self.Q_table[str(s1)])
        td_error = (r + self.gamma * future_Q - self.Q_table[str(s0)][a0])
        self.Q_table[str(s0)][a0] += self.alpha * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


class DQN:
    def __init__(self, state_dim, learning_rate, gamma, initial_epsilon, epsilon_decay, final_epsilon, target_update, device):
        self.action_dim = action_dim
        self.QNet = QNet(state_dim).to(device)

        self.target_QNet = QNet(state_dim).to(device)

        self.optimizer = torch.optim.Adam(self.QNet.parameters(), lr=learning_rate)

        self.gamma = gamma
        self.epsilon = initial_epsilon  # epsilon-Greedy 's parameter
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.target_update = target_update
        self.count = 0
        self.device = device

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def take_action(self, state: np.ndarray) -> int:
        if np.random.random() < self.epsilon:
            action = env.action_space.sample()
        else:
            state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
            action = self.QNet(state).argmax().item()  # 获取价值评估最大的值的动作下标
        return action

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards']).view(-1, 1).to(self.device)
        terminateds = torch.tensor(transition_dict['terminateds']).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)

        Q_values = self.QNet(states).gather(1, actions)

        max_next_Q_values = self.target_QNet(next_states).max(1)[0].view(-1, 1)
        Q_targets = rewards + self.gamma * max_next_Q_values * (torch.logical_not(terminateds))

        DQN_loss = torch.mean(func.mse_loss(Q_values, Q_targets))
        self.optimizer.zero_grad()
        DQN_loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_QNet.load_state_dict(
                self.QNet.state_dict()
            )
        self.count += 1


lr = 2e-3
num_episodes = 100
gamma = 0.98
init_epsilon = 1.0
epsilon_decay = init_epsilon / (num_episodes / 2)
final_epsilon = 0.01
target_update = 10
buffer_size = 100
minimal_size = 16
batch_size = 16  # minimal_size >= batch_size
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
replay_buffer = ReplayBuffer(buffer_size)
state_dim = env.observation_space.shape  # warning
action_dim = env.action_space.n
agent = DQN(state_dim, lr, gamma, init_epsilon, epsilon_decay, final_epsilon, target_update, device)

return_list = []  # 记录回报
for i in range(10):
    with tqdm(total=int(num_episodes / 10), desc='Epoch%d' % (i + 1)) as pbar:
        for i_epoch in range(int(num_episodes / 10)):
            episode_return = 0
            state, info = env.reset()
            done = False

            while not done:
                action = agent.take_action(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                replay_buffer.add(state, action, reward, terminated, next_state)
                state = next_state
                episode_return += reward
                done = truncated or terminated
                if replay_buffer.size() > minimal_size:
                    b_s, b_a, b_r, b_t, b_ns = replay_buffer.sample(batch_size)
                    transition_dict = {
                        'states': b_s,
                        'actions': b_a,
                        'rewards': b_r,
                        'next_states': b_ns,
                        'terminateds': b_t
                    }
                    agent.update(transition_dict)
            agent.decay_epsilon()
            return_list.append(episode_return)
            if (i_epoch + 1) % (int(num_episodes / 10)) == 0:
                pbar.set_postfix(({
                    'epoch': '%d' % (num_episodes / 10 * i + i_epoch + 1),
                    'return': '%.3f' % np.mean(return_list[-10:])
                }))
            pbar.update(1)

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on {}'.format('Pacman'))
plt.show()

env.close()
