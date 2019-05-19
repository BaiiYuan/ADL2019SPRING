import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from agent_dir.agent import Agent
from environment import Environment
from IPython import embed

use_cuda = torch.cuda.is_available()
device = "cuda" if torch.cuda.is_available() else "cpu"

class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_num, hidden_dim, drop_p):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.dropout = nn.Dropout(drop_p)
        self.fc2 = nn.Linear(hidden_dim, action_num)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.fc2(x)
        action_prob = F.softmax(x, dim=1)
        return action_prob

class AgentPG(Agent):
    def __init__(self, env, args):
        self.env = env
        self.model = PolicyNet(state_dim = self.env.observation_space.shape[0],
                               action_num= self.env.action_space.n,
                               hidden_dim=args.hidden,
                               drop_p=args.drop_p)
        self.model = self.model.cuda() if use_cuda else self.model
        self.gamma = args.gamma

        if args.test_pg:
            self.load('pg.cpt')
        # discounted reward
        self.gamma = 0.99

        # training hyperparameters
        self.num_episodes = 100000 # total training episodes (actually too large...)
        self.display_freq = 10 # frequency to display training progress

        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-3)

        # saved rewards and actions
        self.rewards, self.saved_actions = [], []


    def save(self, save_path):
        print('save model to', save_path)
        torch.save(self.model.state_dict(), save_path)

    def load(self, load_path):
        print('load model from', load_path)
        self.model.load_state_dict(torch.load(load_path))

    def init_game_setting(self):
        self.rewards, self.saved_actions = [], []

    def make_action(self, state, test=False):
        # TODO:
        # Use your model to output distribution over actions and sample from it.
        # HINT: google torch.distributions.Categorical
        state = torch.from_numpy(state).float().unsqueeze(0)
        state = state.cuda() if use_cuda else state
        self.model(state)
        probs = self.model(state)
        m = Categorical(probs)
        action = m.sample()

        self.model.saved_log_probs.append(m.log_prob(action))
        # self.saved_actions.append(m.log_prob(action))

        return action.item()

    def update(self):
        # TODO:
        # discount your saved reward
        R = 0
        eps = np.finfo(np.float32).eps.item()
        policy_loss, returns = [], []

        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)
        for log_prob, R in zip(self.model.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)

        # TODO:
        # compute loss

        self.optimizer.zero_grad()
        loss = torch.cat(policy_loss).sum()
        loss.backward(retain_graph=True)
        self.optimizer.step()

        del self.rewards[:]
        del self.saved_actions[:]
        del self.model.saved_log_probs[:]

    def train(self):
        self.train_reward = []
        avg_reward = None # moving average of reward
        for epoch in range(self.num_episodes):
            state = self.env.reset()
            self.init_game_setting()
            done = False
            while(not done):
                action = self.make_action(state)
                state, reward, done, _ = self.env.step(action)

                self.saved_actions.append(action)
                self.rewards.append(reward)

            # for logging
            last_reward = np.sum(self.rewards)
            avg_reward = last_reward if not avg_reward else avg_reward * 0.9 + last_reward * 0.1

            # for plotting learning curve
            self.train_reward.append(last_reward)

            # update model
            self.update()

            print('\rEpochs: %d/%d | Avg reward: %f '%
                       (epoch, self.num_episodes, avg_reward), end="")
            if epoch % self.display_freq == 0:
                print("")

            if epoch > 1500: # to pass baseline, avg. reward > 50 is enough.
                self.save('pg.cpt')
                break

        np.save("pg.npy", self.train_reward)
        # self.plot_learning_curve()

    def plot_learning_curve(self):
        import matplotlib.pyplot as plt
        mean_batch = 100
        reward = [np.mean(self.train_reward[i:i+mean_batch]) for i in range(len(self.train_reward)-mean_batch)]
        train_x = [i for i in range(1, len(reward)+1)]
        plt.figure(figsize=(20,10))
        plt.plot(train_x, reward, '-', label='train')

        plt.title("Learning Curves of Policy Gradient")
        plt.xlabel("Episodes")
        plt.ylabel("Reward")

        plt.legend()
        plt.savefig("pg.png")
