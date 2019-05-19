import random
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from agent_dir.agent import Agent
from environment import Environment
from IPython import embed
from collections import namedtuple

use_cuda = torch.cuda.is_available()
device = "cuda" if torch.cuda.is_available() else "cpu"

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class DQN(nn.Module):
    '''
    This architecture is the one from OpenAI Baseline, with small modification.
    '''
    def __init__(self, channels, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc = nn.Linear(3136, 512)
        self.head = nn.Linear(512, num_actions)

        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU(0.01)


    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.lrelu(self.fc(x.view(x.size(0), -1)))
        q = self.head(x)
        return q

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        state, action, next_state, reward = args

        state = torch.from_numpy(state).permute(2,0,1).unsqueeze(0)
        if next_state is not None:
            next_state = torch.from_numpy(next_state).permute(2,0,1).unsqueeze(0)
        action = torch.tensor([[action]])

        if use_cuda:
            state = state.cuda()
            action = action.cuda()
            if next_state is not None:
                next_state = next_state.cuda()

        self.memory[self.position] = Transition(*(state, action, next_state, reward))
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class AgentDQN(Agent):
    def __init__(self, env, args):
        self.env = env
        self.input_channels = 4
        self.num_actions = self.env.action_space.n
        # TODO:
        # Initialize your replay buffer

        # build target, online network
        self.target_net = DQN(self.input_channels, self.num_actions)
        self.target_net = self.target_net.cuda() if use_cuda else self.target_net
        self.online_net = DQN(self.input_channels, self.num_actions) # policy_net
        self.online_net = self.online_net.cuda() if use_cuda else self.online_net

        if args.test_dqn:
            self.load('dqn')

        # discounted reward
        self.GAMMA = 0.99

        # training hyperparameters
        self.train_freq = 4 # frequency to train the online network
        self.learning_start = 10000 # before we start to update our network, we wait a few steps first to fill the replay.
        self.batch_size = args.batch_size
        self.num_timesteps = 3000000 # total training steps
        self.display_freq = 10 # frequency to display training progress
        self.save_freq = 50000 # frequency to save the model
        self.target_update_freq = 1000 # frequency to update target network

        # optimizer
        self.optimizer = optim.RMSprop(self.online_net.parameters(), lr=1e-4)

        self.steps = 0 # num. of passed steps. this may be useful in controlling exploration

        self.memory = ReplayMemory(10000)

    def save(self, save_path):
        print('save model to', save_path)
        torch.save(self.online_net.state_dict(), save_path + '_online.cpt')
        torch.save(self.target_net.state_dict(), save_path + '_target.cpt')

    def load(self, load_path):
        print('load model from', load_path)
        if use_cuda:
            self.online_net.load_state_dict(torch.load(load_path + '_online.cpt'))
            self.target_net.load_state_dict(torch.load(load_path + '_target.cpt'))
        else:
            self.online_net.load_state_dict(torch.load(load_path + '_online.cpt', map_location=lambda storage, loc: storage))
            self.target_net.load_state_dict(torch.load(load_path + '_target.cpt', map_location=lambda storage, loc: storage))

    def init_game_setting(self):
        # we don't need init_game_setting in DQN
        pass

    def make_action(self, state, test=False):
        state = torch.from_numpy(state).permute(2,0,1).unsqueeze(0)
        state = state.cuda() if use_cuda else state
        # TODO:
        # At first, you decide whether you want to explore the environemnt
        if test:
            action = self.online_net(state).max(1)[1].view(1, 1)
            return action[0, 0].data.item()

        # TODO:
        explore = (sample > eps_threshold)
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps / EPS_DECAY)
        # if explore, you randomly samples one action
        # else, use your model to predict action
        if explore:
            with torch.no_grad():
                action = self.online_net(state).max(1)[1].view(1, 1)
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
        else:
            action = torch.tensor([[random.randrange(self.num_actions)]], device=device, dtype=torch.long)

        return action[0, 0].data.item()

    def update(self):
        # TODO:
        # To update model, we sample some stored experiences as training examples.
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # TODO:
        # Compute Q(s_t, a) with your model.
        state_action_values = self.online_net(state_batch).gather(1, action_batch)
        with torch.no_grad():
            # TODO:
            # Compute Q(s_{t+1}, a) for all next states.
            # Since we do not want to backprop through the expected action values,
            # use torch.no_grad() to stop the gradient from Q(s_{t+1}, a)
            next_state_values = torch.zeros(self.batch_size, device=device)
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        # TODO:
        # Compute the expected Q values: rewards + gamma * max(Q(s_{t+1}, a))
        # You should carefully deal with gamma * max(Q(s_{t+1}, a)) when it is the terminal state.
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch
        # TODO:
        # Compute temporal difference loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.online_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss.item()

    def train(self):
        episodes_done_num = 0 # passed episodes
        total_reward = 0 # compute average reward
        loss = 0
        while(True):
            state = self.env.reset()
            # State: (80,80,4) --> (1,4,80,80)

            done = False
            while(not done):
                # select and perform action
                action = self.make_action(state)
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward
                reward = torch.tensor([reward], device=device)

                if done:
                    next_state = None

                # TODO:
                # store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # move to the next state
                state = next_state
                # Perform one step of the optimization
                if self.steps > self.learning_start and (self.steps % self.train_freq == 0):
                    loss = self.update()

                # update target network
                if self.steps > self.learning_start and self.steps % self.target_update_freq == 0:
                    self.target_net.load_state_dict(self.online_net.state_dict())

                # save the model
                if self.steps % self.save_freq == 0:
                    self.save('dqn')

                self.steps += 1

            print('\rEpisode: %d | Steps: %d/%d | Avg reward: %f | loss: %f'%
                    (episodes_done_num, self.steps, self.num_timesteps, total_reward / self.display_freq, loss), end="")
            if episodes_done_num % self.display_freq == 0:
                total_reward = 0
                print("")

            episodes_done_num += 1
            if self.steps > self.num_timesteps:
                break
        self.save('dqn')
