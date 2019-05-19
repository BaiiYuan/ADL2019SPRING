import random
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch.autograd as autograd

from agent_dir.agent import Agent
from environment import Environment
# from IPython import embed
from collections import namedtuple
from collections import deque

use_cuda = torch.cuda.is_available()
device = "cuda" if torch.cuda.is_available() else "cpu"
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if use_cuda else autograd.Variable(*args, **kwargs)

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

DOUBLE_DQN, DUEL_DQN, NOISY_DQN, PRIORITIZED_DQN = 0,0,0,0

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class DQN(nn.Module):
    '''
    This architecture is the one from OpenAI Baseline, with small modification.
    '''
    def __init__(self, channels, num_actions):
        super(DQN, self).__init__()
        # self.conv1 = nn.Conv2d(channels, 32, kernel_size=8, stride=4)
        # self.bn1 = nn.BatchNorm2d(32)
        # self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        # self.bn2 = nn.BatchNorm2d(64)
        # self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # self.bn3 = nn.BatchNorm2d(64)
        # self.fc = nn.Linear(3136, 512)
        # self.head = nn.Linear(512, num_actions)
        # self.relu = nn.ReLU()
        # self.lrelu = nn.LeakyReLU(0.01)
        self.feature = nn.Sequential(nn.Conv2d(channels, 32, kernel_size=8, stride=4),
                                     nn.BatchNorm2d(32),
                                     nn.ReLU(),
                                     nn.Conv2d(32, 64, kernel_size=4, stride=2),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(),
                                     nn.Conv2d(64, 64, kernel_size=3, stride=1),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(),
                                     )
        self.advantage = nn.Sequential(nn.Linear(3136, 512),
                                       nn.ReLU(),
                                       nn.Linear(512, num_actions)
                                       )

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        a = self.advantage(x)
        return a

class DuelDQN(nn.Module):
    '''
    This architecture is the one from OpenAI Baseline, with small modification.
    '''
    def __init__(self, channels, num_actions):
        super(DuelDQN, self).__init__()
        self.feature = nn.Sequential(nn.Conv2d(channels, 32, kernel_size=8, stride=4),
                                     nn.BatchNorm2d(32),
                                     nn.ReLU(),
                                     nn.Conv2d(32, 64, kernel_size=4, stride=2),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(),
                                     nn.Conv2d(64, 64, kernel_size=3, stride=1),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(),
                                     )
        self.advantage = nn.Sequential(nn.Linear(3136, 512),
                                       nn.ReLU(),
                                       nn.Linear(512, num_actions)
                                       )
        self.value = nn.Sequential(nn.Linear(3136, 512),
                                   nn.ReLU(),
                                   nn.Linear(512, 1)
                                   )

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        a = self.advantage(x)
        v = self.value(x)
        return v+a-a.mean()

class NoisyDQN(nn.Module):
    '''
    This architecture is the one from OpenAI Baseline, with small modification.
    '''
    def __init__(self, channels, num_actions):
        super(NoisyDQN, self).__init__()
        self.feature = nn.Sequential(nn.Conv2d(channels, 32, kernel_size=8, stride=4),
                                     nn.BatchNorm2d(32),
                                     nn.ReLU(),
                                     nn.Conv2d(32, 64, kernel_size=4, stride=2),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(),
                                     nn.Conv2d(64, 64, kernel_size=3, stride=1),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(),
                                     )
        self.noisy1 = NoisyLinear(3136, 512)
        self.noisy2 = NoisyLinear(512, num_actions)


    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.noisy1(x))
        x = self.noisy2(x)
        return x

    def reset_noise(self):
        self.noisy1.reset_noise()
        self.noisy2.reset_noise()

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
        if DUEL_DQN:
            self.model_name = 'duel_dqn'
            self.target_net = DuelDQN(self.input_channels, self.num_actions)
            self.online_net = DuelDQN(self.input_channels, self.num_actions) # policy_net
        elif NOISY_DQN:
            self.model_name = 'noisy_dqn'
            self.target_net = NoisyDQN(self.input_channels, self.num_actions)
            self.online_net = NoisyDQN(self.input_channels, self.num_actions) # policy_net
        else:
            if PRIORITIZED_DQN:
                self.model_name = 'prioritized_dqn'
            elif DOUBLE_DQN:
                self.model_name = 'double_dqn'
            else:
                self.model_name = 'dqn'

            self.target_net = DQN(self.input_channels, self.num_actions)
            self.online_net = DQN(self.input_channels, self.num_actions) # policy_net


        self.target_net = self.target_net.cuda() if use_cuda else self.target_net
        self.online_net = self.online_net.cuda() if use_cuda else self.online_net

        if args.test_dqn:
            self.load(self.model_name)

        # discounted reward
        self.GAMMA = args.gamma

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

        if PRIORITIZED_DQN:
            self.memory = NaivePrioritizedBuffer(10000)
        else:
            self.memory = ReplayBuffer(10000)

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
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps / EPS_DECAY)
        explore = (sample > eps_threshold)
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

    # def update(self):
    #     # TODO:
    #     # To update model, we sample some stored experiences as training examples.
    #     transitions = self.memory.sample(self.batch_size)
    #     batch = Transition(*zip(*transitions))
    #     next_state_mask = torch.tensor(tuple(map(lambda s: s is not None,
    #                                    batch.next_state)), device=device, dtype=torch.uint8)
    #     next_state_batch = torch.cat([s for s in batch.next_state if s is not None])

    #     state_batch = torch.cat(batch.state)
    #     action_batch = torch.cat(batch.action)
    #     reward_batch = torch.cat(batch.reward)

    #     q_values = self.online_net(state_batch)
    #     next_q_values = self.target_net(next_state_batch)
    #     next_q_state_values = self.online_net(next_state_batch)


    #     state_action_values = q_values.gather(1, action_batch)
    #     with torch.no_grad():
    #         next_state_values = torch.zeros(self.batch_size, device=device)

    #         if DOUBLE_DQN:
    #             next_state_values[next_state_mask] = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1).detach()
    #         else:
    #             next_state_values[next_state_mask] = next_q_values.max(1)[0].detach()

    #     expected_state_action_values = reward_batch + (self.GAMMA * next_state_values)
    #     loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()

    #

    #     return loss.item()

    def update(self):
        if PRIORITIZED_DQN:
            state, action, reward, next_state, done, indices, weights = self.memory.sample(self.batch_size)
        else:
            state, action, reward, next_state, done = self.memory.sample(self.batch_size)


        state      = Variable(torch.FloatTensor(np.float32(state)))
        next_state = Variable(torch.FloatTensor(np.float32(next_state)))
        # next_state = Variable(torch.FloatTensor(np.float32(next_state)), volatile=True)
        action     = Variable(torch.LongTensor(action))
        reward     = Variable(torch.FloatTensor(reward))
        done       = Variable(torch.FloatTensor(done))

        if PRIORITIZED_DQN:
            weights    = Variable(torch.FloatTensor(weights))

        state = state.permute(0,3,1,2)
        next_state = next_state.permute(0,3,1,2)

        q_values      = self.online_net(state)
        q_value          = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

        if DOUBLE_DQN:
            next_q_values = self.online_net(next_state)
            next_q_state_values = self.target_net(next_state)
            next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
        else:
            next_q_values = self.target_net(next_state)
            next_q_value = next_q_values.max(1)[0]
        expected_q_value = reward + self.GAMMA * next_q_value * (1 - done)


        if PRIORITIZED_DQN:
            loss  = (q_value - expected_q_value.detach()).pow(2) * weights
            prios = loss + 1e-5
            loss  = loss.mean()
        else:
            loss = (q_value - expected_q_value.detach()).pow(2).mean()


        self.optimizer.zero_grad()
        loss.backward()
        if PRIORITIZED_DQN:
            self.memory.update_priorities(indices, prios.data.cpu().numpy())
        self.optimizer.step()

        if NOISY_DQN:
            self.online_net.reset_noise()
            self.target_net.reset_noise()

        return loss.item()

    def train(self):
        episodes_done_num = 0 # passed episodes
        total_reward = 0 # compute average reward
        loss = 0
        self.train_reward = []
        while(True):
            state = self.env.reset()
            # State: (80,80,4) --> (1,4,80,80)

            done = False
            while(not done):
                # select and perform action
                action = self.make_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.memory.push(state, action, reward, next_state, done)
                total_reward += reward

                # if done:
                #     next_state = None

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
                    self.save(self.model_name)

                self.steps += 1

            print('\rEpisode: %d | Steps: %d/%d | Avg reward: %f | loss: %f'%
                    (episodes_done_num, self.steps, self.num_timesteps, total_reward, loss), end="")

            self.train_reward.append((self.steps, total_reward))
            total_reward = 0
            if episodes_done_num % self.display_freq == 0:
                np.save(f"{self.model_name}_{self.GAMMA}.npy", self.train_reward)
                print("")


            episodes_done_num += 1
            if self.steps > self.num_timesteps:
                break

            # if episodes_done_num > 200:
            #     break

        self.save(self.model_name)

        # embed()
        # self.plot_learning_curve()

    def plot_learning_curve(self):
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        mean_batch = 20
        reward = [np.mean(self.train_reward[i*mean_batch:(i+1)*mean_batch]) for i in range(len(self.train_reward)//mean_batch)]
        train_x = [i*mean_batch for i in range(1, len(reward)+1)]

        plt.figure(figsize=(20,10))
        plt.plot(train_x, reward, '-', label='train')

        plt.title("Learning Curves of DQN")
        plt.xlabel("Episodes")
        plt.ylabel("Reward")

        plt.legend()
        plt.savefig(f"{self.model_name}.png")


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear, self).__init__()

        self.in_features  = in_features
        self.out_features = out_features
        self.std_init     = std_init

        self.weight_mu    = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))

        self.bias_mu    = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma.mul(Variable(self.weight_epsilon))
            bias   = self.bias_mu   + self.bias_sigma.mul(Variable(self.bias_epsilon))
        else:
            weight = self.weight_mu
            bias   = self.bias_mu

        return F.linear(x, weight, bias)

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))

        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))

    def reset_noise(self):
        epsilon_in  = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x


class NaivePrioritizedBuffer(object):
    def __init__(self, capacity, prob_alpha=0.6):
        self.prob_alpha = prob_alpha
        self.capacity   = capacity
        self.buffer     = []
        self.pos        = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        assert state.ndim == next_state.ndim
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs  = prios ** self.prob_alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total    = len(self.buffer)
        weights  = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights  = np.array(weights, dtype=np.float32)

        state, action, reward, next_state, done = zip(*samples)

        return np.concatenate(state), action, reward, np.concatenate(next_state), done, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)