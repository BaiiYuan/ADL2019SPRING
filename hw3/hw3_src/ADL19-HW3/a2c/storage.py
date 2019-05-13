import torch

class RolloutStorage:
    def __init__(self, n_steps, n_processes, obs_shape, action_space,
            hidden_size):
        self.obs = torch.zeros(n_steps + 1, n_processes, *obs_shape)
        self.hiddens = torch.zeros(
            n_steps + 1, n_processes, hidden_size)
        self.rewards = torch.zeros(n_steps, n_processes, 1)
        self.value_preds = torch.zeros(n_steps + 1, n_processes, 1)
        self.returns = torch.zeros(n_steps + 1, n_processes, 1)
        self.actions = torch.zeros(n_steps, n_processes, 1).long()
        self.masks = torch.ones(n_steps + 1, n_processes, 1)

        self.n_steps = n_steps
        self.step = 0

    def to(self, device):
        self.obs = self.obs.to(device)
        self.hiddens = self.hiddens.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)

    def insert(self, obs, hiddens, actions, value_preds, rewards, masks):
        self.obs[self.step + 1].copy_(obs)
        self.hiddens[self.step + 1].copy_(hiddens)
        self.actions[self.step].copy_(actions)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)

        self.step = (self.step + 1) % self.n_steps

    def reset(self):
        self.obs[0].copy_(self.obs[-1])
        self.hiddens[0].copy_(self.hiddens[-1])
        self.masks[0].copy_(self.masks[-1])

    def process_rollout(self):
        _, _, _, _, last_values = steps[-1]
        returns = last_values.data

        advantages = torch.zeros(args.num_workers, 1)
        if cuda: advantages = advantages.cuda()

        out = [None] * (self.n_steps - 1)

        # run Generalized Advantage Estimation, calculate returns, advantages
        for t in reversed(range(self.n_steps - 1)):
            rewards, masks, actions, policies, values = steps[t]
            _, _, _, _, next_values = steps[t + 1]

            returns = rewards + returns * args.gamma * masks

            deltas = rewards + next_values.data * args.gamma * masks - values.data
            advantages = advantages * args.gamma * args.lambd * masks + deltas

            out[t] = actions, policies, values, returns, advantages

        # return data as batched Tensors, Variables
        return map(lambda x: torch.cat(x, 0), zip(*out))
        return self.actions, policies, values, returns, advantages
