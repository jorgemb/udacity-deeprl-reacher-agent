from turtle import forward
import torch
import torch.nn as nn
import numpy as np

import rllib.ReplayBuffer
from AbstractAgent import AbstractAgent


class LinearModel(nn.Module):
    def __init__(self, state_space, action_space, hidden=(32, 32), activation_fn: any = nn.ReLU,
                 end_activation_fn: any = nn.ReLU) -> None:
        super().__init__()

        self.layers = nn.Sequential()

        # Create hidden layers
        last_size = state_space
        for i, val in enumerate(hidden):
            self.layers.add_module(f'fc{i}', nn.Linear(last_size, val))
            self.layers.add_module(f'act{i}', activation_fn())
            last_size = val

            # Add output layer
            if i == len(hidden) - 1:
                self.layers.add_module(f'fc_end', nn.Linear(last_size, action_space))
                if end_activation_fn is not None:
                    self.layers.add_module(f'act_end', end_activation_fn())

    def forward(self, x):
        return self.layers.forward(x)


class DDPGAgent(AbstractAgent):
    def __init__(self, state_size, action_size, *, gamma=1.0, alpha=0.1, seed=-1, **kwargs) -> None:
        super().__init__(state_size, action_size, gamma=gamma, alpha=alpha, seed=seed, **kwargs)

        # Create actor-critic networks
        self.local_actor = LinearModel(state_size, action_size, end_activation_fn=nn.Tanh).to(self.device)
        self.target_actor = LinearModel(state_size, action_size, end_activation_fn=nn.Tanh).to(self.device)

        self.local_critic = LinearModel(state_size + action_size, 1).to(self.device)
        self.target_critic = LinearModel(state_size + action_size, 1).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.local_actor.parameters(), lr=self.alpha)
        self.critic_optimizer = torch.optim.Adam(self.local_critic.parameters(), lr=self.alpha)

        self.last_state = None
        self.last_action = None
        self.step_n = 0

        # Hyperparameters
        self.buffer_size = kwargs.get('buffer_size')
        self.batch_size = kwargs.get('batch_size')
        self.tau = kwargs.get('tau')
        self.learn_every = kwargs.get('learn_every', 10)

        self.replay = rllib.ReplayBuffer.ReplayBuffer(
            action_size,
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            device=self.device
        )

    def take_action(self, state: np.array, do_random: bool = False):
        """
        Takes an action using the local actor or a random value
        @param state:
        @param do_random:
        @return:
        """
        if do_random:
            return np.random.uniform(-1, 1, self.action_size)
        else:
            x = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            return self.target_actor.forward(x).cpu().data.numpy()

    def start(self, state):
        self.step_n = 1

        # Take and action and store
        self.last_state = np.asarray(state, dtype=float)
        self.last_action = self.take_action(state, True)
        return self.last_action

    def step(self, state, reward, learn=True):
        # Store the SARS information in the replay buffer
        next_state = np.asarray(state, dtype=float)
        self.replay.add(self.last_state, self.last_action, reward, next_state, False)

        # Get next action
        self.last_state = next_state
        self.last_action = self.take_action(state)
        self.step_n += 1

        # Check if we should do a learning step
        if learn and len(self.replay) > self.batch_size and self.step_n % self.learn_every == 0:
            self.learn_step()

        return self.last_action

    def learn_step(self):
        states, actions, rewards, next_states, dones = self.replay.sample()

        # Forward pass with target networks
        a_next = self.target_actor.forward(next_states)
        q_next = self.target_critic.forward(torch.cat([next_states, a_next], dim=1))
        q_next = self.gamma * dones * q_next + rewards
        q_next = q_next.detach()

        # Pass with local networks
        q = self.local_critic(torch.cat([states, actions], dim=1))
        critic_loss = (q - q_next).pow(2).mul(0.5).sum(-1).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        next_actions = self.local_actor.forward(states)
        policy_loss = -self.local_critic(torch.cat([states.detach(), next_actions], dim=1)).mean()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        # Soft update the networks
        self.soft_update(self.target_actor, self.local_actor)
        self.soft_update(self.target_critic, self.local_critic)

    def end(self, reward):
        # Store the SARS information in the replay buffer
        self.replay.add(self.last_state, self.last_action, reward, self.last_state, True)

    def soft_update(self, target, src):
        for target_param, param in zip(target.parameters(), src.parameters()):
            target_param.detach_()
            target_param.copy_(target_param * (1.0 - self.tau) +
                               param * self.tau)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['replay']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.replay = rllib.ReplayBuffer.ReplayBuffer(
            self.action_size,
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            device=self.device
        )
