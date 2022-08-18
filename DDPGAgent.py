from turtle import forward
import torch
import torch.nn as nn
import numpy as np

import rllib.ReplayBuffer
from AbstractAgent import AbstractAgent


class LinearModel(nn.Module):
    def __init__(self, state_space, action_space, hidden=(32, 32), activation_fn=torch.relu,
                 end_activation_fn=torch.relu) -> None:
        super().__init__()

        self.layers = []

        # Create hidden layers
        last_size = state_space
        for i, val in enumerate(hidden):
            self.layers.append(nn.Linear(last_size, val))
            last_size = val

            # Add output layer
            if i == len(hidden) - 1:
                self.layers.append(nn.Linear(last_size, action_space))

        # Save values
        self.activation_fn = activation_fn
        self.end_activation_fn = end_activation_fn

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation_fn(x)

        # Last layer
        x = self.layers[-1](x)

        return x if self.end_activation_fn is None else self.end_activation_fn(x)


class DDPGAgent(AbstractAgent):
    def __init__(self, state_size, action_size, *, gamma=1.0, alpha=0.1, seed=-1, **kwargs) -> None:
        super().__init__(state_size, action_size, gamma=gamma, alpha=alpha, seed=seed, **kwargs)

        self.local_actor = LinearModel(state_size, action_size, end_activation_fn=torch.tanh)

        self.last_state = None
        self.last_action = None

        self.buffer_size = kwargs.get('buffer_size')
        self.batch_size = kwargs.get('batch_size')

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
            x = torch.from_numpy(state).float().unsqueeze(0)
            return self.local_actor.forward(x).cpu().data.numpy()

    def start(self, state):
        # Take and action and store
        self.last_state = np.asarray(state, dtype=float)
        self.last_action = self.take_action(state, True)
        return self.last_action

    def step(self, reward, state, learn=True):
        # Store the SARS information in the replay buffer
        next_state = np.asarray(state, dtype=float)
        self.replay.add(self.last_state, self.last_action, reward, next_state, False)

        # Get next action
        self.last_state = next_state
        self.last_action = self.take_action(state, True)
        return self.last_action

    def end(self, reward):
        # Store the SARS information in the replay buffer
        self.replay.add(self.last_state, self.last_action, reward, self.last_state, True)

        # Do a sample
        print('Sample')
        print(self.replay.sample())

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
