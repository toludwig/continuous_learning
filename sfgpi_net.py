# This code is largely copied from
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

import numpy as np
import random
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# online learning of a Successor Features (SF)
# implemented as a neural network: State -> SF for each Action

class SFNet(nn.Module):

    def __init__(self, input_size, output_size):
        super(SFNet, self).__init__()

        self.linearLayers = nn.Sequential(
            nn.Linear(input_size, 50),
            nn.ReLU(nn.Linear(50, 50)),
            nn.Linear(50, 25),
            nn.Linear(25, output_size),
        )

    def forward(self, x):
        return self.linearLayers(x)

#############################################################
# SFGPI predicts Successor Features,
# i.e. only learns which features follow next given a policy
#############################################################

class SFGPI():

    def __init__(self, input_size, output_size, n_actions, n_features,
                 LEARNING_RATE, GAMMA, BUFFER_CAPACITY, BATCH_SIZE, TARGET_UPDATE):

        # target net makes predictions, policy net is updated with TD error
        # and after a while parameters are copied from policy to target net
        self.policy_net = SFNet(input_size, output_size)
        self.target_net = SFNet(input_size, output_size)

        self.n_actions  = n_actions
        self.n_features = n_features

        self.LEARNING_RATE = LEARNING_RATE
        self.GAMMA = GAMMA
        self.BATCH_SIZE = BATCH_SIZE
        self.TARGET_UPDATE = TARGET_UPDATE

        self.replay_buffer = ReplayMemory(BUFFER_CAPACITY)
        self.optimizer = optim.RMSprop(self.policy_net.parameters())

    def predict_psi(self, state):
        with torch.no_grad():
            psi = self.policy_net(state)
        psi = psi.detach().numpy().astype(np.float64)
        return np.reshape(psi, (self.n_actions, self.n_features))

    def train(self, step):
        if len(self.replay_buffer) < self.BATCH_SIZE:
            return

        transitions = self.replay_buffer.sample(self.BATCH_SIZE)
        # this converts an array of Transitions to Transition of arrays
        batch = Transition(*zip(*transitions))
        s_batch       = torch.vstack(batch.state)
        a_batch       = torch.vstack(batch.action)
        phi_batch     = torch.vstack(batch.features)
        s_next_batch  = torch.vstack(batch.next_state)

        # compute TD error (psi vs phi, not Q vs r!)
        psi   = self.policy_net.forward(s_batch) # BATCH_SIZE x n_features
        psi   = torch.reshape(psi, (self.BATCH_SIZE, self.n_actions, self.n_features))
        index = torch.stack([a_batch for _ in range(self.n_features)], 2)
        psi_a = psi.gather(1, index).squeeze() # only SF for chosen a
        # get prediction for next state from target net for better stability
        psi_next = self.target_net(s_next_batch)
        psi_next = torch.reshape(psi_next, (self.BATCH_SIZE, self.n_actions, self.n_features))
        psi_next = psi_next.max(1)[0].squeeze() # max over actions
        delta    = phi_batch + self.GAMMA * psi_next - psi_a

        # backprop
        criterion = nn.SmoothL1Loss() # NB: matrix norm in this case
        loss = criterion(self.LEARNING_RATE * delta, torch.zeros_like(delta))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # copy policy net's parameters to target net
        if step % self.TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())



#############################################################
# REPLAY
#############################################################

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'features'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """Sample with replacement (works even if batch_size > len(buffer))"""
        return random.choices(self.memory, k=batch_size)

    def __len__(self):
        return len(self.memory)
