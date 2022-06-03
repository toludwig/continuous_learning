# This code is largely copied from
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

import numpy as np
import random
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# online learning of a task-conditioned value function
# implemented as a neural network: Task x State -> Policy

class TaskQNet(nn.Module):

    def __init__(self, input_size, output_size):
        super(TaskQNet, self).__init__()

        self.linearLayers = nn.Sequential(
            nn.Linear(input_size, 50),
            nn.ReLU(nn.Linear(50, 50)),
            nn.Linear(50, 25),
            nn.Linear(25, output_size),
        )

    def forward(self, x):
        return self.linearLayers(x)


class UVFA():

    def __init__(self, input_size, output_size, LEARNING_RATE,
                 GAMMA, BUFFER_CAPACITY, BATCH_SIZE, TARGET_UPDATE):

        # target net makes predictions, policy net is updated with TD error
        # and after a while parameters are copied from policy to target net
        self.policy_net = TaskQNet(input_size, output_size)
        self.target_net = TaskQNet(input_size, output_size)

        self.LEARNING_RATE = LEARNING_RATE
        self.GAMMA = GAMMA
        self.BATCH_SIZE = BATCH_SIZE
        self.TARGET_UPDATE = TARGET_UPDATE

        self.replay_buffer = ReplayMemory(BUFFER_CAPACITY)
        self.optimizer = optim.RMSprop(self.policy_net.parameters())

    def predict_Q(self, sw):
        with torch.no_grad():
            return self.policy_net(sw)

    def train(self, step):
        if len(self.replay_buffer) < self.BATCH_SIZE:
            return

        transitions = self.replay_buffer.sample(self.BATCH_SIZE)
        # this converts an array of Transitions to Transition of arrays
        batch = Transition(*zip(*transitions))
        sw_batch      = torch.vstack(batch.state)
        a_batch       = torch.vstack(batch.action)
        r_batch       = torch.vstack(batch.reward)
        sw_next_batch = torch.vstack(batch.next_state)

        # compute TD error
        # get prediction for next state from target net for better stability
        Q_a    = self.policy_net.forward(sw_batch).gather(1, a_batch)
        V_next = self.target_net(sw_next_batch).max(1)[0].detach()
        delta  = r_batch + self.GAMMA * V_next - Q_a

        # backprop
        criterion = nn.SmoothL1Loss()
        loss = criterion(self.LEARNING_RATE * delta, torch.zeros_like(delta))
        #loss = criterion(Q_a, r_batch + self.GAMMA * V_next)
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
                        ('state', 'action', 'next_state', 'reward'))

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
