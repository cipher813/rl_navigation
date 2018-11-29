"""
WORK IN PROGRESS
Deep Q Network helper file.
Udacity Deep Reinforcement Learning Nanodegree, December 2018.
"""
import re
import random
import pickle
import datetime
import numpy as np
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)
BATCH_SIZE = 64
LR = 0.00025
ALPHA = 0.4
GAMMA = 0.99
TAU = 1e-3
UPDATE_FREQ = 4
SEED = 0

timestamp = re.sub(r"\D","",str(datetime.datetime.now()))[:12]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, hidden=[64,64]):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, hidden[0])
        self.fc2 = nn.Linear(hidden[0],hidden[1])
        self.fc3 = nn.Linear(hidden[1],action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, action_size, batch_size, seed, buffer_size=BUFFER_SIZE):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience",field_names=["state","action","reward","next_state","done"])
        self.seed = random.seed(seed)

    def add(self,state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, alpha, beta):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)

class PriorityReplayBuffer(ReplayBuffer):
    def __init__(self, action_size, batch_size, seed, buffer_size=BUFFER_SIZE):
        super().__init__(action_size, batch_size, seed, buffer_size=BUFFER_SIZE)
        self.experience = namedtuple("Experience",field_names=["state","action","reward","next_state","done","priority"])

    def add(self, state, action, reward, next_state, done):
        max_priority = max([m.priority for m in self.memory]) if self.memory else 1.0
        e = self.experience(state,action,reward,next_state,done,max_priority)
        self.memory.append(e)

    def sample(self, alpha, beta):
        # super().sample(alpha, beta)
        # returns (states, actions, rewards, next_states, dones)
        priorities = np.array([sample.priority for sample in self.memory])
        probs = priorities ** alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.memory),self.batch_size, replace=False, p=probs)
        experiences = [self.memory[idx] for idx in indices]
        total = len(self.memory)
        weights = (total*probs[indices])**(-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        weights = torch.from_numpy(np.vstack(weights)).float().to(device)
        indices = torch.from_numpy(np.vstack(indices)).long().to(device)
        return (states, actions, rewards, next_states, dones, weights, indices)

    def update_priorities(self,indices,priorities):
        for i, idx in enumerate(indices):
            self.memory[idx] = self.memory[idx]._replace(priority=priorities[i])

class Vanilla:
    def __init__(self, state_size, action_size, seed, buffer_size=BUFFER_SIZE,
                 batch_size=BATCH_SIZE, gamma=GAMMA, lr=LR, update_freq=UPDATE_FREQ):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.buffer_size = int(buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.update_freq = update_freq

        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(),lr=LR)

        self.memory = ReplayBuffer(action_size, BATCH_SIZE, seed)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done, beta=1.0):
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % UPDATE_FREQ
        if self.t_step == 0:
            if len(self.memory)> BATCH_SIZE:
                experiences = self.memory.sample(ALPHA, beta)
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        if random.random()>eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.qnetwork_local(states).gather(1,actions)

        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.qnetwork_local,self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """θ_target = τ*θ_local + (1 - τ)*θ_target"""
        for target_param, local_param in zip(target_model.parameters(),local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class Double(Vanilla):
    def __init__(self, state_size, action_size, seed, buffer_size=BUFFER_SIZE,
                 batch_size=BATCH_SIZE,gamma=GAMMA,lr=LR,update_freq=UPDATE_FREQ):
        super().__init__(state_size, action_size, seed, buffer_size=BUFFER_SIZE,
                     batch_size=BATCH_SIZE,gamma=GAMMA,lr=LR,update_freq=UPDATE_FREQ)

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        local_max_actions = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
        Q_targets_next = torch.gather(self.qnetwork_target(next_states).detach(),1,local_max_actions)
        Q_targets = rewards + (gamma * Q_targets_next * (1-dones))
        Q_expected = self.qnetwork_local(states).gather(1,actions)

        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.qnetwork_local, self.qnetwork_target,TAU)

class PriorityReplay(Double):
    def __init__(self, state_size, action_size, seed, buffer_size=BUFFER_SIZE,
                 batch_size=BATCH_SIZE, gamma=GAMMA, lr=LR, update_freq=UPDATE_FREQ):
        super().__init__(state_size, action_size, seed, buffer_size=BUFFER_SIZE,
                     batch_size=BATCH_SIZE, gamma=GAMMA, lr=LR, update_freq=UPDATE_FREQ)
        self.memory = PriorityReplayBuffer(action_size, batch_size, seed)

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones, weights, indices = experiences
        local_max_actions = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
        Q_targets_next = torch.gather(self.qnetwork_target(next_states).detach(),1,local_max_actions)
        Q_targets = rewards + (gamma * Q_targets_next * (1-dones))
        Q_expected = self.qnetwork_local(states).gather(1,actions)
        loss = (Q_expected - Q_targets).pow(2)*weights
        prios = loss + 1e-5
        loss = loss.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.memory.update_priorities(indices.sqeeze().to("cpu").data.numpy(),prios.squeeze().to("cpu").data.numpy())
        self.soft_update(self.qnetwork_local,self.qnetwork_target,TAU)
