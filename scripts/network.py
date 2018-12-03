"""
Deep Q Network (DQN) helper file.
Project 1: Navigation
Udacity Deep Reinforcement Learning Nanodegree
Brian McMahon
December 2018
"""
from collections import namedtuple, deque

import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):
    """Actor (Policy) Model.
    Inspired by code from https://github.com/udacity/deep-reinforcement-learning/tree/master/dqn/
    """
    def __init__(self, ss, acts, seed, hidden=[64,64]):
        """
        ss (int): Dimension of state
        acts (int): Dimension of action
        seed (int): Random seed
        hidden list(int): Nodes in (currently two) hidden layers
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(ss, hidden[0])
        self.fc2 = nn.Linear(hidden[0],hidden[1])
        self.fc3 = nn.Linear(hidden[1],acts)

    def forward(self, state):
        """Build network that maps state to action values"""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    """
    Fixed size buffer to store experience tuples.
    Inspired by code from https://github.com/udacity/deep-reinforcement-learning/tree/master/dqn/
    """
    def __init__(self, acts, bs, seed, buf_sz):
        """
        acts (int): Dimension of action
        bs (int): Size of each training batch
        seed (int): Random seed
        buf_sz (int): maximum size of buffer
        """
        self.acts = acts
        self.memory = deque(maxlen=buf_sz)
        self.bs = bs
        self.experience = namedtuple("Experience",field_names=["state","action","reward","next_state","done"])
        self.seed = random.seed(seed)

    def add(self,state, action, reward, next_state, done):
        """Add a new experience to memory"""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, a, b):
        """Randomly sample a batch of experiences from memory"""
        expers = random.sample(self.memory, k=self.bs)

        sts = torch.from_numpy(np.vstack([e.state for e in expers if e is not None])).float().to(device)
        acts = torch.from_numpy(np.vstack([e.action for e in expers if e is not None])).long().to(device)
        rwds = torch.from_numpy(np.vstack([e.reward for e in expers if e is not None])).float().to(device)
        nxt_sts = torch.from_numpy(np.vstack([e.next_state for e in expers if e is not None])).float().to(device)
        dns = torch.from_numpy(np.vstack([e.done for e in expers if e is not None]).astype(np.uint8)).float().to(device)
        return (sts, acts, rwds, nxt_sts, dns)

    def __len__(self):
        """Return current size of internal memory"""
        return len(self.memory)

class PriorityReplayBuffer(ReplayBuffer):
    """
    Fixed-size buffer to store experience tuples
    See paper "Prioritized Experience Replay" at https://arxiv.org/abs/1511.05952
    Inspired by code from https://github.com/franckalbinet/drlnd-project1/blob/master/dqn_agent.py
    """
    def __init__(self, acts, bs, seed, buf_sz):
        """Prioritizes experience replay buffer to store experience tuples"""
        super(PriorityReplayBuffer, self).__init__(acts, bs, seed, buf_sz)
        # super().__init__(acts, bs, seed, buf_sz=buf_sz)
        self.experience = namedtuple("Experience",field_names=["state","action","reward","next_state","done","priority"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory"""
        max_priority = max([m.priority for m in self.memory]) if self.memory else 1.0
        e = self.experience(state,action,reward,next_state,done,max_priority)
        self.memory.append(e)

    def sample(self, a, b):
        """Randomly sample a batch of expers from memory"""
        priorities = np.array([sample.priority for sample in self.memory])
        probs = priorities ** a
        probs /= probs.sum()

        idxs = np.random.choice(len(self.memory),self.bs, replace=False, p=probs)
        expers = [self.memory[idx] for idx in idxs]
        total = len(self.memory)
        wts = (total*probs[idxs])**(-b)
        wts /= wts.max()
        wts = np.array(wts, dtype=np.float32)

        sts = torch.from_numpy(np.vstack([e.state for e in expers if e is not None])).float().to(device)
        acts = torch.from_numpy(np.vstack([e.action for e in expers if e is not None])).long().to(device)
        rwds = torch.from_numpy(np.vstack([e.reward for e in expers if e is not None])).float().to(device)
        nxt_sts = torch.from_numpy(np.vstack([e.next_state for e in expers if e is not None])).float().to(device)
        dns = torch.from_numpy(np.vstack([e.done for e in expers if e is not None]).astype(np.uint8)).float().to(device)
        wts = torch.from_numpy(np.vstack(wts)).float().to(device)
        idxs = torch.from_numpy(np.vstack(idxs)).long().to(device)
        return (sts, acts, rwds, nxt_sts, dns, wts, idxs)

    def update_priorities(self,idxs,priorities):
        for i, idx in enumerate(idxs):
            self.memory[idx] = self.memory[idx]._replace(priority=priorities[i])

class NoisyLinear(nn.Module):
    """
    For rainbow DQN
    See paper "Rainbow: Combining Improvements in Deep Reinforcement Learning" at https://arxiv.org/abs/1710.02298
    Inspired by code at https://github.com/higgsfield/RL-Adventure/
    """
    def __init__(self, ss, acts, use_cuda, std_init=0.4):
        super(NoisyLinear,self).__init__()
        self.use_cuda = use_cuda
        self.ss = ss
        self.acts = acts
        self.std_init = std_init

        self.wt_m = nn.Parameter(torch.FloatTensor(acts,ss))
        self.wt_s = nn.Parameter(torch.FloatTensor(acts,ss))
        self.register_buffer("wt_e",torch.FloatTensor(acts,ss))
        self.bias_m = nn.Parameter(torch.FloatTensor(acts))
        self.bias_s = nn.Parameter(torch.FloatTensor(acts))
        self.register_buffer("bias_e",torch.FloatTensor(acts))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        if self.use_cuda:
            wt_e = self.wt_e.cuda()
            bias_e = self.bias_e.cuda()
        else:
            wt_e = self.wt_e
            bias_e = self.bias_e

        if self.training:
            wt = self.wt_m + self.wt_s.mul(Variable(wt_e))
            bias = self.bias_m + self.bias_s.mul(Variable(bias_e))
        else:
            wt = self.wt_m
            bias = self.bias_m

        return F.linear(x, wt, bias)

    def reset_parameters(self):
        m_range = 1/math.sqrt(self.wt_m.size(1))

        self.wt_m.data.uniform_(-m_range,m_range)
        self.wt_s.data.fill_(self.std_init / math.sqrt(self.wt_s.size(1)))
        self.bias_m.data.uniform_(-m_range,m_range)
        self.bias_s.data.fill_(self.std_init / math.sqrt(self.bias_s.size(0)))

    def reset_noise(self):
        e_in = self._scale_noise(self.ss)
        e_out = self._scale_noise(self.acts)
        self.wt_e.copy_(e_out.ger(e_in))
        self.bias_e.copy_(self._scale_noise(self.acts))

    def _scale_noise(self,size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x
