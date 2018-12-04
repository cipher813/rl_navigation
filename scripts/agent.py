from network import QNetwork, ReplayBuffer, PriorityReplayBuffer, NoisyLinear

import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

# hyperparameters
lr = 0.00025        # learning rate
buf_sz = int(1e5)   # replay buffer size
bs = 64             # minibatch size
a = 0.4             # alpha
g = 0.99            # gamma, discount factor
t = 1e-3            # tau, for soft update of target parameters
fq = 4              # frequency, how often to update the network
seed = 0            # seed

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Vanilla:
    """
    Base agent which interacts with an learns from environment
    Inspired by code from https://github.com/udacity/deep-reinforcement-learning/tree/master/dqn/
    """
    def __init__(self, ss, acts, seed, buf_sz=buf_sz,
                 bs=bs, g=g, lr=lr, update_fq=fq):
        """
        ss (int): Dimension of state
        acts (int): Dimension of action
        seed (int): Random seed
        """
        self.ss = ss
        self.acts = acts
        self.seed = random.seed(seed)
        self.buf_sz = buf_sz
        self.bs = bs
        self.g = g
        self.lr = lr
        self.update_fq = update_fq

        # Q Network
        self.qnetwork_local = QNetwork(ss, acts, seed).to(device)
        self.qnetwork_target = QNetwork(ss, acts, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(),lr=lr)

        # Replay memory
        self.memory = ReplayBuffer(acts, bs, seed, buf_sz)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done, b=1.0):
        # save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # learn every fq time steps
        self.t_step = (self.t_step + 1) % fq
        if self.t_step == 0:
            # if enough samples available in memory, get random subset and learn
            if len(self.memory)> bs:
                expers = self.memory.sample(a, b)
                self.learn(expers, g)

    def act(self, state, e=0.):
        """Return acts for given state per current policy
        state (arr): current state
        e (float): epsilon for e-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # e-greedy action selection
        if random.random()>e:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.acts))

    def learn(self, expers, g):
        """Update value params using given batch of experience tuples
        expers (Tuple[torch.Tensor]): tuple of (s, a, r, s', done)
        g (float): discount factor
        """
        sts, acts, rwds, nxt_sts, dns = expers

        # get max predicted Q values (for next sts) from target model
        Q_tgts_nxt = self.qnetwork_target(nxt_sts).detach().max(1)[0].unsqueeze(1)

        # compute Q targets for current sts
        Q_tgts = rwds + (g * Q_tgts_nxt * (1 - dns))

        # get expected Q values from local model
        Q_expd = self.qnetwork_local(sts).gather(1,acts)

        # compute loss
        loss = F.mse_loss(Q_expd, Q_tgts)

        # minimize loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update target network
        self.soft_update(self.qnetwork_local,self.qnetwork_target, t)

    def soft_update(self, local_model, target_model, t):
        """
        Soft update model params
        θ_target = τ*θ_local + (1 - τ)*θ_target
        local_model (PyTorch model): weights to copy from
        target_model (PyTorch model): weights to copy to
        t (float): interpolation param
        """
        for target_param, local_param in zip(target_model.parameters(),local_model.parameters()):
            target_param.data.copy_(t*local_param.data + (1.0-t)*target_param.data)

class Double(Vanilla):
    """
    Addresses a DQN's tendency to overestimate action values
    See paper "Deep Reinforcement Learning with Double Q-Learning" at https://arxiv.org/abs/1509.06461
    Inspired by code from https://github.com/franckalbinet/drlnd-project1/blob/master/dqn_agent.py
    """
    def __init__(self, ss, acts, seed, buf_sz=buf_sz,
                 bs=bs,g=g,lr=lr,update_fq=fq):
        super(Double, self).__init__(ss, acts, seed)

    def learn(self, expers, g):
        """
        Update value params using given batch of experience tuples
        expers (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
        g (float): discount factor
        """
        sts, acts, rwds, nxt_sts, dns = expers
        local_max_acts = self.qnetwork_local(nxt_sts).detach().max(1)[1].unsqueeze(1)

        # get max predicted Q values (for next states) from target model
        Q_tgts_nxt = torch.gather(self.qnetwork_target(nxt_sts).detach(),1,local_max_acts)

        # compute Q targets for current states
        Q_tgts = rwds + (g * Q_tgts_nxt * (1-dns))

        # get expected Q values from local model
        Q_expd = self.qnetwork_local(sts).gather(1,acts)

        # compute loss
        loss = F.mse_loss(Q_expd, Q_tgts)

        # minimize loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # udpate target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target,t)
#
class PriorityReplay(Double):
    """
    An agent can learn more effectively from some transitions than from others; the more important
    transitions should be sampled with higher probability
    See paper "Prioritized Experience Replay" at https://arxiv.org/abs/1511.05952
    Inspired by code from https://github.com/franckalbinet/drlnd-project1/blob/master/dqn_agent.py
    """
    def __init__(self, ss, acts, seed, buf_sz=buf_sz,
                 bs=bs, g=g, lr=lr, update_fq=fq):
        super(PriorityReplay,self).__init__(ss, acts, seed)
        self.memory = PriorityReplayBuffer(acts, bs, seed, buf_sz)

    def learn(self, expers, g):
        """
        Update value params using given batch of experience tuples
        expers (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
        g (float): gamma discount factor
        """
        sts, acts, rwds, nxt_sts, dns, wts, idxs = expers
        local_max_acts = self.qnetwork_local(nxt_sts).detach().max(1)[1].unsqueeze(1)

        # get max predicted Q values (for next states) from target model
        Q_tgts_nxt = torch.gather(self.qnetwork_target(nxt_sts).detach(),1,local_max_acts)

        # compute Q targets for current states
        Q_tgts = rwds + (g * Q_tgts_nxt * (1-dns))

        # Get expected Q values from local model
        Q_expd = self.qnetwork_local(sts).gather(1,acts)

        # compute loss
        loss = (Q_expd - Q_tgts).pow(2)*wts
        prios = loss + 1e-5
        loss = loss.mean()

        # minimize loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update priorities based on td error
        self.memory.update_priorities(idxs.squeeze().to("cpu").data.numpy(),prios.squeeze().to("cpu").data.numpy())

        # update target network
        self.soft_update(self.qnetwork_local,self.qnetwork_target,t)

class Dueling(PriorityReplay):
    """
    Model-free reinforcement learning; duel network represents two separate estimators, one for
    state value function, one for state-dependent action advantage function, to generalize learning
    across actions without imposing change to the underlying RL algorithm.
    See paper "Dueling Network Architectures for Deep Reinforcement Learning" at https://arxiv.org/abs/1511.06581
    Inspired by code at https://github.com/dxyang/DQN_pytorch/blob/master/model.py.
    """
    def __init__(self, ss, acts, seed):
        super(Dueling, self).__init__(ss, acts, seed)
        self.ss = ss
        self.acts = acts
        self.seed = random.seed(seed)

        self.conv1 = nn.Conv2d(in_channels=ss, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4,stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        self.fc1_adv = nn.Linear(in_features=7*7*64, out_features=512)
        self.fc1_val = nn.Linear(in_features=7*7*64, out_features=512)

        self.fc2_adv = nn.Linear(in_features=512, out_features=acts)
        self.fc2_val = nn.Linear(in_features=512, out_features=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        bs = x.size(0)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0),-1)

        adv = self.relu(self.fc1_adv(x))
        val = self.relu(self.fc1_val(x))

        adv = self.fc2_adv(adv)
        val = self.fc2_val(val).expand(x.size(0),self.acts)

        x = val+adv-adv.mean(1).unsqueeze(1).expand(x.size(0),self.acts)
        return x

class A3C(Dueling):
    """
    Uses asynchronous gradient descent for optimization;
    parallel actor-learners have a stabilizing effect on training
    See paper "Asynchronous Methods for Deep Reinforcement Learning" at https://arxiv.org/abs/1602.01783
    Inspired by code at https://github.com/ikostrikov/pytorch-a3c/blob/master/model.py
    """
    def __init__(self, ss, acts, seed):
        super(A3C, self).__init__(ss, acts, seed)
        self.conv1 = nn.Conv2d(ss, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        self.lstm = nn.LSTMCell(32 * 3 * 3, 256)

        self.acts = acts
        self.critic_linear = nn.Linear(256, 1)
        self.actor_linear = nn.Linear(256, acts)

        self.wts_init()
        self.actor_linear.weight.data = self.normalized_columns_initializer(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = self.normalized_columns_initializer(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

#        self.train()

    def forward(self, inputs):
        inputs, (hx, cx) = inputs
        x = F.elu(self.conv1(inputs))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))

        x = x.view(-1, 32 * 3 * 3)
        hx, cx = self.lstm(x, (hx, cx))
        x = hx

        return self.critic_linear(x), self.actor_linear(x), (hx, cx)

    def normalized_columns_initializer(self, wts, std=1.0):
        out = torch.randn(wts.size())
        out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
        return out

    def wts_init(self):
        m = self
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            weight_shape = list(m.weight.data.size())
            fan_in = np.prod(weight_shape[1:4])
            fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
            w_bound = np.sqrt(6. / (fan_in + fan_out))
            m.weight.data.uniform_(-w_bound, w_bound)
            m.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            weight_shape = list(m.weight.data.size())
            fan_in = weight_shape[1]
            fan_out = weight_shape[0]
            w_bound = np.sqrt(6. / (fan_in + fan_out))
            m.weight.data.uniform_(-w_bound, w_bound)
            m.bias.data.fill_(0)

class Rainbow(Vanilla):
    """
    Implementation of rainbow DQN
    See paper "Rainbow: Combining Improvements in Deep Reinforcement Learning" at https://arxiv.org/abs/1710.02298
    Inspired by code at https://github.com/higgsfield/RL-Adventure/
    """
    def __init__(self, ss, acts, seed, num_atoms, Vmin, Vmax):
        super(Rainbow,self).__init__(ss, acts, seed)
        self.ss = ss                # state space
        self.acts = acts            # action space
        self.num_atoms = num_atoms
        self.Vmin = Vmin
        self.Vmax = Vmax

        self.linear1 = nn.Linear(ss, 32)
        self.linear2 = nn.Linear(32,64)
        self.noisy_value1 = NoisyLinear(64,64,use_cuda=USE_CUDA)
        self.noisy_value2 = NoisyLinear(64, self.num_atoms, use_cuda=USE_CUDA)
        self.noisy_advantage1 = NoisyLinear(64, 64, use_cuda=USE_CUDA)
        self.noisy_advantage2 = NoisyLinear(64, self.num_atoms * self.acts, use_cuda=USE_CUDA)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))

        value = F.relu(self.noisy_value1(x))
        value = self.noisy_value2(value)

        advantage = F.relu(self.noisy_advantage1(x))
        advantage = self.noisy_advantage2(advantage)

        value = value.view(batch_size, 1, self.num_atoms)
        advantage = advantage.view(batch_size, self.acts, self.num_atoms)

        x = value + advantage - advantage.mean(1,keepdim=True)
        x = F.softmax(x.view(-1,self.num_atoms)).view(-1, self.acts, self.num_atoms)

        return x

    def reset_noise(self):
        self.noisy_value1.reset_noise()
        self.noisy_value2.reset_noise()
        self.noisy_advantage1.reset_noise()
        self.noisy_advantage2.reset_noise()

    def act(self, state, e=0.):
        state = Variable(torch.FloatTensor(state).unsqueeze(0),volatile=True)
        dist = self.forward(state).data.cpu()
        dist = dist * torch.linspace(self.Vmin, self.Vmax, self.num_atoms)
        action = dist.sum(2).max(1)[1].numpy()[0]
        return action
