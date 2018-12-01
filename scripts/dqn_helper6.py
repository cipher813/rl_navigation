"""
Deep Q Network helper file.
Udacity Deep Reinforcement Learning Nanodegree, December 2018.

Note that unityagents and gym are called only when specified in the run function.
"""
import re
import time
import pickle
import random
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

lr = 0.00025        # learning rate
buf_sz = int(1e5)   # replay buffer size
bs = 64             # minibatch size
a = 0.4             # alpha
g = 0.99            # gamma, discount factor
t = 1e-3            # tau, for soft update of target parameters
fq = 4              # frequency, how often to update the network
seed = 0            # seed

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
    def __init__(self, acts, bs, seed, buffer_size=buf_sz):
        """
        acts (int): Dimension of action
        bs (int): Size of each training batch
        seed (int): Random seed
        buffer_size (int): maximum size of buffer
        """
        self.acts = acts
        self.memory = deque(maxlen=buffer_size)
        self.bs = bs
        self.experience = namedtuple("Experience",field_names=["state","action","reward","next_state","done"])
        self.seed = random.seed(seed)

    def add(self,state, action, reward, next_state, done):
        """Add a new experience to memory"""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, alpha, b):
        """Randomly sample a batch of experiences from memory"""
        experiences = random.sample(self.memory, k=self.bs)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return current size of internal memory"""
        return len(self.memory)

class PriorityReplayBuffer(ReplayBuffer):
    """
    Fixed-size buffer to store experience tuples
    Inspired by code from https://github.com/franckalbinet/drlnd-project1/blob/master/dqn_agent.py
    """
    def __init__(self, acts, bs, seed, buffer_size=buf_sz):
        """Prioritizes experience replay buffer to store experience tuples"""
        super(PriorityReplayBuffer, self).__init__()
        # super().__init__(acts, bs, seed, buffer_size=buf_sz)
        self.experience = namedtuple("Experience",field_names=["state","action","reward","next_state","done","priority"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory"""
        max_priority = max([m.priority for m in self.memory]) if self.memory else 1.0
        e = self.experience(state,action,reward,next_state,done,max_priority)
        self.memory.append(e)

    def sample(self, alpha, b):
        """Randomly sample a batch of experiences from memory"""
        priorities = np.array([sample.priority for sample in self.memory])
        probs = priorities ** alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.memory),self.bs, replace=False, p=probs)
        experiences = [self.memory[idx] for idx in indices]
        total = len(self.memory)
        weights = (total*probs[indices])**(-b)
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
    """
    Base agent which interacts with an learns from environment
    Inspired by code from https://github.com/udacity/deep-reinforcement-learning/tree/master/dqn/
    """
    def __init__(self, ss, acts, seed, buffer_size=buf_sz,
                 bs=bs, g=g, lr=lr, update_freq=fq):
        """
        ss (int): Dimension of state
        acts (int): Dimension of action
        seed (int): Random seed
        """
        self.ss = ss
        self.acts = acts
        self.seed = random.seed(seed)
        self.buffer_size = int(buffer_size)
        self.bs = bs
        self.g = g
        self.lr = lr
        self.update_freq = update_fq

        # Q Network
        self.qnetwork_local = QNetwork(ss, acts, seed).to(device)
        self.qnetwork_target = QNetwork(ss, acts, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(),lr=lr)

        # Replay memory
        self.memory = ReplayBuffer(acts, bs, seed)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done, b=1.0):
        # save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # learn every fq time steps
        self.t_step = (self.t_step + 1) % fq
        if self.t_step == 0:
            # if enough samples available in memory, get random subset and learn
            if len(self.memory)> bs:
                experiences = self.memory.sample(a, b)
                self.learn(experiences, g)

    def act(self, state, e=0.):
        """Return actions for given state per current policy
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

    def learn(self, experiences, g):
        """Update value params using given batch of experience tuples
        experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done)
        g (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        # compute Q targets for current states
        Q_targets = rewards + (g * Q_targets_next * (1 - dones))

        # get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1,actions)

        # compute loss
        loss = F.mse_loss(Q_expected, Q_targets)

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
    def __init__(self, ss, acts, seed, buffer_size=buf_sz,
                 bs=bs,g=g,lr=lr,update_freq=fq):
        super().__init__(ss, acts, seed, buffer_size=buf_sz,
                     bs=bs,g=g,lr=lr,update_freq=fq)

    def learn(self, experiences, g):
        states, actions, rewards, next_states, dones = experiences
        local_max_actions = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
        Q_targets_next = torch.gather(self.qnetwork_target(next_states).detach(),1,local_max_actions)
        Q_targets = rewards + (g * Q_targets_next * (1-dones))
        Q_expected = self.qnetwork_local(states).gather(1,actions)

        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.qnetwork_local, self.qnetwork_target,t)
# class Double(Vanilla):
#     """
#     Interacts with and learns from the environment.
#     Inspired by code from https://github.com/franckalbinet/drlnd-project1/blob/master/dqn_agent.py
#     """
#     def __init__(self, ss, acts, seed,buffer_size=buf_sz,
#                  bs=bs, g=g, lr=lr,update_every=UPDATE_EVERY):
#         super().__init__(ss, acts, seed,buffer_size=buf_sz,
#                  bs=bs, g=g, lr=lr,update_every=UPDATE_EVERY)
#
#     def learn(self, experiences, g):
#         """Update value parameters using given batch of experience tuples.
#
#         Params
#         ======
#             experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
#             g (float): discount factor
#         """
#         states, actions, rewards, next_states, dones = experiences
#
#         local_max_actions = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
#         # Get max predicted Q values (for next states) from target model
#         Q_targets_next = torch.gather(self.qnetwork_target(next_states).detach(),1,local_max_actions)
# #         Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
#         # Compute Q targets for current states
#         Q_targets = rewards + (g * Q_targets_next * (1 - dones))
#
#         # Get expected Q values from local model
#         Q_expected = self.qnetwork_local(states).gather(1, actions)
#
#         # Compute loss
#         loss = F.mse_loss(Q_expected, Q_targets)
#         # Minimize the loss
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()
#
#         # ------------------- update target network ------------------- #
#         self.soft_update(self.qnetwork_local, self.qnetwork_target, t)
#
class PriorityReplay(Double):
    def __init__(self, ss, acts, seed, buffer_size=buf_sz,
                 bs=bs, g=g, lr=lr, update_freq=fq):
        super().__init__(ss, acts, seed, buffer_size=buf_sz,
                     bs=bs, g=g, lr=lr, update_freq=fq)
        self.memory = PriorityReplayBuffer(acts, bs, seed)

    def learn(self, experiences, g):
        states, actions, rewards, next_states, dones, weights, indices = experiences
        local_max_actions = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
        Q_targets_next = torch.gather(self.qnetwork_target(next_states).detach(),1,local_max_actions)
        Q_targets = rewards + (g * Q_targets_next * (1-dones))
        Q_expected = self.qnetwork_local(states).gather(1,actions)
        loss = (Q_expected - Q_targets).pow(2)*weights
        prios = loss + 1e-5
        loss = loss.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.memory.update_priorities(indices.sqeeze().to("cpu").data.numpy(),prios.squeeze().to("cpu").data.numpy())
        self.soft_update(self.qnetwork_local,self.qnetwork_target,t)
# class PriorityReplay(Double):
#     """
#     Interacts with and learns from the environment.
#     Inspired by code from https://github.com/franckalbinet/drlnd-project1/blob/master/dqn_agent.py
#     """
#     def __init__(self, ss, acts, seed,buffer_size=buf_sz,
#                  bs=bs, g=g, lr=lr,update_every=UPDATE_EVERY):
#         super().__init__(ss, acts, seed,buffer_size=buf_sz,
#                  bs=bs, g=g, lr=lr,update_every=UPDATE_EVERY)
#
#         self.memory = PriorityReplayBuffer(acts, bs, seed)
#
#     def learn(self, experiences, g):
#         """Update value parameters using given batch of experience tuples.
#
#         Params
#         ======
#             experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
#             g (float): discount factor
#         """
#         states, actions, rewards, next_states, dones, weights, indices = experiences
#
#         local_max_actions = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
#         # Get max predicted Q values (for next states) from target model
#         Q_targets_next = torch.gather(self.qnetwork_target(next_states).detach(),1,local_max_actions)
# #         Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
#         # Compute Q targets for current states
#         Q_targets = rewards + (g * Q_targets_next * (1 - dones))
#
#         # Get expected Q values from local model
#         Q_expected = self.qnetwork_local(states).gather(1, actions)
#
#         # Compute loss
#         loss = (Q_expected - Q_targets).pow(2)*weights
#         prios = loss + 1e-5
#         loss = loss.mean()
# #         loss = F.mse_loss(Q_expected, Q_targets)
#         # Minimize the loss
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()
#
#         # update priorities based on td error
#         self.memory.update_priorities(indices.squeeze().to("cpu").data.numpy(),prios.squeeze().to("cpu").data.numpy())
#
#         # ------------------- update target network ------------------- #
#         self.soft_update(self.qnetwork_local, self.qnetwork_target, t)

class Dueling(PriorityReplay):
    """Inspired by code at https://github.com/dxyang/DQN_pytorch/blob/master/model.py."""
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
    """Inspired by code at https://github.com/ikostrikov/pytorch-a3c/blob/master/model.py."""
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

        self.weights_init()
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

    def normalized_columns_initializer(self, weights, std=1.0):
        out = torch.randn(weights.size())
        out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
        return out

    def weights_init(self):
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

def train_gym(CHART_PATH, CHECKPOINT_PATH, agent_dict, module, timestamp, seed, score_target,
              n_episodes,max_t,e_start,e_end,e_decay): #agent_dict,
    """Trains OpenAI Gym environments."""
    import gym
    start = time.time()
    label = "gym"
    env = gym.make(module)
    ss = env.observation_space.shape[0]
    acts = env.action_space.n
    result_dict = {}
    for k,v in agent_dict.items():
        agent_name = k
        print(f"Agent: {k}")
        agent = v(ss, acts, seed)
        scores = []                        # list containing scores from each episode
        scores_window = deque(maxlen=100)  # last 100 scores
        e = e_start
        for i_episode in range(1, n_episodes+1):
            state = env.reset()
            score = 0
            for t in range(max_t):
                action = agent.act(state,e)
                next_state, reward, done, _ = env.step(action)
                agent.step(state, action, reward, next_state, done)
                state = next_state
                score += reward                                # update the score
                if done:                                       # exit loop if episode finished
                    break
            scores_window.append(score)       # save most recent score
            scores.append(score)              # save most recent score
            e = max(e_end, e_decay*e) # decrease epsilon
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
            if i_episode % 100 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            if np.mean(scores_window)>=score_target:
                end = time.time()
                print(f'Environment solved in {i_episode:d} episodes!\tAverage Score: {np.mean(scores_window):.2f}\tRuntime: {(end-start)/60:.2f}')
                checkpath = CHECKPOINT_PATH + f'checkpoint-{timestamp}-{label}-{module}-{agent_name}.pth'
                torch.save(agent.qnetwork_local.state_dict(), checkpath)
                print(f"Checkpoint saved at {checkpath}")
                break
        result_dict[agent_name] = {
                        "scores":scores,
                        "clocktime":round((end-start)/60,2)
                        }
        pklpath = CHART_PATH + f"ResultDict-{timestamp}-{label}-{module}-{agent_name}.pkl"
        # pklpath = CHART_PATH + f"ResultDict-{timestamp}-{label}-{module}.pkl"
        with open(pklpath, 'wb') as handle:
            pickle.dump(result_dict, handle)
        print(f"Scores pickled at {pklpath}")
    return result_dict

def train_unity(PATH, CHART_PATH, CHECKPOINT_PATH, agent_dict, module, timestamp, seed, score_target,
                n_episodes,max_t,e_start,e_end,e_decay):
    """Trains Unity 3D Editor environments."""
    from unityagents import UnityEnvironment
    APP_PATH = PATH + f"data/{module}"
    label = "unity"
    start = time.time()
    env = UnityEnvironment(file_name=APP_PATH)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset()[brain_name]
    ss = len(env_info.vector_observations[0])
    acts = brain.vector_action_space_size
    result_dict = {}
    for k,v in agent_dict.items():
        agent_name = k
        print(f"Agent: {k}")
        agent = v(ss,acts,seed)
        scores = []                        # list containing scores from each episode
        scores_window = deque(maxlen=100)  # last 100 scores
        e = e_start
        for i_episode in range(1, n_episodes+1):
            state = env.reset()
            score = 0
            for t in range(max_t):
                state = env_info.vector_observations[0]  # get the current state
                action = agent.act(state,e)
                env_info = env.step(action)[brain_name]        # send the action to the environment
                next_state = env_info.vector_observations[0]   # get the next state
                reward = env_info.rewards[0]                   # get the reward
                done = env_info.local_done[0]
                agent.step(state, action, reward, next_state, done)
                score += reward                                # update the score
                state = next_state                             # roll over the state to next time step
                if done:                                       # exit loop if episode finished
                    break
            scores_window.append(score)       # save most recent score
            scores.append(score)              # save most recent score
            e = max(e_end, e_decay*e) # decrease epsilon
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
            if i_episode % 100 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            if np.mean(scores_window)>=score_target:
                end = time.time()
                print(f'Environment solved in {i_episode:d} episodes!\tAverage Score: {np.mean(scores_window):.2f}\tRuntime: {(end-start)/60:.2f}')
                checkpath = CHECKPOINT_PATH + f'checkpoint-{timestamp}-{label}-{agent_name}.pth'
                torch.save(agent.qnetwork_local.state_dict(), checkpath)
                print(f"Checkpoint saved at {checkpath}")
                break
        result_dict[agent_name] = {
                        "scores":scores,
                        "clocktime":round((end-start)/60,2)
                        }
        pklpath = CHART_PATH + f"ResultDict-{timestamp}-{label}-{agent_name}.pkl"
        with open(pklpath, 'wb') as handle:
            pickle.dump(result_dict, handle)
        print(f"Scores pickled at {pklpath}")
    return result_dict

def chart_results(CHART_PATH, pklfile):
    """Charts performance results by agent."""
    pklpath = CHART_PATH + pklfile
    timestamp = pklpath.split(".")[-2].split("-")[-1]

    with open(pklpath, 'rb') as handle:
        results = pickle.load(handle)
    for module in results.keys():
        mod_data = results[module]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for key in mod_data.keys():
            scores = mod_data[key]['scores']
            avg_scores = []
            for i in range(1,len(scores)+1):
                start = np.max(i-roll_length,0)
                end = i
                nm = np.sum(scores[start:end])
                dn = len(scores[start:end])
                avg_scores.append(nm/dn)
            plt.plot(np.arange(len(scores)), avg_scores,label=key)
            plt.ylabel('Score')
            plt.xlabel('Episode #')
            plt.title(f"{module}")
            plt.legend()
        chartpath = CHART_PATH + f"NavigationTrainChart-{timestamp}-{module}-{key}.png"
        plt.savefig(chartpath)
        print(f"Chart saved at {chartpath}")
    plt.show()
    display(pd.DataFrame(results))
    return results

def train_envs(PATH, CHART_PATH, CHECKPOINT_PATH, agent_dict, timestamp, env_dict, seed=0,
               n_episodes=3000,max_t=1000,e_start=0.4,e_end=0.01,e_decay=0.995):
    """Main trian function for all envs in env_dict."""
    rd = {}
    for k,v in env_dict.items():
        start = time.time()
        module = k
        platform = v[0]
        print(f"Begin training {module}-{platform}.")
        score_target = v[1]
        print(f"Module: {module}-{platform}")
        if platform == "gym":
            results = train_gym(CHART_PATH, CHECKPOINT_PATH, agent_dict, module, timestamp, seed, score_target,
                            n_episodes,max_t,e_start,e_end,e_decay)
        elif platform == "unity":
            results = train_unity(PATH, CHART_PATH, CHECKPOINT_PATH, agent_dict, module, timestamp, seed, score_target,
                          n_episodes,max_t,e_start,e_end,e_decay)
        else:
            print("Check your model and platform inputs.")
        rd[module] = results
        end = time.time()
        print(f"Finished training {module}-{platform} in {(end-start)/60:.2f} minutes.")
    pklpath = CHART_PATH + f"ResultDict-All-{timestamp}.pkl"
    with open(pklpath, 'wb') as handle:
        pickle.dump(rd, handle)
        print(f"Scores pickled at {pklpath}")
    return rd
