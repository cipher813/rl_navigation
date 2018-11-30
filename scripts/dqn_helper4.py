"""
Deep Q Network helper file.
Udacity Deep Reinforcement Learning Nanodegree, December 2018.
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
#from unityagents import UnityEnvironment

#import gym

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
ALPHA = 0.4
TAU = 1e-3              # for soft update of target parameters
LR = 0.00025               # learning rate
UPDATE_EVERY = 4        # how often to update the network
SEED = 0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, batch_size, seed, buffer_size=BUFFER_SIZE):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, alpha, beta):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class PriorityReplayBuffer(ReplayBuffer):
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, batch_size, seed, buffer_size=BUFFER_SIZE):
        super().__init__(action_size, batch_size, seed, buffer_size=BUFFER_SIZE)
        """
        Prioritizes Experience Replay buffer to store experience tuples.
        """
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done","priority"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        max_priority = max([m.priority for m in self.memory]) if self.memory else 1.0
        e = self.experience(state, action, reward, next_state, done, max_priority)
        self.memory.append(e)

    def sample(self, alpha, beta):
        """Randomly sample a batch of experiences from memory."""
        priorities = np.array([sample.priority for sample in self.memory])
        probs = priorities ** alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.memory), self.batch_size, replace=False, p=probs)

#         experiences = random.sample(self.memory, k=self.batch_size)
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

    def update_priorities(self, indices, priorities):
        for i, idx in enumerate(indices):
            self.memory[idx] = self.memory[idx]._replace(priority=priorities[i])

class Vanilla:
    """
    Interacts with and learns from the environment.
    Inspired by code from https://github.com/franckalbinet/drlnd-project1/blob/master/dqn_agent.py
    """

    def __init__(self, state_size, action_size, seed,buffer_size=BUFFER_SIZE,
                 batch_size=BATCH_SIZE, gamma=GAMMA, lr=LR,update_every=UPDATE_EVERY):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.buffer_size = int(buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.update_every = update_every

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done, beta=1.0):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample(ALPHA, beta)
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class Double(Vanilla):
    """
    Interacts with and learns from the environment.
    Inspired by code from https://github.com/franckalbinet/drlnd-project1/blob/master/dqn_agent.py
    """

    def __init__(self, state_size, action_size, seed,buffer_size=BUFFER_SIZE,
                 batch_size=BATCH_SIZE, gamma=GAMMA, lr=LR,update_every=UPDATE_EVERY):
        super().__init__(state_size, action_size, seed,buffer_size=BUFFER_SIZE,
                 batch_size=BATCH_SIZE, gamma=GAMMA, lr=LR,update_every=UPDATE_EVERY)

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        local_max_actions = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
        # Get max predicted Q values (for next states) from target model
        Q_targets_next = torch.gather(self.qnetwork_target(next_states).detach(),1,local_max_actions)
#         Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

class PriorityReplay(Double):
    """
    Interacts with and learns from the environment.
    Inspired by code from https://github.com/franckalbinet/drlnd-project1/blob/master/dqn_agent.py
    """

    def __init__(self, state_size, action_size, seed,buffer_size=BUFFER_SIZE,
                 batch_size=BATCH_SIZE, gamma=GAMMA, lr=LR,update_every=UPDATE_EVERY):
        super().__init__(state_size, action_size, seed,buffer_size=BUFFER_SIZE,
                 batch_size=BATCH_SIZE, gamma=GAMMA, lr=LR,update_every=UPDATE_EVERY)

        self.memory = PriorityReplayBuffer(action_size, batch_size, seed)

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, weights, indices = experiences

        local_max_actions = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
        # Get max predicted Q values (for next states) from target model
        Q_targets_next = torch.gather(self.qnetwork_target(next_states).detach(),1,local_max_actions)
#         Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = (Q_expected - Q_targets).pow(2)*weights
        prios = loss + 1e-5
        loss = loss.mean()
#         loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update priorities based on td error
        self.memory.update_priorities(indices.squeeze().to("cpu").data.numpy(),prios.squeeze().to("cpu").data.numpy())

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

class Dueling(PriorityReplay):
    """Inspired by code at https://github.com/dxyang/DQN_pytorch/blob/master/model.py."""
    def __init__(self, state_size, action_size, seed):
        super(Dueling, self).__init__(state_size, action_size, seed)
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        self.conv1 = nn.Conv2d(in_channels=state_size, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4,stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        self.fc1_adv = nn.Linear(in_features=7*7*64, out_features=512)
        self.fc1_val = nn.Linear(in_features=7*7*64, out_features=512)

        self.fc2_adv = nn.Linear(in_features=512, out_features=action_size)
        self.fc2_val = nn.Linear(in_features=512, out_features=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0),-1)

        adv = self.relu(self.fc1_adv(x))
        val = self.relu(self.fc1_val(x))

        adv = self.fc2_adv(adv)
        val = self.fc2_val(val).expand(x.size(0),self.action_size)

        x = val+adv-adv.mean(1).unsqueeze(1).expand(x.size(0),self.action_size)
        return x

def train_gym(CHART_PATH, CHECKPOINT_PATH, module, timestamp, seed, score_target,
              n_episodes,max_t,eps_start,eps_end,eps_decay): #agent_dict,
    import gym
    start = time.time()
    label = "gym"
    env = gym.make(module)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent_dict = {
              "Dueling":Dueling(state_size,action_size,seed),
              "PriorityReplay":PriorityReplay(state_size, action_size, seed),
              "Double":Double(state_size, action_size, seed),
              "Vanilla":Vanilla(state_size, action_size, seed)
             }
    result_dict = {}
    for k,v in agent_dict.items():
        agent_name = k
        agent = v
        scores = []                        # list containing scores from each episode
        scores_window = deque(maxlen=100)  # last 100 scores
        eps = eps_start
        for i_episode in range(1, n_episodes+1):
            state = env.reset()
            score = 0
            for t in range(max_t):
                action = agent.act(state,eps)
                next_state, reward, done, _ = env.step(action)
                agent.step(state, action, reward, next_state, done)
                state = next_state
                score += reward                                # update the score
                if done:                                       # exit loop if episode finished
                    break
            scores_window.append(score)       # save most recent score
            scores.append(score)              # save most recent score
            eps = max(eps_end, eps_decay*eps) # decrease epsilon
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
            if i_episode % 100 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            if np.mean(scores_window)>=score_target:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
                checkpath = CHECKPOINT_PATH + f'checkpoint-{timestamp}-{label}-{module}-{agent_name}.pth'
                torch.save(agent.qnetwork_local.state_dict(), checkpath)
                print(f"Checkpoint saved at {checkpath}")
                break
        end = time.time()
        result_dict[agent_name] = {
                        "scores":scores,
                        "clocktime":round((end-start)/60,2)
                        }
        pklpath = CHART_PATH + f"ResultDict-{timestamp}-{label}-{module}.pkl"
        with open(pklpath, 'wb') as handle:
            pickle.dump(result_dict, handle)
        print(f"Scores pickled at {pklpath}")
    return result_dict

def train_unity(APP_PATH, CHART_PATH, CHECKPOINT_PATH, timestamp, seed, score_target,
                n_episodes,max_t,eps_start,eps_end,eps_decay):
    from unityagents import UnityEnvironment
    label = "unity"
    start = time.time()
    env = UnityEnvironment(file_name=APP_PATH)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset()[brain_name]
    state_size = len(env_info.vector_observations[0])
    action_size = brain.vector_action_space_size
    agent_dict = {
              "Dueling":Dueling(state_size,action_size,seed),
              "PriorityReplay":PriorityReplay(state_size, action_size, seed),
              "Double":Double(state_size, action_size, seed),
              "Vanilla":Vanilla(state_size, action_size, seed)
             }
    result_dict = {}
    for k,v in agent_dict.items():
        agent_name = k
        agent = v
        scores = []                        # list containing scores from each episode
        scores_window = deque(maxlen=100)  # last 100 scores
        eps = eps_start
        for i_episode in range(1, n_episodes+1):
            state = env.reset()
            score = 0
            for t in range(max_t):
                state = env_info.vector_observations[0]  # get the current state
                action = agent.act(state,eps)
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
            eps = max(eps_end, eps_decay*eps) # decrease epsilon
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
            if i_episode % 100 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            if np.mean(scores_window)>=score_target:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
                checkpath = CHECKPOINT_PATH + f'checkpoint-{timestamp}-{label}-{module}-{agent_name}.pth'
                torch.save(agent.qnetwork_local.state_dict(), checkpath)
                print(f"Checkpoint saved at {checkpath}")
                break
        end = time.time()
        result_dict[agent_name] = {
                        "scores":scores,
                        "clocktime":round((end-start)/60,2)
                        }
        pklpath = CHART_PATH + f"ResultDict-{label}-{timestamp}.pkl"
        with open(pklpath, 'wb') as handle:
            pickle.dump(result_dict, handle)
        print(f"Scores pickled at {pklpath}")
    return result_dict

def chart_results(CHART_PATH, pklfile):
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

def train_envs(PATH, CHART_PATH, CHECKPOINT_PATH, timestamp, env_dict, seed=0,
               n_episodes=3000,max_t=1000,eps_start=0.4,eps_end=0.01,eps_decay=0.995):
    for k,v in env_dict.items():
        start = time.time()
        module = k
        platform = v[0]
        print(f"Begin training {module}-{platform}.")
        score_target = v[1]
        # seed = 0
        # n_episodes=3000
        # max_t=1000
        # eps_start=0.4
        # eps_end=0.01
        # eps_decay=0.995
        print(f"Module: {module}-{platform}")
        if platform == "gym":
            results = train_gym(CHART_PATH, CHECKPOINT_PATH, module, timestamp, seed, score_target,
                            n_episodes,max_t,eps_start,eps_end,eps_decay)
        elif platform == "unity":
            APP_PATH = PATH + "data/Banana.app"
            results = train_unity(APP_PATH, CHART_PATH, CHECKPOINT_PATH, timestamp, seed, score_target,
                          n_episodes,max_t,eps_start,eps_end,eps_decay)
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
