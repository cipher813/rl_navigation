"""
DQN implementations.
Adapted from code at https://github.com/udacity/deep-reinforcement-learning/tree/master/dqn
"""
# imports
import re
import time
import pickle
import datetime
import numpy as np

from dqn_helper import *

# path information
PATH = "/Volumes/BC_Clutch/Dropbox/DeepRLND/rl_navigation/"
APP_PATH = PATH + "data/Banana.app"
CHART_PATH = PATH + "charts/"
CHECKPOINT_PATH = PATH + "models/"

score_target = 0.0 # 13.0 to meet project goals

state_size = 37
action_size = 4
seed = 0
n_episodes=1000
max_t=1000
eps_start=0.4
eps_end=0.01
eps_decay=0.995
train_mode=True

timestamp = re.sub(r"\D","",str(datetime.datetime.now()))[:12]

def dqn(model_name, n_episodes, max_t, eps_start, eps_end, eps_decay,train_mode):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    env_info = env.reset(train_mode)[brain_name]
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start
    print("Loading monkey.")
    print("Monkey training on bananas.")
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            state = env_info.vector_observations[0]            # get the current state
            action = agent.act(state, eps)
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
            checkpath = CHECKPOINT_PATH + f'checkpoint-{model_name}-{timestamp}.pth'
            torch.save(agent.qnetwork_local.state_dict(), checkpath)
            print(f"Checkpoint saved at {checkpath}")
            break
    return scores

# setup and examine environment
env = UnityEnvironment(file_name=APP_PATH)

brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode)[brain_name]

# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space
state = env_info.vector_observations[0]
print('States look like:\n', state)
state_size = len(state)
print('States have length:', state_size)

# train agent
result_dict = {}
function_list = [
                 ("Base",Base(state_size, action_size, seed)),
                 ("Double", Double(state_size, action_size, seed)),
                 ("PrioritizedReplay",PrioritizedReplay(state_size, action_size, seed))
                ]

for function in function_list:
    start = time.time()
    name = function[0]
    agent = function[1]
    print(f"**{name}**")
    scores = dqn(name, n_episodes, max_t, eps_start, eps_end, eps_decay,train_mode)
    end = time.time()
    result_dict[name] = {
                    "scores":scores,
                    "clocktime":round((end-start)/60,2),
                    "state_size":state_size,
                    "action_size":action_size,
                    "seed":seed,
                    "n_episodes":n_episodes,
                    "max_t":max_t,
                    "eps_start":eps_start,
                    "eps_end":eps_end,
                    "eps_decay":eps_decay,
                    "train_mode":train_mode
                    }

pklpath = CHART_PATH + f"ResultDict-{timestamp}.pkl"
with open(pklpath, 'wb') as handle:
    pickle.dump(result_dict, handle)
print(f"Scores pickled at {pklpath}")
