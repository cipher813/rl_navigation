"""
DQN implementations.
Adapted from code at https://github.com/udacity/deep-reinforcement-learning/tree/master/dqn
"""
from dqn_helper3 import *

import time 
import numpy as np
import pandas as pd

import gym
#from unityagents import UnityEnvironment

# path information
PATH = "/Volumes/BC_Clutch/Dropbox/DeepRLND/rl_navigation/"
CHART_PATH = PATH + "charts/"
CHECKPOINT_PATH = PATH + "models/"

def train_gym(CHART_PATH, module, timestamp, seed, score_target,
              n_episodes,max_t,eps_start,eps_end,eps_decay): #agent_dict,
    start = time.time()
    label = "gym"
    env = gym.make(module)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent_dict = {
              "Vanilla":Vanilla(state_size, action_size, seed),
              "Double":Double(state_size, action_size, seed),
              "PriorityReplay":PriorityReplay(state_size, action_size, seed)
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
                checkpath = CHECKPOINT_PATH + f'checkpoint-{label}-{module}-{agent_name}-{timestamp}.pth'
                torch.save(agent.qnetwork_local.state_dict(), checkpath)
                print(f"Checkpoint saved at {checkpath}")
                break
        end = time.time()
        result_dict[agent_name] = {
                        "scores":scores,
                        "clocktime":round((end-start)/60,2)
                        }
    pklpath = CHART_PATH + f"ResultDict-{label}-{module}-{timestamp}.pkl"
    with open(pklpath, 'wb') as handle:
        pickle.dump(result_dict, handle)
    print(f"Scores pickled at {pklpath}")
#    plot_results(CHART_PATH, result_dict, label, timestamp,100)
    return result_dict

env_dict = {
            "LunarLander-v2":200.0,
            "CartPole-v0":195.0,
            "MountainCar-v0":-110,
            # "BipedalWalker-v2":-300.0
            }

rd = {}
for k,v in env_dict.items():
    module = k
    score_target = v
    seed = 0
    n_episodes=5000
    max_t=1000
    eps_start=0.4
    eps_end=0.01
    eps_decay=0.995
    print(f"Module: {module}")
    results = train_gym(CHART_PATH, module, timestamp, seed, score_target,
                        n_episodes,max_t,eps_start,eps_end,eps_decay)
    rd[module] = results
pklpath = CHART_PATH + f"ResultDict-AllGym-{timestamp}.pkl"
with open(pklpath, 'wb') as handle:
    pickle.dump(rd, handle)
    print(f"Scores pickled at {pklpath}")
