"""
DQN implementations.
Adapted from code at https://github.com/udacity/deep-reinforcement-learning/tree/master/dqn
"""
from dqn_helper2 import *

import numpy as np
import pandas as pd

# import gym
from unityagents import UnityEnvironment

# path information
PATH = "/Volumes/BC_Clutch/Dropbox/DeepRLND/rl_navigation/"
CHART_PATH = PATH + "charts/"
CHECKPOINT_PATH = PATH + "models/"

def train_unity(APP_PATH, CHART_PATH, timestamp, seed, score_target, 
                n_episodes,max_t,eps_start,eps_end,eps_decay):
    label = "unity"
    start = time.time()
    env = UnityEnvironment(file_name=APP_PATH)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset()[brain_name]
    state_size = len(env_info.vector_observations[0])
    action_size = brain.vector_action_space_size   
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
                checkpath = CHECKPOINT_PATH + f'checkpoint-{label}-{agent_name}-{timestamp}.pth'
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
#    plot_results(CHART_PATH, result_dict, label, timestamp,100)
    return result_dict

APP_PATH = PATH + "data/Banana.app"
score_target = 13.0 # 13.0 to meet project goals
seed = 0
n_episodes=1000
max_t=1000
eps_start=0.5
eps_end=0.01
eps_decay=0.995

results = train_unity(APP_PATH, CHART_PATH, timestamp, seed, score_target, 
                      n_episodes,max_t,eps_start,eps_end,eps_decay)