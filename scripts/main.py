"""
Deep Q Network (DQN) main train file.
Project 1: Navigation
Udacity Deep Reinforcement Learning Nanodegree
Brian McMahon
December 2018
"""
from util import train_gym, train_unity, train_envs, chart_results
from agent import Vanilla, Double, PriorityReplay, Dueling, A3C, Rainbow

import re
import datetime

# path information
PATH = "/Volumes/BC_Clutch/Dropbox/DeepRLND/rl_navigation/" # for mac
# PATH = "/home/bcm822_gmail_com/rl_navigation/"
CHART_PATH = PATH + "charts/"
CHECKPOINT_PATH = PATH + "models/"

timestamp = re.sub(r"\D","",str(datetime.datetime.now()))[:12]

seed = 0
n_episodes=3000
max_t=1000
eps_start=0.4
eps_end=0.01
eps_decay=0.995

rd = {}
env_dict = {
            # unity, for google cloud
             # "Banana_Linux_NoVis/Banana.x86_64":["unity",13.0],
#            "VisualBanana_Linux_NoVis/Banana.x86_64":["unity",13.0],

            # unity, for mac
           "Banana.app":["unity",13.0],
            # "VisualBanana.app":["unity",13.0],

            # OpenAI Gym
            "LunarLander-v2":["gym",200.0],
            # "CartPole-v0":["gym",195.0],
            # "MountainCar-v0":["gym",-110],
            }

agent_dict = {
              # "Rainbow":Rainbow,
              "A3C":A3C,
              "Dueling":Dueling,
              "PriorityReplay":PriorityReplay,
              "Double":Double,
              "Vanilla":Vanilla
             }

rd = train_envs(PATH, CHART_PATH, CHECKPOINT_PATH, agent_dict, timestamp, env_dict, seed,
                n_episodes,max_t,eps_start,eps_end,eps_decay)

#results = chart_results(CHART_PATH, rd)
