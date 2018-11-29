from dqn_helper4 import *

import re
import datetime
# import numpy as np
# import pandas as pd

# import gym
# from unityagents import UnityEnvironment

# path information
PATH = "/Volumes/BC_Clutch/Dropbox/DeepRLND/rl_navigation/"
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
            "Bananas":["unity",13.0],
            "LunarLander-v2":["gym",200.0],
            "CartPole-v0":["gym",195.0],
            "MountainCar-v0":["gym",-110],
            }

rd = train_envs(PATH, CHART_PATH, CHECKPOINT_PATH, timestamp, env_dict, seed,
                n_episodes,max_t,eps_start,eps_end,eps_decay)
