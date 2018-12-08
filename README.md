# Reinforcement Learning Navigation

## Table of Contents

[Project Overview](#overview)

[Environment Setup](#setup)

[Model](#model)

[Report](https://github.com/cipher813/rl_navigation/blob/master/report.md#report)

- [RL Environment](https://github.com/cipher813/rl_navigation/blob/master/report.md#environment)

- [Algorithm](https://github.com/cipher813/rl_navigation/blob/master/report.md#algorithm)

- [Hyperparameters](https://github.com/cipher813/rl_navigation/blob/master/report.md#hyperparameters)

- [Network Architecture](https://github.com/cipher813/rl_navigation/blob/master/report.md#network)

- [Next Steps](https://github.com/cipher813/rl_navigation/blob/master/report.md#nextsteps)

<a name="overview"></a>
## Project Overview

For [Project 1](https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation) of Udacity's [Deep Reinforcement Learning Nanodegree](https://github.com/udacity/deep-reinforcement-learning), we were tasked with teaching an agent to navigate a Unity environment where it must collect yellow bananas (reward +1) while avoiding blue bananas (reward -1).  The Deep Q Network (DQN) algorithm is further explained in the accompanying [Report](https://github.com/cipher813/rl_navigation/blob/master/report.md).

<a name="setup"></a>
## Environment Setup

To set up the python (conda) environment, in the root directory, type:

`conda env update --file=environment_drlnd.yml`

This requires installation of [OpenAI Gym](https://github.com/openai/gym) and Unity's [ML-Agents](https://github.com/Unity-Technologies/ml-agents).

To download the Unity environment for your OS, see the links on the [Udacity Project Description](https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation).    

Potential areas of the project to expand upon in future work is explored in the accompanying [Report](https://github.com/cipher813/rl_navigation/blob/master/report.md).

<a name="model"></a>
## The Model

The key files in this repo include:

### Scripts

**network.py**
Contains classes for QNetwork, Replay Buffer (Vanilla), Priority Replay Buffer (Priority Replay agent) and Noisy Linear (Rainbow agent)

**agent.py**
Contains Deep Q Network (DQN) agent classes for Vanilla, Double, Priority Replay, Dueling, A3C and Rainbow (work in progress).  See [report.md](https://github.com/cipher813/rl_navigation/blob/master/report.md) for additional details on agent implementations.

**util.py**
Contains functions to train in Unity and OpenAI environments, and to chart results.

**main.py**
Execute this script to train in the environment(s) and agent(s) specified on this script in the environment and agent dictionaries, respectively.  


To train the agent, in the command line run:

`source activate drlnd` # to activate python (conda) environment
`python main.py` # to train the agent


### Notebooks

**rln_results.ipynb**

Charts the results from specified results dictionary pickle file.  

### Models

Contains the "checkpoint" model weights of each implementation.  
