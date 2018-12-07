# Overview

For [Project 1](https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation) of Udacity's [Deep Reinforcement Learning Nanodegree](https://github.com/udacity/deep-reinforcement-learning), we were tasked with teaching an agent to navigate a Unity environment where it must collect yellow bananas (reward +1) while avoiding blue bananas (reward -1).  The Deep Q Network (DQN) algorithm is further explained in the accompanying [Report](https://github.com/cipher813/rl_navigation/blob/master/report.md).



# The Reinforcement Learning (RL) Environment
**TODO link to Environment Info**

State space: 37

Action space: 4

The environment is considered solved once the agent is able to attain an average score of 13.0 over 100 episodes.  

# The Model

The key files in this repo include:

### Scripts

**network.py**


**agent.py**


**util.py**


**main.py**

To train the agent, in the command line run:

`source activate drlnd` # to activate python (conda) environment
`python main.py` # to train the agent


### Notebooks

**rln_results.ipynb**


### Models

Contains the model weights of each implementation.  

**TODO: Add Report.md**
- description of implementation
- learning algorithm
- chosen hyperparameters
- model architectures for neural networks

**TODO: Add plot of rewards per episode**


# Environment Setup

To set up the python (conda) environment, in the root directory, type:

`conda env update --file=environment_drlnd.yml`

This requires installation of [OpenAI Gym](https://github.com/openai/gym) and Unity's [ML-Agents](https://github.com/Unity-Technologies/ml-agents).

To download the Unity environment for your OS, see the links on the [Udacity Project Description](https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation).    

Potential areas of the project to expand upon in future work is explored in the accompanying [Report](https://github.com/cipher813/rl_navigation/blob/master/report.md).


# TODO
