# Overview

For [Project 1](https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation) of Udacity's [Deep Reinforcement Learning Nanodegree](https://github.com/udacity/deep-reinforcement-learning), we were tasked with teaching an agent to navigate a Unity environment where it must collect yellow bananas (reward +1) while avoiding blue bananas (reward -1).

A Deep Q Network (DQN) is used in training the agent to solve the environment.  In this project, we implement the following versions of DQN:

1. "Vanilla", or base implementation. This is based on DeepMind's research on ["Human-level control through deep reinforcement learning"](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) (Nature, 26 February 2015)

2. Double, which addresses the inherent overestimation of action values in the base case.  Research from Hasselt's ["Deep Reinforcement Learning with Double Q-learning"](https://arxiv.org/abs/1509.06461)(Arxiv, 8 December 2015).  *Class inherits from Vanilla.*

3. Prioritized Experience Replay, which replays important episodes more frequently to learn more efficiently.  Research from Schaul's ["Prioritized Experience Replay"](https://arxiv.org/abs/1511.05952) (Arxiv, 25 February 2016) *Class inherits from Double.*

4. Dueling, with a neural network architecture for model-free learning.  The dueling network reprsents two separate estimators, one fo the state value function and the other for the state-dependent action advantage function.  This generalizes learning across actions without imposing change to the underlying RL algorithm. Research from Wang's ["Dueling Network Architectures for Deep Reinforcement Learning"](https://arxiv.org/abs/1511.06581) (Arxiv, 5 April 2016) *Class inherits from Priority Replay.*

5. A3C, which learns from multi-step bootstrap targets using "asynchronous gradient descent for optimization of deep neural network controllers".  Research from Mnih's ["Asynchronous Methods for Deep Reinforcement Learning"](https://arxiv.org/abs/1602.01783) (Arxiv, 16 June 2016) *Class inherits from Dueling.*

Other popular implementations of the DQN algorithm that may be implemented into this project at a later date include:

6. Distributional, which applies Bellman's equation to the learning of approximate value distributions (in contrast to solely modelling the value ie expectation of the return).  Research per Bellemare's ["A Distributional Perspective on Reinforcement Learning"](https://arxiv.org/abs/1707.06887) (Arxiv, 21 July 2017)

7. Noisy, which adds parametric noise to the weights which can aid efficient exploration.  Research per Fortunato's ["Noisy Networks for Exploration"](https://arxiv.org/abs/1706.10295)(Arxiv, 15 February 2018)

- Rainbow, which incorporates all of the above modifications.  Research per Hessel's ["Rainbow: Combining Improvements in Deep Reinforcement Learning"](https://arxiv.org/abs/1710.02298)(Arxiv, 6 October 2017)

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

# Next Steps
**TODO: add concrete future ideas for improving agent performance**


# TODO

**Process**
- read all research papers in DQN lesson
- work through all openai gym code examples in DQN lesson


**Project Deliverables**
- rewrite readme, describing how someone not familiar with the project should use this repo
  - describe the environment solved
  - describe how to install the requirements before running code in repo
  - report describing the learning algo, including details of implementation and ideas for future work
