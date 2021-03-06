<a name="report"></a>
# Reinforcement Learning Navigation: Project Report

## Report Table of Contents

[RL Environment](#environment)

[Algorithm](#algorithm)

[Hyperparameters](#hyperparameters)

[Network Architecture](#network)

[Next Steps](#nextsteps)

<a name="environment"></a>
## The Reinforcement Learning (RL) Environment

Train an agent to collect bananas while navigating a large square world.  

Yellow bananas are worth +1.0 point while a blue is -1.0 point. The environment is considered solved once the agent is able to attain an average score of 13.0 over 100 consecutive episodes.

State space: 37 dimensions containing agent's velocity and ray-based perception of objects around the agent's forward direction.

Action space: 4 possible actions for the agent to take, including:
- 0 forward
- 1 backward
- 2 turn left
- 3 turn right

Additional Unity environment details can be found on the [project description](https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation).  

<a name="algorithm"></a>
## The Algorithm

A Deep Q Network (DQN) is used in training the agent to solve the environment.  In this project, we implement the following versions of DQN:

1. "Vanilla", or base implementation. This is based on DeepMind's research on ["Human-level control through deep reinforcement learning"](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) (Nature, 26 February 2015)

2. Double, which addresses the inherent overestimation of action values in the base case.  Research from Hasselt's ["Deep Reinforcement Learning with Double Q-learning"](https://arxiv.org/abs/1509.06461)(Arxiv, 8 December 2015).  *Class inherits from Vanilla.*

3. Prioritized Experience Replay, which replays important episodes more frequently to learn more efficiently.  More frequently replay transitions with high expected learning progress as measured by the magnitude of their temporal-difference (TD) error. Research from Schaul's ["Prioritized Experience Replay"](https://arxiv.org/abs/1511.05952) (Arxiv, 25 February 2016) *Class inherits from Double.*

4. Dueling, with a neural network architecture for model-free learning.  The dueling network reprsents two separate estimators, one fo the state value function and the other for the state-dependent action advantage function.  This generalizes learning across actions without imposing change to the underlying RL algorithm. Research from Wang's ["Dueling Network Architectures for Deep Reinforcement Learning"](https://arxiv.org/abs/1511.06581) (Arxiv, 5 April 2016) *Class inherits from Priority Replay.*

5. A3C, which learns from multi-step bootstrap targets using "asynchronous gradient descent for optimization of deep neural network controllers".  Research from Mnih's ["Asynchronous Methods for Deep Reinforcement Learning"](https://arxiv.org/abs/1602.01783) (Arxiv, 16 June 2016) *Class inherits from Dueling.*

![alt text](https://github.com/cipher813/rl_navigation/blob/master/charts/RLTrainChart-201812021453-Banana_Linux_NoVisBanana.x86_64-Vanilla.png "Banana Results by Algorithm")

<a name="hyperparameters"></a>
## Hyperparameters

Hyperparameters used in all implementations of the DQN algorithm include:
- learning rate:  2.5e-4  learning rate used by RMSProp
- buffer size:    1e5     experience replay buffer
- batch size:     64      SGD update by this number of training cases
- alpha:          0.4     prioritization factor (a=0 is uniform)
- beta:           1.0     importance-sampling weight
- gamma:          0.99    discount factor
- tau:            1e-3    soft update of target parameters
- frequency:      4       how often to update the network

<a name="network"></a>
# Neural Network Architecture

The underlying Q Network used in the implementation is a 3 layer network with the input and output layers mapping to the state and action sizes of 37 and 4 (per [Environment](#environment)), respectively.  There are two hidden layers of 64 nodes each which ReLU as the activation layers.  The output is a fully-connectd linear layer with a single output for each valid action.  

<a name="nextsteps"></a>
# Next Steps

Potential areas to explore in further work include:

**Algorithm implementations**

In this project we explored several implementations of the Deep Q Network (DQN) algorithm.  

Other popular implementations of the DQN algorithm which may be implemented into this project at a later date include:

6. Distributional, which applies Bellman's equation to the learning of approximate value distributions (in contrast to solely modelling the value ie expectation of the return).  Research per Bellemare's ["A Distributional Perspective on Reinforcement Learning"](https://arxiv.org/abs/1707.06887) (Arxiv, 21 July 2017)

7. Noisy, which adds parametric noise to the weights which can aid efficient exploration.  Research per Fortunato's ["Noisy Networks for Exploration"](https://arxiv.org/abs/1706.10295)(Arxiv, 15 February 2018)

8. Rainbow, which incorporates all of the above modifications.  Research per Hessel's ["Rainbow: Combining Improvements in Deep Reinforcement Learning"](https://arxiv.org/abs/1710.02298)(Arxiv, 6 October 2017) *The Rainbow and related NoisyLinear classes are currently a work in progress in the scripts in this repo*


**Environments**

Our current experiments including testing the algorithm on the Udacity/Unity custom Banana collector environment (not to be confused with Unity's [Banana Collector environment](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md)).  We also tested our algorithm implementations on OpenAI gym environments, including [Lunar Lander](https://gym.openai.com/envs/LunarLander-v2/) and [CartPole](https://gym.openai.com/envs/CartPole-v1/).

We could experiment with additional environment types, including [Atari](https://gym.openai.com/envs/#atari) or with a physics simulator engine such as [MuJoCo](https://gym.openai.com/envs/#mujoco).  Unity's ML Agents platform also features a variety of other [interesting environments](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md) which could be implemented.  
