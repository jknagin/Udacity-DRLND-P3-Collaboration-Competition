[Udacity's Deep Reinforcement Learning Nanodegree]: https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893
[scores]: images/scores.png
[MADDPG]: images/MADDPG.png

# Udacity Deep Reinforcement Learning Nanodegree Project 3: Collaboration and Competition

## Introduction
The purpose of this project is to train two agents to play tennis together.

## Environment
In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to hit the ball over the net.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. The maximum of these scores is taken.
- This yields a single **score** for each episode.

## Multi Agent Deep Deterministic Policy Gradient (MADDPG) Algorithm
Multi Agent Deep Deterministic Policy Gradient (MADDPG) is an appropriate algorithm for this use case between the agents are interacting with the environment together.

The MADDPG algorithm extends the DDPG algorithm to multiple interacting DDPG agents. Each DDPG agent computes its individual action using its individual observation of the environment. However, each DDPG agent has access to the other DDPG agents' actions and observations during training. In particular, each agent's critic network maps the observations and actions of all agents in one time step to an action-value for that time step.

The image below is a description of the algorithm in pseudocode, taken from the [following paper](https://arxiv.org/abs/1706.02275).

![MADDPG]

## Implementation
### Descriptions of each file
#### `ddpg_agent.py`
Implements an individual DDPG agent class, providing the following methods:
<!-- TODO -->

#### `maddpg_agent.py`
Implements the MADDPG algorithm using the MADDPG_Agent class, providing the following methods:
<!-- TODO -->

#### `model.py`
Implements a simple neural network with one hidden layer and one batch normalization layer. These networks are used to approximate actions (actor) and action-values (critic).

#### `ou_noise.py`
Implements an Ornstein-Uhlenbeck noise generator, providing the following methods:
<!-- TODO -->

#### `replay_buffer.py`
Implements an experience replay buffer, providing the following methods:
<!-- TODO -->

#### `Collaboration_Competition.ipynb`
Main notebook for running the code. The notebook loads the Tennis environment, instantiates the MADDPG agent, trains the agent, saves the agent's weight file, and plots the agent's score per episode during training. The notebook can also be used to load a weight file into an MADDPG agent and play the environment to see how well the agent performs.

### Hyperparameters
<!-- TODO -->
| Hyperparameter | Value | Description | Defined In |
