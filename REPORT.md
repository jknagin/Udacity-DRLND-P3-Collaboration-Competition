[Udacity's Deep Reinforcement Learning Nanodegree]: https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893
[scores]: images/scores.png
[MADDPG]: images/MADDPG.png

# Udacity Deep Reinforcement Learning Nanodegree Project s3: Collaboration and Competition

## Introduction
The purpose of this project is to train two agents to play tennis together.

## Environment
In this environment, two DDPG agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each DDPG agent is to hit the ball over the net.

The observation space for each DDPG agent consists of 3 stacked frames of 8 variables corresponding to the position and velocity of the ball and racket, for a total observation size of 24. Each DDPG agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. The maximum of these scores is taken.
- This yields a single score for each episode.

## Multi Agent Deep Deterministic Policy Gradient (MADDPG) Algorithm
Multi Agent Deep Deterministic Policy Gradient (MADDPG) is an appropriate algorithm for this use case because the DDPG agents are interacting with the environment together.

The MADDPG algorithm extends the DDPG algorithm to multiple interacting DDPG agents. Each DDPG agent computes its individual action using its individual observation of the environment. However, each DDPG agent has access to the other DDPG agents' actions and observations during training. In particular, each agent's critic network maps the observations and actions of all agents in one time step to an action-value for that time step.

The image below is a description of the algorithm in pseudocode, taken from the [following paper](https://arxiv.org/abs/1706.02275).

![MADDPG]

## Implementation
### Descriptions of each file

#### `ddpg_agent.py`
Implements an individual DDPG agent class, providing the following methods:
* `__init__()`
  * Takes in each agent's observation and action space dimensions as arguments
  * Initializes actor and critic neural networks and optimizers

* `hard_update()`
  * Copies local AC network weights to target AC networks

* `local_act_noisy()`
  * Calculates an agent's action from an agent's observation of the environment using the local actor network

* `soft_update()`
  * Updates target AC network weights using update parameter `TAU`

#### `maddpg_agent.py`
Implements the MADDPG algorithm using the MADDPG_Agent class, providing the following methods:
* `__init__()`
  * The environment is provided as an argument
  * Initializes DDPG agents, OU noise generator, and replay buffer

* `act()`
  * Calculates actions for all agents using all states noiselessly

* `load()`
  * Loads model weights from a directory into each DDPG agent

* `_local_act()`
  * Calculates actions for all agents using all states
  * Adds OU noise to simulate exploration

* `_save()`
  * Saves model weights of each DDPG agent to a directory

* `_terminal_reset()`
  * Called by `train_for_episode` at the end of an episode
  * Zeros the episode score
  * Resets the state and environment

* `train_for_episode`
  * Trains the MADDPG agent for one episode using the MADDPG algorithm.
  * Updates each agent's AC networks every `UPDATE_EVERY` time steps

#### `model.py`
Implements a simple neural network with one hidden layer and one batch normalization layer. These networks are used to approximate actions (actor) and action-values (critic).

#### `ou_noise.py`
Implements an Ornstein-Uhlenbeck noise generator, providing the following methods:
* `__init__()`
  * Initializes the noise generator with default process parameters

* `reset()`
  * Resets the noise generator state

* `sample()`
  * Samples a noise vector from the noise generator

#### `replay_buffer.py`
Implements an experience replay buffer as a deque, providing the following methods:
* `__init__()`
  * Initializes deque to specified capacity `BUFFER_SIZE`

* `__len__()`
  * Returns current size of deque

* `add()`
  * Adds a (S, A, R, S, done) tuple to the deque

* `sample()`
  * Samples `BATCH_SIZE` unique elements from the buffer without replacement

#### `Collaboration_Competition.ipynb`
Main notebook for running the code. The notebook loads the Tennis environment, instantiates the MADDPG agent, trains the DDPG agents, saves the DDPG agents' weight files to a directory, and plots the MADDPG agent's score per episode during training. The notebook can also be used to load a directory containing weight files into an MADDPG agent and play the environment to see how well the agent performs.

### Hyperparameters
| Hyperparameter | Value | Description | Defined In |
|-|-|-|-|
| `BATCH_SIZE` | 1024 | Number of (S, A, R, S, done) tuples to sample from experience replay buffer at a time | `maddpg_agent.py` |
| `BUFFER_SIZE` | 100000 | Maximum number of (S, A, R, S, done) tuple experience replay buffer | `maddpg_agent.py` |
| `DISCOUNT` | 0.95 | Discount factor | `maddpg_agent.py` |
| `NOISE_ORIGINAL` | 1.0 | Original noise scaling factor | `maddpg_agent.py` |
| `NOISE_REDUCTION` | 1.0 | Noise reduction factor per time step | `maddpg_agent.py` |
| `T_STOP_NOISE` | 30000 | Number of time steps to inject noise into actions during training | `maddpg_agent.py` |
| `UPDATE_EVERY` | 10 | Number of time steps to update each DDPG agent's AC networks using the experience replay buffer | `maddpg_agent.py` |
| `LR_ACTOR` | 0.001 | Learning rate for Adam optimizer for actor network | `ddpg_agent.py` |
| `LR_CRITIC` | 0.01 | Learning rate for Adam optimizer for critic network | `ddpg_agent.py` |
| `TAU` | 0.01 | Soft update parameter for updating target AC weights with local AC weights | `ddpg_agent.py` |
| `WEIGHT_DECAY_CRITIC` | 0.0 | Weight decay factor for Adam optimizer for critic network | `ddpg_agent.py` |

### Network Architecture

The neural network architecture is defined fully in `model.py`.

| Layer | Input Dim | Output Dim | Activation | Notes |
|-|-|-|-|-|
| `FC1` | 24 (actor) or 52 (critic) | 256 | ReLU | For the actor network, the input dimension is the dimension of an individual DDPG agent's observation space. For the critic network, the input dimension is the dimension of the joint observation space plus the dimension of the joint action space for all agents. |
| `BN` | 256 | 256 | None | Batch norm layer. |
| `FC2` | 256 | 256 | ReLU | |
| `FC3` | 256 | 2 (actor) or 1 (critic) | tanh (actor) or None (critic) | For the actor network, the output dimension is the dimension of an individual DDPG agent's action space. For the critic network, the output dimension is 1 because the critic network returns a scalar action-value. |

### Running `Collaboration_Competition.ipynb`

To run the notebook, follow these steps:

1. Ensure that the following Python libraries are installed:
  * `numpy`
  * `pandas`
  * `matplotlib`
  * `pytorch`

  Also ensure that the Tennis environment is installed following the instructions in the README.

1. Run the first code cell to import all necessary libraries.

1. In the second code cell, update the `file_name` argument to the `UnityEnvironment` function with the location of the Tennis environment (`env = UnityEnvironment(file_name=...)`).

1. Run the second code cell to load the Tennis environment and instantiate the MADDPG agent.

1. Run the third code cell to train the MADDPG agent. The code will loop until the maximum number of episodes have been played (specified by `n_episodes`) or the agent achieves an average score of 0.5 or greater over the 100 most recent episodes. If the agent achieves such a score, this cell will save the DDPG agents' weights to a directory called `solution` and the list of scores to a file called `scores.npy`.

1. After training the MADDPG agent, run the next code cell to plot the score and the running average of the 100 most recent scores.

1. The next two code cells are used to load weights from an existing `solution` directory into the MADDPG agent and watch the DDPG agents perform in the Tennis environment.

1. The final code cell closes the environment.

## Results

### Score Plot
![scores]

The above image shows the plot of the MADDPG agent's score for each episode (blue) and the running average of the scores of the previous 100 episodes (red). **The MADDPG agent achieves an average score greater than or equal to 0.5 for 100 episodes after episode 2110.**

## Future Work
### Using a different network architecture
I did not experiment significantly with different neural network architectures for the actor and critic networks of each DDPG agent. A variety of network architectures could be tested to find a good tradeoff between bias and variance.

### Noise reduction
I did not reduce the OU noise scaling factor in this project (the value of the hyperparameter `NOISE_REDUCTION` is 1). Decreasing this hyperparameter to a value below 1 may allow the MADDPG to learn more quickly by reducing the amount of exploration over time.
