# Udacity-DRLND-P3-Collaboration-Competition

[Trained Agent]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"

[Udacity repository]: https://github.com/udacity/deep-reinforcement-learning#dependencies

# Project 3: Collaboration and Competition

### Introduction

In this project, two agents were trained to play tennis together.

![Trained Agent][Trained Agent]

Two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to hit the ball over the net.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. The maximum of these scores is taken.
- This yields a single **score** for each episode.

### Getting Started

1. Follow the instructions on the [Udacity repository] to configure a Python environment with the dependencies and Unity environments.

1. Clone this project and ensure it can be ran with the Python environment.

1. Download the Tennis environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - macOS: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)


1. Place the file in the root directory of this repo and unzip the file.


1. Follow the instructions in the project [report](https://github.com/jknagin/Udacity-DRLND-P3-Collaboration-Competition/blob/master/REPORT.md#running-collaboration_competitionipynb) and the main Jupyter [notebook](https://github.com/jknagin/Udacity-DRLND-P3-Collaboration-Competition/blob/master/Collaboration_Competition.ipynb) to get started with training the agent.

