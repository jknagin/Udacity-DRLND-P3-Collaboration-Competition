import torch
from unityagents import UnityEnvironment
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from maddpg_agent import MADDPG_Agent
import pandas as pd
from typing import List

TRAIN = True
env = UnityEnvironment(file_name="Tennis.app")


def maddpg(agent_: MADDPG_Agent, solution_score: float = 0.5, n_episodes: int = 10000) -> List[float]:
    """ Train an agent for a number of episodes.

    :param agent_: MADDPG_Agent to be trained
    :param solution_score: score at which environment is considered solved
    :param n_episodes: number of episodes to train the MADDPG agent
    :return: list of scores, one per episode, max over all DDPG agents' scores for each episode
    """

    all_scores = []
    latest_scores = deque(maxlen=100)
    for i in range(n_episodes):
        score = agent_.train_for_episode()
        latest_scores.append(score)
        all_scores.append(score)

        # Print status updates
        print('\rEpisode {}\tAverage Score: {:.4f}'.format(i, np.mean(latest_scores)), end="")
        if i % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.4f}'.format(i, np.mean(latest_scores)))

        # Notify when environment is solved and save agent model parameters and score
        if np.mean(latest_scores) >= solution_score:
            print("\nEnvironment solved in {} episodes".format(i + 1))
            agent.save()  # Save local model weights to solution.pth
            np.save('scores.npy', np.array(all_scores))
            break

    return all_scores


def play_with_trained_model(agent_: MADDPG_Agent, n_episodes: int = 1000) -> List[float]:
    """ Play the environment with a (trained) agent.

    :param agent_: MADDPG_Agent to play the environment with. This agent should be trained.
    :param n_episodes: number of episodes to play
    :return: list of scores, one per episode
    """

    all_scores = []
    for _ in range(n_episodes):
        rewards_list = []
        env_info = agent_.env.reset(train_mode=False)[agent_.brain_name]  # reset the environment
        state = env_info.vector_observations.reshape(1, -1)
        while True:
            actions = agent_.act(torch.from_numpy(state.reshape(1, -1)).float())
            env_info = env.step(actions)[agent_.brain_name]
            state = env_info.vector_observations.reshape(1, -1)
            rewards = env_info.rewards
            rewards_list.append(rewards)
            dones = env_info.local_done
            if np.any(dones):
                break
        all_scores.append(np.max(np.sum(np.array(rewards_list), axis=0)))

    return all_scores


# Create agent
agent = MADDPG_Agent(env, seed=3)

if TRAIN:
    # Train new agent from scratch
    scores = maddpg(agent, n_episodes=10000)

    # Plot scores
    scores_100 = pd.Series(scores).rolling(window=100).mean().iloc[99:].values  # rolling average of 100 scores
    plt.figure()
    plt.plot(scores, color='b', label='All Scores')
    plt.plot(scores_100, color='r', label='Average of latest 100 scores', linewidth=5)
    plt.legend()
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.show()

else:
    # Load existing weights into an agent and play the environment
    agent.load()
    agent.add_noise = False
    scores = play_with_trained_model(agent)

env.close()
