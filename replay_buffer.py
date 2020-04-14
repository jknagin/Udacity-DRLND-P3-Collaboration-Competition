import random
import numpy as np
from collections import deque, namedtuple
import torch
from typing import List, Tuple

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    """ Buffer class for handling and sampling experiences."""

    def __init__(self, buffer_size: int, batch_size: int, seed: int = 0) -> None:
        """Initialize buffer with given capacity.

        :param buffer_size: maximum number of experiences to store
        :param batch_size: number of experiences to sample per call to sample()
        :param seed: random number generator seed
        """
        random.seed(seed)
        np.random.seed(seed)
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state: np.ndarray, action: np.ndarray, reward: List[float], next_state: np.ndarray, done: List[bool]) -> None:
        """Add a (S, A, R, S, done) tuple to the buffer."""

        self.memory.append(self.experience(state, action, reward, next_state, done))

    def sample(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a batch of experiences."""

        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)
