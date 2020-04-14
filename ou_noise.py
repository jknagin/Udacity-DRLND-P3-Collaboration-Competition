import numpy as np
import random
import copy


class OUNoise:
    """Ornstein-Uhlenbeck noise generator."""

    def __init__(self, size: int, seed: int, mu: float = 0.0, theta: float = 0.15, sigma: float = 0.2) -> None:
        """Construct noise generator.

        :param size: dimension of noise vector
        :param seed: random number generator seed
        :param mu: OU process parameter
        :param theta: OU process parameter
        :param sigma: OU process parameter
        """

        random.seed(seed)
        np.random.seed(seed)
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.state = copy.copy(self.mu)

    def reset(self) -> None:
        """Reset state."""

        self.state = copy.copy(self.mu)

    def sample(self) -> np.ndarray:
        """Sample a noise vector.

        :return: noise vector
        """

        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.size)
        self.state = x + dx
        return self.state
