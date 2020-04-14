import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from typing import Tuple, Callable


def hidden_init(layer: torch.nn.modules.linear.Linear) -> Tuple[float, float]:
    """Initialize layer weights randomly depending on the size of the input.

    :param layer: nn.Linear layer
    :return: Bounds for a uniform random distribution from which to sample for random weight initialization
    """

    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return -lim, lim


class Network(nn.Module):
    """Deep neural network with one hidden layer."""

    def __init__(self, input_size: int, output_size: int, activation: Callable = None, seed: int = 0) -> None:

        super(Network, self).__init__()

        self.seed = torch.manual_seed(seed)
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation

        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_size)
        self.batch_norm = nn.BatchNorm1d(256)
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights randomly, with near zero values."""
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, inp):
        if inp.dim() == 1:
            inp = torch.unsqueeze(inp, 0)
        x = F.relu(self.fc1(inp))
        x = self.batch_norm(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # Activation is for actor network to keep actions within (-1, 1)
        if self.activation is not None:
            x = self.activation(x)

        return x
