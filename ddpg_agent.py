from model import Network
import torch.nn.functional as F
from torch.optim import Adam
from ou_noise import OUNoise
import torch
import random
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TAU = 0.001
LR_ACTOR = 1e-4
LR_CRITIC = 1e-3
WEIGHT_DECAY_CRITIC = 0.0


class DDPGAgent:

    def __init__(self, _id: int, observation_size: int, action_size: int, num_agents: int, seed=0) -> None:
        """

        :param _id: ID for extracting data from tensors using index_select
        :param observation_size: dimension of the individual agent's observation space
        :param action_size: dimension of the each individual agent's action space
        :param num_agents: number of agents, used by critic network to compute action-values over the joint action space
        :param seed: random number generator seed
        """

        random.seed(seed)
        self.id = torch.Tensor([_id]).type(torch.LongTensor).to(device)
        self.observation_size = observation_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.state_size = self.observation_size * self.num_agents
        self.joint_action_size = self.action_size * self.num_agents
        self.actor_local = Network(self.observation_size, self.action_size, activation=F.tanh, seed=seed).to(
            device)
        self.actor_target = Network(self.observation_size, self.action_size, activation=F.tanh, seed=seed).to(
            device)
        self.critic_local = Network(self.state_size + self.joint_action_size, 1, seed=seed).to(device)
        self.critic_target = Network(self.state_size + self.joint_action_size, 1, seed=seed).to(device)

        self.tau = TAU

        self.actor_optimizer = Adam(self.actor_local.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY_CRITIC)

        self.hard_update()

        self.noise = OUNoise(self.action_size, seed)

    def local_act_noisy(self, observation: torch.Tensor, noise: float = 0.0, add_noise: bool = True) -> np.ndarray:
        """

        :param observation: agent's observation, size = 1 x self.observation_size where P = batch size
        :param noise: OU noise amplitude to add randomness to actions for exploration
        :param add_noise: whether or not to add noise to the actions
        :return: action vector for the agent
        """

        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(observation).cpu().data.numpy()  # size = 1 x self.action_size = 1 x self.action_size
        self.actor_local.train()
        return np.reshape(np.clip(actions + int(add_noise) * noise * self.noise.sample(), -1, 1), (-1, self.action_size))

    def soft_update(self, tau: float = None) -> None:
        """Soft update target networks' weights using local networks' weights and hyperparameter TAU."""

        if tau is None:
            tau = TAU
        for target_param, param in zip(self.actor_target.parameters(), self.actor_local.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        for target_param, param in zip(self.critic_target.parameters(), self.critic_local.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def hard_update(self) -> None:
        """Copy weights from local networks into target networks."""

        self.soft_update(tau=1)
