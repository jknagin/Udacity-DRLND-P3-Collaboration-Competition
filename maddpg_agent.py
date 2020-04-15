from ddpg_agent import DDPGAgent
import torch
from replay_buffer import ReplayBuffer
import torch.nn.functional as F
import os
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 1024
BUFFER_SIZE = int(1e6)
DISCOUNT = 0.95
NOISE_ORIGINAL = 1.0
NOISE_REDUCTION = 1.0
T_STOP_NOISE = 30000
UPDATE_EVERY = 10



class MADDPG_Agent:
    """Implements MADDPG algorithm by handling multiple individual DDPG agents. """

    def __init__(self, env, seed=0) -> None:
        """

        :param env: UnityEnvironment environment
        :param seed: random number generator seed
        """

        # Save reference to environment and get info on the state and action space and number of agents
        self.env = env
        self.brain_name = env.brain_names[0]
        self.brain = env.brains[self.brain_name]
        env_info = self.env.reset(train_mode=True)[self.brain_name]
        self.state = env_info.vector_observations
        self.action_size = self.brain.vector_action_space_size
        self.observation_size = env_info.vector_observations.shape[1]
        self.num_agents = len(env_info.agents)

        # Create DDPG agents
        self.agents = []
        for x in range(self.num_agents):
            self.agents.append(DDPGAgent(x, self.observation_size, self.action_size, self.num_agents, seed=seed))

        # Initialize noise
        self.noise_original = NOISE_ORIGINAL
        self.noise = self.noise_original
        self.noise_reduction = NOISE_REDUCTION
        self.add_noise = True
        self.t_step = 0

        self.t_stop_noise = T_STOP_NOISE

        # Discount factor
        self.discount = DISCOUNT

        # Initialize experience replay buffer
        self.buffer = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed=seed)

        # Set episode score to 0 for each agent
        self.episode_score = np.zeros(self.num_agents)

    def act(self, states: np.ndarray) -> np.ndarray:
        """Calculate actions for all agents using all states noiselessly."""
        return self._local_act(torch.from_numpy(states.reshape(1, -1)).float().to(device))

    def load(self, solution_dir: str = 'solution') -> None:
        """Load model weights from directory.

        :param solution_dir: name of directory from which to load model weights
        """

        for i, ag in enumerate(self.agents):
            ag.actor_local.load_state_dict(torch.load(os.path.join(solution_dir, "actor{}.pth".format(i))))
            ag.critic_local.load_state_dict(torch.load(os.path.join(solution_dir, "critic{}.pth".format(i))))
            ag.hard_update()

    def _local_act(self, states: torch.Tensor, noise: float = 0.0) -> np.ndarray:
        """Get (noisy) actions from all agents using all agents' observations. P = batch size.

        :param states: all states as a P x 48 tensor
        :param noise: OU noise amplitude
        :return: actions for all agents as P x 4 numpy array
        """

        P = states.shape[0]
        actions = np.zeros((P, self.num_agents * self.action_size))
        for idx in range(P):
            for agent_idx in range(self.num_agents):
                actions[idx, agent_idx * self.action_size:(agent_idx + 1) * self.action_size] = self.agents[
                    agent_idx].local_act_noisy(
                    states[idx, self.observation_size * agent_idx:self.observation_size * (agent_idx + 1)],
                    noise=noise, add_noise=self.add_noise)
        return actions

    def save(self, solution_dir: str = 'solution') -> None:
        """Save model weights to directory.

        :param solution_dir: name of directory to save weights to
        """

        if not os.path.exists(solution_dir):
            os.mkdir(solution_dir)
        for i, ag in enumerate(self.agents):
            torch.save(ag.actor_local.state_dict(), os.path.join(solution_dir, "actor{}.pth".format(i)))
            torch.save(ag.critic_local.state_dict(), os.path.join(solution_dir, "critic{}.pth".format(i)))

    def _terminal_reset(self):
        """Zero the episode score. Reset the state and environment."""

        self.episode_score = np.zeros(self.num_agents)
        env_info = self.env.reset(train_mode=True)[self.brain_name]
        self.state = env_info.vector_observations

    def train_for_episode(self) -> float:
        """Train the agents for one episode.

        :return: maximum of all agents' undiscounted rewards for the episode
        """

        done = False
        while not done:  # Loop until an agent reaches a terminal state
            # Reduce action noise amplitude each episode to reduce exploration over time
            self.noise *= self.noise_reduction

            # Get actions from state
            # actions has size 1 x num_agents*action_size = 1 x 4
            actions = self._local_act(torch.from_numpy(self.state.reshape(1, -1)).float().to(device), noise=self.noise)

            # Step the environment with the chosen actions
            env_info = self.env.step(actions)[self.brain_name]

            # Get rewards, next state, and dones. Add to experience replay buffer.
            rewards = env_info.rewards  # size num_agents = 2
            next_state = env_info.vector_observations  # size num_agents x observation_size = 2 x 24
            dones = env_info.local_done  # size num_agents = 2
            done = any(env_info.local_done)
            self.buffer.add(self.state.reshape(1, -1), actions.reshape(1, -1), rewards, next_state.reshape(1, -1),
                            dones)

            # Add rewards to episode score for each agent
            self.episode_score += np.array(rewards)

            # Advance the state and time step counter
            self.state = next_state
            self.t_step += 1

            # Determine whether to stop injecting noise based on number of episodes trained
            if self.t_step > self.t_stop_noise:
                self.add_noise = False

            # Learn from experience replay buffer every UPDATE_EVERY episodes if buffer is long enough
            if self.t_step % UPDATE_EVERY == 0 and len(self.buffer) > BATCH_SIZE:
                experiences = [self.buffer.sample() for _ in range(self.num_agents)]
                all_actions = []
                all_next_actions = []
                # For each agent, calculate the actions and next actions using the agent's individual observations
                for idx, ag in enumerate(self.agents):
                    states, _, _, next_states, _ = experiences[idx]
                    # states and next_states: P x 48 tensors
                    observations_of_specific_agent = states.reshape(-1, 2, 24).index_select(1, ag.id).squeeze(
                        1)  # P x 24 tensor
                    next_observations_of_specific_agent = next_states.reshape(-1, 2, 24).index_select(1, ag.id).squeeze(
                        1)  # P x 24 tensor
                    new_actions_of_specific_agent = ag.actor_local(observations_of_specific_agent)  # P x 2 tensor
                    new_next_actions_of_specific_agent = ag.actor_target(
                        next_observations_of_specific_agent)  # P x 2 tensor
                    all_actions.append(new_actions_of_specific_agent)
                    all_next_actions.append(new_next_actions_of_specific_agent)

                all_next_actions = torch.cat(all_next_actions, dim=1).to(device)  # P x 4 tensor
                for idx, ag in enumerate(self.agents):
                    states, actions, rewards, next_states, dones = experiences[idx]

                    # Update local critic
                    ag.critic_optimizer.zero_grad()
                    with torch.no_grad():
                        q_target_next = ag.critic_target(torch.cat((next_states, all_next_actions), dim=1))
                    q_expected = ag.critic_local(torch.cat((states, actions), dim=1))
                    rewards_of_specific_agent = rewards.index_select(1, ag.id)
                    dones_of_specific_agent = 1 - dones.index_select(1, ag.id)
                    q_target = rewards_of_specific_agent + self.discount * q_target_next * dones_of_specific_agent
                    critic_loss = F.mse_loss(q_expected, q_target.detach())
                    critic_loss.backward()
                    ag.critic_optimizer.step()

                    # Update local actor
                    ag.actor_optimizer.zero_grad()
                    all_actions_detached_except_for_specific_agent = [matti if i == idx else matti.detach() for
                                                                      i, matti in enumerate(all_actions)]
                    all_actions_detached_except_for_specific_agent = torch.cat(
                        all_actions_detached_except_for_specific_agent, dim=1).to(device)  # P x 4 tensor
                    actor_loss = -ag.critic_local(
                        torch.cat((states, all_actions_detached_except_for_specific_agent), dim=1)).mean()
                    actor_loss.backward()
                    ag.actor_optimizer.step()

                    # Update target models using updated local models
                    ag.soft_update()

        # Return score
        score = np.max(self.episode_score)
        self._terminal_reset()
        return score




