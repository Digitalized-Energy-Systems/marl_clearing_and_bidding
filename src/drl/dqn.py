"""
DQN algorithms with diverse extensions like noisy DQN, double DQN,
n-step DQN, etc.

"""

import pdb

import numpy as np
import torch

from .agent import DrlAgent, TargetNetMixin
from .shared_code.exploration import EpsilonGreedy
from .networks import DqNet, NoisyDqNet
from .shared_code.memory import (
    ReplayMemory, PrioritizedReplayMemory, NStepBuffer)
from .shared_code.processing import batch_to_tensors


class Dqn(DrlAgent, TargetNetMixin):
    def __init__(self, env, memory_size=100000,
                 gamma=0.99, batch_size=128, tau=0.001, target_update_freq=1,
                 start_train=5000, fc_dims=[512, 512], learning_rate=0.0002,
                 *args, **kwargs):
        self.start_train = max(start_train, batch_size)
        super().__init__(env, gamma, *args, **kwargs)

        self.tau = tau
        self.target_update_freq = target_update_freq
        self.update_counter = 0
        self.batch_size = batch_size  # * self.n_envs regarding to Lapan
        self.batch_idxs = np.arange(
            self.batch_size, dtype=np.int32)   # Move to superclass?

        self.eps_greedy = EpsilonGreedy(
            epsilon_min=0.0, epsilon_end_step=0.5)

        self._init_networks(fc_dims, learning_rate)
        self._hard_target_update(self.net, self.target_net)
        self.device = self.net.device
        self._init_memory(memory_size)

    def _init_memory(self, memory_size: int):
        self.memory = ReplayMemory(memory_size, self.n_obs, 1)

    def _init_networks(self, fc_dims, learning_rate):
        self.net = DqNet(self.n_obs, fc_dims, self.n_act, learning_rate)
        self.target_net = DqNet(
            self.n_obs, fc_dims, self.n_act, learning_rate)

    def act(self, obs):
        """ Choose action with the highest q value under exploration. """
        if self.memory.memory_counter < self.start_train:
            return self.env.action_space.sample()
        return self.eps_greedy.explore(self.net.n_act, self.test_act, obs)

    @torch.no_grad()
    def test_act(self, obs):
        """ Choose action without noise or other exploration. """
        q_values = self.net.forward(torch.tensor(obs, dtype=torch.float))
        return q_values.cpu().detach().numpy().argmax()

    def remember(self, obs, action, reward, next_obs, done, env_idx=0):
        self.memory.store_transition(obs, action, reward, next_obs, done)

    def learn(self, obs, act, reward, next_obs, done, state=None, next_state=None, env_idx=0):
        self.remember(obs, act, reward, next_obs, done, env_idx)

        if self.memory.memory_counter < self.start_train:
            return None

        self._learn()  # TODO: maybe choose different name
        self._update()

    def _learn(self):
        batch = self.memory.sample_random_batch(self.batch_size)
        obss, acts, rewards, next_obss, dones = batch_to_tensors(
            batch, self.device, continuous=False)

        values = self.net.forward(obss)[self.batch_idxs, acts.flatten()]

        # TODO: Currently only works for scalar rewards
        targets = self._compute_targets(next_obss, dones, rewards.flatten())

        self.net.optimizer.zero_grad()
        loss = self.net.loss(targets, values).to(self.device)
        loss.backward()
        self.net.optimizer.step()

    def _compute_targets(self, next_obss, dones, rewards):
        target_values = self._compute_target_values(next_obss)
        target_values[dones == 1.0] = 0.0
        return rewards + self.gamma * target_values.detach()

    @torch.no_grad()
    def _compute_target_values(self, next_obss):

        return self.target_net.forward(next_obss).max(axis=1)[0]

    def _update(self):
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self._soft_target_update(self.net, self.target_net, self.tau)
            self.update_counter = 0

        self.eps_greedy.update_epsilon(self.n_train_steps)


class NStepDqn(Dqn):
    """ Condense multiple samples to achieve longer transition sequences.
    Speeds up training.
    https://link.springer.com/content/pdf/10.1007/BF00115009.pdf
    """

    def __init__(self, env, n_steps: int=4, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.n_steps = n_steps
        self.n_step_buffer = [NStepBuffer(n_steps, self.gamma)
                              for _ in range(self.n_envs)]
        self.gamma = self.gamma**self.n_steps

    def remember(self, obs, action, reward, next_obs, done, env_idx):
        squashed_transitions = self.n_step_buffer[env_idx].append_and_squash(
            obs, action, reward, next_obs, done)
        # IDEA: Would it be preverable to alter the replay memory instead, so that arbitrary sequences can be drawn from it? The current way destroys the original data, which could be bad, if we need it at some point.
        # IDEA: Is n=const the best? Maybe increase/decrease over time? (probaby decrease to the end to reduce errors)
        if squashed_transitions is not None:
            for transition in squashed_transitions:
                super().remember(*transition)


class DoubleDqn(Dqn):
    @torch.no_grad()
    def _compute_target_values(self, next_obss):
        """ Use actual actions from q net instead of best actions from
        target net.
        Prevents overestimation of q values.
        Hasselt et al.: https://arxiv.org/pdf/1509.06461.pdf
        """
        next_acts = self.net.forward(next_obss).max(axis=1)[1].unsqueeze(-1)
        target_values = self.target_net.forward(next_obss)
        return target_values.gather(1, next_acts).squeeze(-1)


class NoisyDqn(Dqn):
    # TODO: Remove/reduce epsilon greedy exploration? (but would worsen results)
    def _init_networks(self, fc_dims, learning_rate):
        """ Initialize with noisy networks instead of normal nets. """
        self.net = NoisyDqNet(self.n_obs, fc_dims, self.n_act, learning_rate)
        self.target_net = NoisyDqNet(
            self.n_obs, fc_dims, self.n_act, learning_rate)


class PrioReplayDqn(NStepDqn):
    def __init__(self, env, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        assert isinstance(self.net.loss, torch.nn.MSELoss)

    def _init_memory(self, memory_size: int):
        """ Replace the ReplayMemory with a Prioritized one. """
        self.memory = PrioritizedReplayMemory(memory_size, self.n_obs, 1)

    def _learn(self):
        batch, memory_idxs, batch_weights = self.memory.sample_random_batch(
            self.batch_size)
        batch_weights = torch.tensor(batch_weights).to(self.device)
        obss, acts, rewards, next_obss, dones = batch_to_tensors(
            batch, self.device, continuous=False)

        values = self.net.forward(obss)[self.batch_idxs, acts.flatten()]
        targets = self._compute_targets(next_obss, dones, rewards)

        self.net.optimizer.zero_grad()
        td_error = (values - targets)
        mse_loss = (td_error**2 * batch_weights).mean().to(self.device)
        mse_loss.backward()
        self.net.optimizer.step()

        td_error = td_error.detach().cpu().abs().numpy()
        self.memory.update_priorities(td_error, memory_idxs)


class DuelingDqn(Dqn):
    pass


def compose_DQN_class(
        n_step=True, noisy=False, double=True, prioritized=True,
        dueling=False):
    inherit_from = []
    if n_step is True:
        inherit_from.append(NStepDqn)
    if noisy is True:
        inherit_from.append(NoisyDqn)
    if double is True and dueling is True:
        raise NotImplementedError()
    elif double is True:
        inherit_from.append(DoubleDqn)
    elif dueling is True:
        raise NotImplementedError()
    if prioritized is True:
        inherit_from.append(PrioReplayDqn)

    if not inherit_from:
        return Dqn

    class ComposedDqn(*inherit_from):
        pass

    return ComposedDqn


class RainbowDqn(NStepDqn, DoubleDqn):
    # How to make choosable which to inherit from? eg: `double = True`, `N_step=False`
    # How to deal with methods changed from two parents?
    pass


def main():
    pass
    # TODO: Run minimal experiment with DQN


if __name__ == '__main__':
    main()
