"""
Implementation of the REINFORCE algorithm.

TODO: This whole implementation seems to be far from perfect. Also maybe explore some other baseline (learned baseline, moving average) and also check discount needs to be considered in baseline

TODO: Where exactly is the difference between reinforce and vanilla policy gradient? Which of my implementations is which?

"""

import math

import numpy as np
from gym.spaces import Discrete, Box
from gym.wrappers import RescaleAction
import torch
from torch.nn.functional import softmax, log_softmax

from .agent import DrlAgent
from .networks import ReinforceNet, ContinuousA2CNet, DiscreteA2CNet
from .shared_code.memory import NStepBuffer, Episode


# TODO: Currently only works for discrete environments
class BaseReinforce(DrlAgent):
    def __init__(self, env, gamma=0.99, episodes_per_update=3,
                 fc_dims=[256, 256], learning_rate=0.0001, *args, **kwargs):
        super().__init__(env, gamma, *args, **kwargs)
        self.episodes_per_update = episodes_per_update

        # assert discrete env

        self._init_net(fc_dims, learning_rate)
        self.device = self.net.device
        self._init_buffer()

    def _init_net(self, fc_dims, learning_rate):
        self.net = ReinforceNet(self.n_obs, fc_dims, self.n_act, learning_rate)

    def _init_buffer(self):
        self.q_values = []
        self.obs_memory = []
        self.action_memory = []
        self.episode_counter = 0
        self.episodes = [Episode(self.gamma) for _ in range(self.n_envs)]

    @torch.no_grad()
    def act(self, obs):
        obs = torch.tensor(obs, dtype=torch.float)
        action_probs = softmax(self.net.forward(obs))
        return np.random.choice(range(self.n_act), p=action_probs.numpy())

    def test_act(self, obs):
        return self.act(obs)

    def learn(self, obs, act, reward, next_obs, done, state=None,
              next_state=None, env_idx=0):
        self.episodes[env_idx].step(obs, act, reward, next_obs, done)

        if not done:
            return

        # Full episode rollout collected
        self.episode_counter += 1
        self.q_values.extend(self.episodes[env_idx].q_values)
        self.action_memory.extend(self.episodes[env_idx].actions)
        self.obs_memory.extend(self.episodes[env_idx].obss)
        self.episodes[env_idx] = Episode(self.gamma)

        if not self.episode_counter >= self.episodes_per_update:
            return

        self._learn()
        self._init_buffer()

    def _learn(self):
        obss, chosen_actions, q_values = self._batch_to_tensors()

        self.net.optimizer.zero_grad()
        action_logits = self.net.forward(obss)
        self._train_net(action_logits, chosen_actions, q_values)

    def _batch_to_tensors(self):
        chosen_actions = torch.LongTensor(self.action_memory)
        q_values = torch.FloatTensor(self.q_values)
        obss = torch.FloatTensor(self.obs_memory)
        return obss, chosen_actions, q_values

    def _train_net(self, action_logits, chosen_actions, q_values):
        loss = self._compute_loss(action_logits, chosen_actions, q_values)
        loss.backward()
        self.net.optimizer.step()

    def _compute_loss(self, action_logits, chosen_actions, q_values):
        log_probs = log_softmax(action_logits, dim=1)
        batch_idxs = np.arange(len(chosen_actions), dtype=np.int32)
        action_log_probs = log_probs[batch_idxs, chosen_actions]

        # TODO: is this correct for the subclasses?
        loss = -(q_values * action_log_probs).mean()
        return loss


class Reinforce(BaseReinforce):
    """ Add several extensions to base reinforce, namely entropy for
    exploration, a simply baseline to reduce variance, and n-step unrolling
    so that not full episodes are required. However, multiple envs and low
    gamma values are helpful or maybe even necessary. """

    def __init__(self, env, baseline=False, n_steps=25,
                 entropy_beta=0.02, batch_size=64, *args, **kwargs):
        self.n_steps = n_steps
        # TODO: should depend on gamma as default
        super().__init__(env, *args, **kwargs)
        self.base = baseline
        self.batch_size = batch_size
        self.entropy_beta = entropy_beta
        self.total_reward = 0

    def _init_buffer(self):
        super()._init_buffer()
        self.env_idxs = []
        self.n_step_buffer = [NStepBuffer(self.n_steps, self.gamma)
                              for _ in range(self.n_envs)]

    def learn(self, obs, act, reward, next_obs, done,
              state=None, next_state=None, env_idx=0):
        # TODO: Is this long new method really required?
        # Do n-step unrolling of the transitions to estimate q value without # running full episodes
        transitions = self.n_step_buffer[env_idx].append_and_squash(
            obs, act, reward, next_obs, done)

        if transitions is None:
            return

        for (obs, action, discounted_reward, _, _) in transitions:
            self.total_reward += discounted_reward
            baseline = self.total_reward / (self.n_steps + 1)
            if self.base:
                self.q_values.append(discounted_reward - baseline)
            else:
                self.q_values.append(discounted_reward)
            self.obs_memory.append(obs)
            self.action_memory.append(action)

        if len(self.q_values) >= self.batch_size:
            self._learn()
            # Re-init all buffers since old data is now outdated (on-policy)
            self._init_buffer()

    def _compute_loss(self, action_logits, chosen_actions, q_values):
        loss = super()._compute_loss(action_logits, chosen_actions, q_values)
        return loss + self._entropy_loss(action_logits)

    def _entropy_loss(self, action_logits):
        action_probs = softmax(action_logits, dim=1)
        log_probs = log_softmax(action_logits, dim=1)
        entropy = -(action_probs * log_probs).sum(dim=1).mean()
        return -self.entropy_beta * entropy


class A2C(Reinforce):
    def __init__(self, env, entropy_beta=1e-4, *args, **kwargs):
        kwargs['baseline'] = False  # Critic as baseline is used anyway
        # TODO: can this be deleted?!

        if isinstance(env.action_space, Discrete):
            self._act = self.act_discrete
            self.continuous = False
        elif isinstance(env.action_space, Box):
            self._act = self.act_continuous
            self._test_act = self._test_act_continuous
            self._compute_loss = self._compute_continuous_loss
            self.continuous = True
        else:
            raise ValueError('Expected "Box" or "Discrete" action space')
        super().__init__(env, *args, **kwargs)

    def _init_net(self, fc_dims, learning_rate):
        if not self.continuous:
            self.net = DiscreteA2CNet(self.n_obs, self.n_act, learning_rate)
        else:
            self.net = ContinuousA2CNet(self.n_obs, self.n_act, learning_rate)
            self.env = RescaleAction(self.env, -1, 1)
            # The wrapper destroys seeding of the action space -> re-seed
            self.env.action_space.seed(kwargs['seed'])

    def act(self, obs):
        return self._act(torch.tensor(obs, dtype=torch.float).to(self.device))

    @torch.no_grad()
    def act_discrete(self, obs):
        action_probs = softmax(self.net.forward(obs))
        return np.random.choice(range(self.n_act), p=action_probs.numpy())

    @torch.no_grad()
    def act_continuous(self, obs):
        mu, var = self.net.forward(obs)
        sigma = torch.sqrt(var).numpy()
        actions = np.random.normal(mu.cpu().numpy(), sigma)
        return np.clip(actions, -1, 1)

    def test_act(self, obs):
        # TODO for continuous: make sure random seed is set! or use only mu?!
        return self._test_act(torch.tensor(obs, dtype=torch.float).to(self.device))

    @torch.no_grad()
    def _test_act_continuous(self, obs):
        """ Return just the mean actions without any noise for exploration. """
        return self.net.forward(obs)[0].cpu().numpy()

    def _learn(self):
        self.net.optimizer.zero_grad()
        obss, chosen_actions, self.q_values = self._batch_to_tensors()

        action_logits, self.expected_value = self.net.full_forward(obss)

        advantages = self.q_values - self.expected_value
        self._train_net(action_logits, chosen_actions, advantages)

    def _train_net(self, action_logits, chosen_actions, q_values):
        loss = self._compute_loss(action_logits, chosen_actions, q_values)
        loss.backward()
        self.net.optimizer.step()

    def _compute_loss(self, action_logits, chosen_actions, advantages):
        """ A2C loss = Reinforce loss + critic loss. """
        actor_loss = super()._compute_loss(
            action_logits, chosen_actions, advantages)
        critic_loss = self.net.critic_loss(
            self.q_values, self.expected_value.flatten())
        return actor_loss + critic_loss

    def _compute_continuous_loss(self, action_logits, chosen_actions, advantages):
        mu, var = action_logits
        critic_loss = self.net.critic_loss(
            self.q_values, self.expected_value.flatten())

        policy_loss = -(advantages * self._logprob(
            mu, var, chosen_actions)).mean()

        entropy_loss = self._continuous_entropy_loss(variance=action_logits[1])

        return critic_loss + policy_loss + entropy_loss

    def _logprob(self, mu, var, actions):
        p1 = -((mu - actions) ** 2) / (2 * var.clamp(min=1e-3))
        p2 = -torch.log(torch.sqrt(2 * math.pi * var))
        return p1 + p2

    def _continuous_entropy_loss(self, variance):
        entropy = -(torch.log(2 * math.pi * variance) + 1) / 2
        return self.entropy_beta * entropy.mean()
