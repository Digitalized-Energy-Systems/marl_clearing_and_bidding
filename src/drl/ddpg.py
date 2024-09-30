
import copy

import gym
import numpy as np
import torch

from .agent import DrlAgent, TargetNetMixin
from .networks import DDPGActorNet, DDPGCriticNet
from .shared_code.processing import batch_to_tensors
from .shared_code.memory import (ReplayMemory, PrioritizedReplayMemory)
# TODO: Add Prio replay buffer to DDPG as optional
from .shared_code.exploration import GaussianNoise


class Ddpg(DrlAgent, TargetNetMixin):
    def __init__(self, env, memory_size=100000,
                 gamma=0.99, batch_size=128, tau=0.001, start_train=1000,
                 actor_fc_dims=[500, 500], critic_fc_dims=[500, 500],
                 actor_learning_rate=0.0002, critic_learning_rate=0.001,
                 noise_std_dev=0.1, optimizer='Adam', activation='tanh',
                 grad_clip=None, layer_norm=True, *args, **kwargs):
        self.start_train = max(start_train, batch_size)

        assert isinstance(env.action_space, gym.spaces.Box)
        if activation == 'tanh':
            # Actor only outputs tanh action space [-1, 1]
            # Also clips to action space
            env = gym.wrappers.RescaleAction(env, -1, 1)
            self.min_range = -1
        elif activation == 'sigmoid':
            env = gym.wrappers.RescaleAction(env, 0, 1)
            self.min_range = 0
        # The wrapper destroys seeding of the action space -> re-seed
        env.action_space.seed(kwargs['seed'])

        super().__init__(env, gamma, *args, **kwargs)

        self.tau = tau
        self.update_counter = 0
        self.batch_size = batch_size  # Move to superclass?
        self.batch_idxs = np.arange(
            self.batch_size, dtype=np.int32)   # Move to superclass?
        self.grad_clip = grad_clip

        try:
            self.n_rewards = len(env.reward_space.low)
        except AttributeError:
            self.n_rewards = 1

        self._init_networks(actor_fc_dims, actor_learning_rate,
                            critic_fc_dims, critic_learning_rate, optimizer, activation, layer_norm)

        self.device = self.actor.device  # TODO: How to do for multiple nets?
        self._init_memory(memory_size)
        self._init_noise(noise_std_dev)

    def _init_networks(self, actor_fc_dims, actor_learning_rate,
                       critic_fc_dims, critic_learning_rate, optimizer,
                       activation, layer_norm):
        self.actor = DDPGActorNet(
            self.n_obs, actor_fc_dims, self.n_act, actor_learning_rate,
            optimizer=optimizer, output_activation=activation, layer_norm=layer_norm)
        self.actor_target = copy.deepcopy(self.actor)
        # TODO: variable optimizer for Critics as well
        self.critic = DDPGCriticNet(
            self.n_obs, self.n_act, critic_learning_rate, critic_fc_dims,
            n_rewards=self.n_rewards, layer_norm=layer_norm)
        self.critic_target = copy.deepcopy(self.critic)

    def _init_memory(self, memory_size: int):
        self.memory = ReplayMemory(
            memory_size, self.n_obs, self.n_act, n_rewards=self.n_rewards)

    def _init_noise(self, std_dev):
        self.noise = GaussianNoise((self.n_act,), std_dev)

    def act(self, obs):
        """ Use actor to create actions and add noise for exploration. """
        if self.memory.memory_counter < self.start_train:
            act = self.env.action_space.sample()
            return act
        action = self.test_act(obs)
        action += self.noise()
        return np.clip(action, self.min_range, 1)

    @torch.no_grad()
    def test_act(self, obs):
        return self.actor.forward(torch.tensor(obs, dtype=torch.float).to(self.device)).cpu().numpy()

    def learn(self, obs, act, reward, next_obs, done,
              state=None, next_state=None, env_idx=0):
        # TODO: Consider states as well!
        self.remember(obs, act, reward, next_obs, done)

        if len(self.memory) < self.start_train:
            return

        batch = self.memory.sample_random_batch(self.batch_size)
        batch = batch_to_tensors(batch, self.device, continuous=True)
        obss, acts, rewards, next_obss, dones = batch

        self._learn(obss, acts, rewards, next_obss, dones)

    def remember(self, obs, action, reward, next_obs, done):
        # TODO: Maybe create a mixin for replay buffer based algos
        self.memory.store_transition(obs, action, reward, next_obs, done)

    def _learn(self, obss, acts, rewards, next_obss, dones):
        self._train_critic(obss, acts, rewards, next_obss, dones)

        self._train_actor(obss, acts, rewards, next_obss, dones)

        self._soft_target_update(self.actor, self.actor_target, self.tau)
        self._soft_target_update(self.critic, self.critic_target, self.tau)

    def _train_critic(self, obss, acts, rewards, next_obss, dones):
        self.critic.optimizer.zero_grad()
        q_values = self.critic(obss, acts)
        targets = self._compute_targets(next_obss, dones, rewards)
        critic_loss = self.critic.loss(targets, q_values)
        critic_loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.critic.parameters(), self.grad_clip)
        self.critic.optimizer.step()
        return targets

    def _train_actor(self, obss, acts, rewards, next_obss, dones):
        self.actor.optimizer.zero_grad()
        # If there are multiple rewards: Maximize sum of them
        actor_loss = -self.critic(obss, self.actor(obss)).sum(axis=1).mean()
        actor_loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.grad_clip)
        self.actor.optimizer.step()

    @torch.no_grad()
    def _compute_targets(self, next_obss, dones, rewards):
        next_acts = self.actor_target(next_obss)
        target_values = self.critic_target(next_obss, next_acts)
        target_values[dones == 1.0] = 0.0
        return rewards + (self.gamma * target_values)


class Td3(Ddpg):
    def __init__(self, env, update_delay=2, target_noise_std_dev=0.2,
                 noise_clip=0.5, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.update_delay = update_delay
        # self.target_action_noise = GaussianNoise((self.n_act,), target_noise_std_dev)
        self.target_noise_std_dev = target_noise_std_dev
        self.noise_clip = noise_clip

    def _init_networks(self, actor_fc_dims, actor_learning_rate,
                       critic_fc_dims, critic_learning_rate, optimizer, activation, layer_norm):
        super()._init_networks(actor_fc_dims, actor_learning_rate,
                               critic_fc_dims, critic_learning_rate, optimizer, activation, layer_norm)
        self.critic2 = DDPGCriticNet(
            self.n_obs, self.n_act, critic_learning_rate, critic_fc_dims,
            n_rewards=self.n_rewards, layer_norm=layer_norm)
        self.critic2_target = copy.deepcopy(self.critic2)

    def _learn(self, obss, acts, rewards, next_obss, dones):
        self._train_critic(obss, acts, rewards, next_obss, dones)

        if self.step % self.update_delay == 0:
            # TODO: Why is it okay to only use critic1 here?!
            self._train_actor(obss, acts, rewards, next_obss, dones)
            # Update all (!) targets delayed
            self._soft_target_update(self.actor, self.actor_target, self.tau)
            self._soft_target_update(self.critic, self.critic_target, self.tau)
            self._soft_target_update(
                self.critic2, self.critic2_target, self.tau)

    def _train_critic(self, obss, acts, rewards, next_obss, dones):
        targets = super()._train_critic(obss, acts, rewards, next_obss, dones)
        self.critic2.optimizer.zero_grad()
        q_values = self.critic2(obss, acts)
        critic_loss = self.critic2.loss(targets, q_values)
        critic_loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.critic2.parameters(), self.grad_clip)
        self.critic2.optimizer.step()

    @torch.no_grad()
    def _compute_targets(self, next_obss, dones, rewards):
        """ Add noise to target actions and use min q value from 2 critics. """
        next_acts = self.actor_target(next_obss)
        noise = (
            torch.randn_like(next_acts) * self.target_noise_std_dev
        ).clamp(-self.noise_clip, self.noise_clip)
        next_acts = (next_acts + noise).clamp(self.min_range, 1)

        target_values1 = self.critic_target(next_obss, next_acts)
        target_values2 = self.critic2_target(next_obss, next_acts)
        target_values = torch.minimum(target_values1, target_values2)
        target_values[dones == 1.0] = 0.0
        return rewards + (self.gamma * target_values)


class Ddpg1Step(Ddpg):
    """ Lots of classic optimization problems can be formulated as 1-step
    RL problems, which allows some simplifications in the DRL algos. For
    example no target networks are required and targets=rewards. """

    # def act(self, obs):
    #     """ We (probably) do not need noise. TODO Test this! """
    #     if self.memory.memory_counter < self.start_train:
    #         return self.env.action_space.sample()
    #     return self.test_act(obs)

    def _learn(self, obss, acts, rewards, next_obss, dones):
        """ No target updates needed. """
        self._train_critic(obss, acts, rewards, next_obss, dones)
        self._train_actor(obss, acts, rewards, next_obss, dones)

    def _compute_targets(self, next_obss, dones, rewards):
        """ Targets equal the reward in this setting. """
        return rewards
