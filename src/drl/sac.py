import copy

import gym
import numpy as np
import torch

from .agent import DrlAgent, TargetNetMixin
from .ddpg import Ddpg
from .networks import SACActorNet, DDPGCriticNet
from .shared_code.processing import batch_to_tensors
from .shared_code.memory import (ReplayMemory, PrioritizedReplayMemory)
# TODO: Add Prio replay buffer to SAC as optional
from .shared_code.exploration import GaussianNoise


class Sac(Ddpg):
    def __init__(self, env, memory_size=100000,
                 gamma=0.99, batch_size=128, tau=0.001, start_train=3000,
                 actor_fc_dims=(512,), critic_fc_dims=(512,),
                 actor_learning_rate=0.0002, critic_learning_rate=0.0005,
                 optimizer='Adam', fixed_alpha=None,
                 grad_clip=None, layer_norm=True, *args, **kwargs):
        self.start_train = max(start_train, batch_size)

        # TODO: Discrete actions?!
        assert isinstance(env.action_space, gym.spaces.Box)
        # Actor only outputs tanh action space [-1, 1] -> rescale
        # Also clips to action space
        env = gym.wrappers.RescaleAction(env, -1, 1)
        # The wrapper destroys seeding of the action space -> re-seed
        env.action_space.seed(kwargs['seed'])

        super(Ddpg, self).__init__(env, gamma, *args, **kwargs)

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

        self._init_networks(list(actor_fc_dims), actor_learning_rate,
                            list(critic_fc_dims), critic_learning_rate,
                            optimizer, layer_norm)

        self.device = self.actor.device  # TODO: How to do for multiple nets?
        self._init_memory(memory_size)

        self.fixed_alpha = fixed_alpha
        if self.fixed_alpha:
            self.alpha = fixed_alpha
        else:
            # Heuristic from SAC paper, TODO: Maybe allow user to set this?!
            self.target_entropy = -np.prod(
                self.env.action_space.shape).item()
            self.log_alpha = torch.zeros(1, requires_grad=True).to(self.device)
            self.alpha_optimizer = torch.optim.Adam(
                [self.log_alpha], lr=actor_learning_rate)
            self.alpha = self.log_alpha.exp()

    def _init_networks(self, actor_fc_dims, actor_learning_rate,
                       critic_fc_dims, critic_learning_rate, optimizer, layer_norm):
        self.actor = SACActorNet(
            self.n_obs, actor_fc_dims, self.n_act, actor_learning_rate,
            optimizer=optimizer, output_activation='tanh', layer_norm=layer_norm)
        # self.actor_target = copy.deepcopy(self.actor)
        # TODO: variable optimizer for Critics as well
        self.critic1 = DDPGCriticNet(
            self.n_obs, self.n_act, critic_learning_rate, critic_fc_dims,
            n_rewards=self.n_rewards, layer_norm=layer_norm)
        self.critic1_target = copy.deepcopy(self.critic1)

        self.critic2 = DDPGCriticNet(
            self.n_obs, self.n_act, critic_learning_rate, critic_fc_dims,
            n_rewards=self.n_rewards, layer_norm=layer_norm)
        self.critic2_target = copy.deepcopy(self.critic2)

    @ torch.no_grad()
    def act(self, obs):
        """ Use actor to create actions and add noise for exploration. """
        if self.memory.memory_counter < self.start_train:
            return self.env.action_space.sample()

        obs = torch.tensor(obs, dtype=torch.float).to(self.device)
        return self.actor.forward(obs, act_only=True).cpu().numpy()

    @ torch.no_grad()
    def test_act(self, obs):
        obs = torch.tensor(obs, dtype=torch.float).to(self.device)
        # Return only the mean, deterministic action for testing
        return self.actor.forward(obs, act_only=True, deterministic=True).cpu().numpy()

    def _learn(self, obss, acts, rewards, next_obss, dones):
        self._train_critic(obss, acts, rewards, next_obss, dones)

        self._train_actor(obss, acts, rewards, next_obss, dones)

        if not self.fixed_alpha:
            self._update_alpha(obss)

        # self._soft_target_update(self.actor, self.actor_target, self.tau)
        self._soft_target_update(self.critic1, self.critic1_target, self.tau)
        self._soft_target_update(self.critic2, self.critic2_target, self.tau)

    def _train_critic(self, obss, acts, rewards, next_obss, dones):
        targets = self._compute_targets(next_obss, dones, rewards)
        for critic in (self.critic1, self.critic2):
            critic.optimizer.zero_grad()
            q_values = critic(obss, acts)
            critic_loss = critic.loss(targets, q_values)
            critic_loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    critic.parameters(), self.grad_clip)
            critic.optimizer.step()

    def _train_actor(self, obss, acts, rewards, next_obss, dones):
        self.actor.optimizer.zero_grad()
        log_prob, acts = self.actor.forward(obss)
        q_values = torch.minimum(self.critic1(obss, acts).sum(axis=1),
                                 self.critic2(obss, acts).sum(axis=1))
        actor_loss = (-q_values + self.alpha * log_prob).mean()
        actor_loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.grad_clip)
        self.actor.optimizer.step()

    def _update_alpha(self, obss):
        log_prob, _ = self.actor.forward(obss)
        self.alpha_optimizer.zero_grad()
        alpha_loss = -(self.log_alpha * (
            log_prob + self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()
        if self.step % 200 == 0:
            print('alpha: ', self.alpha)

    @ torch.no_grad()
    def _compute_targets(self, next_obss, dones, rewards):
        log_prob, next_acts = self.actor.forward(next_obss)
        target_values1 = self.critic1_target(next_obss, next_acts)
        target_values2 = self.critic2_target(next_obss, next_acts)
        target_values = torch.minimum(target_values1, target_values2)

        target_values = (self.gamma * target_values) - self.alpha * log_prob
        target_values[dones == 1.0] = 0.0

        return rewards + target_values
