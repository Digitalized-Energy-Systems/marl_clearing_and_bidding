""" Idea: Use DDPG with multiple actors and critics. Multiple critic provide an
improved gradient for the actors (see double q learning). Multiple actors
improve exploration and can be combined with standard ensemble learning
techniques. Further, the existence of multiple parallel models allows to use
multiple sets of hyperparameters, e.g. different learning rates.

"""

import copy

import gym
import numpy as np
import torch

from ..agent import DrlAgent, TargetNetMixin
from ..networks import DDPGActorNet, DDPGCriticNet
from ..shared_code.processing import batch_to_tensors
from ..shared_code.memory import ReplayMemory
from ..shared_code.exploration import GaussianNoise


"""
TODOs:
- Important: Currently actor/critic trained on different data!


Done:
- Computation time is a big problem, because increases with n of Neural Nets
-> Train exactly same number of NNs as DDPG to allow for better comparison?! -> set train_how_many=1

"""


class EnsembleDdpg(DrlAgent, TargetNetMixin):
    def __init__(self, env, memory_size=100000,
                 gamma=0.99, batch_size=128, tau=0.001, start_train=2500,
                 actor_fc_dims=[512, 512], critic_fc_dims=[512, 512],
                 actor_learning_rate=0.0002, critic_learning_rate=0.001,
                 noise_std_dev=0.1, optimizer='Adam', activation='tanh',
                 grad_clip=None, n_critics=2, n_actors=5, train_how_many=2,
                 *args, **kwargs):
        self.start_train = max(start_train, batch_size)

        assert isinstance(env.action_space, gym.spaces.Box)

        if activation == 'tanh':  # Multiple ones possible???
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

        self.n_critics = n_critics
        self.n_actors = n_actors

        # Generally: Different hyperparams make sense only, if there is a method to test which actor is the best
        self.tau = tau  # Multiple ones possible??? -> would result in some targets to be slower. Effects unclear
        # Multiple ones possible??? Not really, because actor/critic get trained together
        self.batch_size = batch_size
        self.batch_idxs = np.arange(
            self.batch_size, dtype=np.int32)   # Move to superclass?
        self.grad_clip = grad_clip
        self.train_how_many = train_how_many

        try:
            self.n_rewards = len(env.reward_space.low)
        except AttributeError:
            self.n_rewards = 1

        if isinstance(actor_learning_rate, float):
            actor_learning_rate = [actor_learning_rate] * n_actors
        if isinstance(critic_learning_rate, float):
            critic_learning_rate = [critic_learning_rate] * n_critics

        self._init_networks(actor_fc_dims, actor_learning_rate,
                            critic_fc_dims, critic_learning_rate, optimizer, activation)

        # TODO: How to do for multiple nets?
        self.device = self.critics[0].device
        self._init_memory(memory_size)
        self._init_noise(noise_std_dev)

    def _init_networks(self, actor_fc_dims, actor_learning_rate,
                       critic_fc_dims, critic_learning_rate, optimizer, activation):
        self.actors = [DDPGActorNet(
            self.n_obs, actor_fc_dims, self.n_act, actor_learning_rate[i],
            optimizer=optimizer, output_activation=activation)
            for i in range(self.n_actors)]
        # TODO: Use deep copy instead copy.deepcopy(self.actor)
        self.actor_targets = [copy.deepcopy(a) for a in self.actors]
        # TODO: variable optimizer for Critics as well
        self.critics = [DDPGCriticNet(
            self.n_obs, self.n_act, critic_learning_rate[i], critic_fc_dims,
            n_rewards=self.n_rewards)
            for i in range(self.n_critics)]
        self.critic_targets = [copy.deepcopy(c) for c in self.critics]

    def _init_memory(self, memory_size: int):
        """ Same as normal DDPG. """
        self.memory = ReplayMemory(
            memory_size, self.n_obs, self.n_act, n_rewards=self.n_rewards)

    def _init_noise(self, std_dev):
        """ Same as normal DDPG. """
        self.noise = GaussianNoise((self.n_act,), std_dev)

    @torch.no_grad()
    def act(self, obs, test=False):
        """ Use actor to create actions and add noise for exploration. """
        if self.memory.memory_counter < self.start_train:
            act = self.env.action_space.sample()
            return act
        if test is True:
            return self.test_act(obs)

        # Choose random actor to improve exploration
        # TODO: Choose best one or with some ensemble algo?
        actor = np.random.choice(self.actors)
        action = actor.forward(torch.tensor(
            obs, dtype=torch.float).to(self.device)).cpu().numpy()
        action += self.noise()
        return np.clip(action, self.min_range, 1)

    @torch.no_grad()
    def test_act(self, obs):
        # TODO: Better way than simply to use first one?
        # 1. Use the one that has proven it's the best (how?)
        # 2. Use critics to test, which actor proposes (expected) best action (similar to Q-learning! it's argmax! funny! )
        # 3. Use randomly (all of them should be equally good, but randomness is sometimes helpful)
        # 4. Use average of all
        # 5. Use randomly weighted average of all (combination of 3. and 4., also great for exploration)
        # 6. As 5. for weight for every combination of actor/action -> n x m random weights (eg actor A decides action 1, but actor B decides action 2 -> even more exploration. Funny: Similar to recombination in GAs)
        # 7. Maybe learn some meta-actor in the end, whcih learns to recombine all the actor's capabilities
        return self.actors[0].forward(torch.tensor(obs, dtype=torch.float).to(self.device)).cpu().numpy()

    def learn(self, obs, act, reward, next_obs, done,
              state=None, next_state=None, env_idx=0):
        """ Same as normal DDPG """
        # TODO: Consider states as well!
        self.remember(obs, act, reward, next_obs, done)

        if len(self.memory) < self.start_train:
            return

        obss, acts, rewards, next_obss, dones = batch
        self._learn(obss, acts, rewards, next_obss, dones)

    def _learn(self, obss, acts, rewards, next_obss, dones):
        # Randomly assign actors to critics
        # The actors get trained with the same data their current critic was trained before
        n_train = min(self.n_actors, self.n_critics, self.train_how_many)
        actor_critic_pairs = zip(
            # TODO: test which one works better
            # Randint allow for double usage of the same network -> Some nets get trained multiple times per step
            # np.random.randint(self.n_actors, size=n_train),
            # np.random.randint(self.n_critics, size=n_train))
            np.random.choice(self.n_actors, size=n_train),
            np.random.choice(self.n_critics, size=n_train))
        # TODO: Test what happens when we choose only one of each (more similar to DDPG)

        for actor_idx, critic_idx in actor_critic_pairs:
            batch = self.memory.sample_random_batch(self.batch_size)
            batch = batch_to_tensors(batch, self.device, continuous=True)
            obss, acts, rewards, next_obss, dones = batch

            self._train_critic(critic_idx, actor_idx, obss, acts,
                               rewards, next_obss, dones)
            # TODO: Löschen!
            batch = self.memory.sample_random_batch(self.batch_size)
            batch = batch_to_tensors(batch, self.device, continuous=True)
            obss, acts, rewards, next_obss, dones = batch
            #
            self._train_actor(critic_idx, actor_idx, obss,
                              acts, rewards, next_obss, dones)

        for idx, actor in enumerate(self.actors):
            self._soft_target_update(actor, self.actor_targets[idx], self.tau)
        for idx, critic in enumerate(self.critics):
            self._soft_target_update(
                critic, self.critic_targets[idx], self.tau)

    def remember(self, obs, action, reward, next_obs, done):
        """ Same as normal DDPG """
        # TODO: Maybe create a mixin for replay buffer based algos
        self.memory.store_transition(obs, action, reward, next_obs, done)

    def _train_critic(self, critic_idx, actor_idx, obss, acts, rewards, next_obss, dones):
        critic = self.critics[critic_idx]
        critic.optimizer.zero_grad()
        q_values = critic(obss, acts)
        targets = self._compute_targets(
            critic_idx, actor_idx, next_obss, dones, rewards)
        critic_loss = critic.loss(targets, q_values)
        critic_loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(critic.parameters(), self.grad_clip)
        critic.optimizer.step()
        return targets

    def _train_actor(self, critic_idx, actor_idx, obss, acts, rewards, next_obss, dones):
        actor = self.actors[actor_idx]
        actor.optimizer.zero_grad()
        # TODO: Choose one randomly, or use all of them (mean), or use double DQN stuff?
        critic = self.critics[critic_idx]
        actor_loss = -critic(obss, actor(obss)).sum(axis=1).mean()
        actor_loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(actor.parameters(), self.grad_clip)
        actor.optimizer.step()

    @ torch.no_grad()
    def _compute_targets(self, critic_idx, actor_idx, next_obss, dones, rewards):
        # TODO: better strategy than random sampling?
        actor_target = self.actor_targets[actor_idx]
        next_acts = actor_target(next_obss)
        target_values = self.critic_targets[critic_idx](next_obss, next_acts)
        target_values[dones == 1.0] = 0.0
        return rewards + (self.gamma * target_values)
