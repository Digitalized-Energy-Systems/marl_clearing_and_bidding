import copy
import csv

import gym
import numpy as np
import torch
import torch.nn.functional as functional

from .agent import MarlAgent, TargetNetMixin, PettingZooActScaler
from .networks import DDPGActorNet, DDPGCriticNet
from .shared_code.memory import (
    ReplayMemory, MarlReplayMemory, PrioritizedReplayMemory, NStepBuffer)
from .shared_code.exploration import GaussianNoise


import pdb


class Maddpg(MarlAgent, TargetNetMixin):
    def __init__(self, env, memory_size=200000,
                 gamma=0.97, batch_size=128, tau=0.001, start_train=2000,
                 start_train_actors=0, actor_train_interval=1,
                 critic_train_interval=1,
                 actor_fc_dims=[128, 128], critic_fc_dims=[512, 512],
                 actor_learning_rate=0.0005, critic_learning_rate=0.001,
                 noise_std_dev=0.2, optimizer='Adam', *args, **kwargs):
        self.start_train = max(start_train, batch_size)
        self.start_train_actors = max(start_train_actors, self.start_train)
        self.actor_train_interval = actor_train_interval
        self.critic_train_interval = critic_train_interval

        # Actual output spaces of the neural nets (assumption: tanh activation)
        self.nn_action_spaces = {
            a_id: gym.spaces.Box(
                -np.ones(env.action_space(a_id).shape),
                np.ones(env.action_space(a_id).shape),
                seed=kwargs['seed'])
            for a_id in env.possible_agents}
        env = PettingZooActScaler(env, self.nn_action_spaces)

        super().__init__(env, gamma, *args, **kwargs)

        self._init_networks(actor_fc_dims, actor_learning_rate,
                            critic_learning_rate, critic_fc_dims, optimizer)
        # TODO: How to do for multiple nets?
        self.device = self.actors[self.a_ids[0]].device

        self.tau = tau
        self.update_counter = 0
        self.batch_size = batch_size  # Move to superclass?
        self.batch_idxs = np.arange(
            self.batch_size, dtype=np.int32)   # Move to superclass?

        self._init_memory(memory_size)
        self._init_noise(noise_std_dev)

    def _init_networks(self, actor_fc_dims, actor_learning_rate,
                       critic_learning_rate, critic_fc_dims, optimizer):
        """ Init actor and critic networks + target nets for all agents. """
        # TODO: Probably good idea to distribute networks over GPUs. How?
        # Create one actor per agent
        self.actors = {
            a_id: DDPGActorNet(self.env.observation_space(a_id).shape[0],
                               actor_fc_dims,
                               self.env.action_space(a_id).shape[0],
                               actor_learning_rate,
                               optimizer=optimizer)
            for a_id in self.a_ids}

        # Create one critic per agent
        assert hasattr(
            self, 'state_space'), 'TODO: Use obs spaces as alternative'
        self.critics = {
            a_id: DDPGCriticNet(self.state_space.shape[0], self.n_act,
                                critic_learning_rate, critic_fc_dims)
            for a_id in self.a_ids}

        self.actor_targets = {
            a_id: copy.deepcopy(a) for a_id, a in self.actors.items()}
        self.critic_targets = {
            a_id: copy.deepcopy(c) for a_id, c in self.critics.items()}

    def _init_memory(self, memory_size: int):
        self.memory = MarlReplayMemory(
            memory_size, self.a_ids,
            {a: self.env.observation_space(a) for a in self.a_ids},
            {a: self.env.action_space(a) for a in self.a_ids},
            self.env.state_space,
            n_rewards=self.n_agents)

    def _init_noise(self, std_dev):
        self.noise = {a: GaussianNoise(self.env.action_space(a).shape, std_dev)
                      for a in self.a_ids}

    def act(self, obs_dict):
        # TODO: Maybe move this exploration to main loop (always the same)
        if len(self.memory) < self.start_train_actors:
            # No training happened yet -> use random actions
            return {a_id: space.sample()
                    for a_id, space in self.nn_action_spaces.items()}

        return self.test_act(obs_dict, noisy=True)

    @ torch.no_grad()
    def test_act(self, obs_dict, noisy=False):
        act_dict = {a_id: self.single_act(a_id, local_obs, noisy)
                    for a_id, local_obs in obs_dict.items()}

        return act_dict

    @ torch.no_grad()
    def single_act(self, a_id, local_obs, noisy=False):
        """ A single agent gets its observation and choses an action. """
        local_obs = torch.tensor(local_obs, dtype=torch.float).to(self.device)
        # Reshape tensor to one dimension in case of 0-dim tensor
        act = self.actors[a_id](local_obs).cpu().numpy()
        if noisy:
            act = np.clip(act + self.noise[a_id](),
                          a_min=self.nn_action_spaces[a_id].low,
                          a_max=self.nn_action_spaces[a_id].high)

        return act

    def learn(self, obs, act, reward, next_obs, done, state=None,
              next_state=None, env_idx=0):
        self._remember(obs, act, reward, next_obs, done, state, next_state)

        if len(self.memory) < self.start_train:
            return

        for a_id, critic in self.critics.items():
            batch = self._get_batch()
            if self.step % self.critic_train_interval == 0:
                self._train_critic(a_id, critic, batch)
                critic_target = self.critic_targets[a_id]
                self._soft_target_update(critic, critic_target, self.tau)

            if self.step < self.start_train_actors:
                continue
            if self.step % self.actor_train_interval != 0:
                continue

            actor = self.actors[a_id]
            self._train_actor(a_id, actor, batch)
            actor_target = self.actor_targets[a_id]
            self._soft_target_update(actor, actor_target, self.tau)

    def _remember(self, obs, act, reward, next_obs, done, state, next_state):
        self.memory.store_transition(
            obs, state, act, reward, next_obs, next_state, done)

    def _train_critic(self, agent_id, critic, batch):
        obss, next_obss, acts, states, next_states, rewards, dones = batch

        critic.optimizer.zero_grad()
        concat_acts = self._torch_dict_to_full_tensor(acts)
        q_values = critic(states, concat_acts).flatten()

        concat_next_acts = self._get_acts(next_obss, target=True)
        target_q_values = self._compute_targets(
            concat_next_acts, next_states, dones, rewards, agent_id)
        critic_loss = critic.loss(target_q_values, q_values)
        critic_loss.backward(retain_graph=True)
        critic.optimizer.step()

    def _train_actor(self, agent_id, actor, batch):
        obss, next_obss, actions, states, next_states, rewards, dones = batch
        actor.optimizer.zero_grad()
        concat_acts = self._get_acts(obss, target=False)
        actor_loss = -self.critics[agent_id](states, concat_acts).mean()
        actor_loss.backward(retain_graph=True)
        actor.optimizer.step()

    def _get_batch(self):
        batch = self.memory.sample_random_batch(self.batch_size)
        obss, next_obss, actions, states, next_states, rewards, dones = batch
        obss = {key: torch.tensor(values, dtype=torch.float).to(self.device)
                for key, values in obss.items()}
        next_obss = {key: torch.tensor(values, dtype=torch.float).to(self.device)
                     for key, values in next_obss.items()}
        states = torch.tensor(states, dtype=torch.float).to(self.device)
        next_states = torch.tensor(
            next_states, dtype=torch.float).to(self.device)
        actions = {key: torch.tensor(values, dtype=torch.float).to(self.device)
                   for key, values in actions.items()}
        rewards = {key: torch.tensor(values, dtype=torch.float).to(self.device)
                   for key, values in rewards.items()}
        dones = {key: torch.tensor(values, dtype=torch.float).to(self.device)
                 for key, values in dones.items()}

        return obss, next_obss, actions, states, next_states, rewards, dones

    @ torch.no_grad()
    def _compute_targets(self, next_acts, next_states, dones, rewards, a_id):
        target_values = self.critic_targets[a_id](next_states, next_acts)
        target_values[dones[a_id] == 1.0] = 0.0
        return (rewards[a_id] + (self.gamma * target_values)).flatten()

    def _get_acts(self, obss, target: bool):
        """ Target actions as batch-sized tensor """
        actors = self.actor_targets if target else self.actors
        acts = {a_id: actor(obss[a_id]) for a_id, actor in actors.items()}

        return self._torch_dict_to_full_tensor(acts)

    def _torch_dict_to_full_tensor(self, dict_):
        return torch.cat([dict_[a_id] for a_id in self.a_ids], dim=1).to(self.device)


class MaddpgSingleCritic(Maddpg):
    """ Own implementation of MADDPG where all agents share their critic.
    A single critic is trained with m outputs, where m is the number of
    different rewards, i.e. the number of "teams".
    With n agents, only n+1 optimiziation are required instead of 2n, when all
    agents have their own critic. Therefore, training should be faster. In
    addition, weights can be shared. So maybe even better results?!
    """

    def _init_networks(self, actor_fc_dims, actor_learning_rate,
                       critic_learning_rate, critic_fc_dims, optimizer):
        """ Init actor networks and only a single (!) critic network + target
        nets for all agents. """
        # TODO: Probably good idea to distribute networks over GPUs. How?
        # Create one actor per agent
        self.actors = {
            a_id: DDPGActorNet(self.env.observation_space(a_id).shape[0],
                               actor_fc_dims,
                               self.env.action_space(a_id).shape[0],
                               actor_learning_rate,
                               optimizer=optimizer)
            for a_id in self.a_ids}

        # Create a single critic with one Q output for every reward
        assert hasattr(
            self, 'state_space'), 'TODO: Use obs spaces as alternative'
        self.critic = DDPGCriticNet(
            self.state_space.shape[0], self.n_act,
            critic_learning_rate, critic_fc_dims, n_rewards=len(self.a_ids))

        self.actor_targets = {
            a_id: copy.deepcopy(a) for a_id, a in self.actors.items()}
        self.critic_target = copy.deepcopy(self.critic)

    def learn(self, obs, act, reward, next_obs, done, state=None,
              next_state=None, env_idx=0):
        self._remember(obs, act, reward, next_obs, done, state, next_state)

        if len(self.memory) < self.start_train:
            return

        batch = self._get_batch()
        if self.step % self.critic_train_interval == 0:
            self._train_critic(agent_id=None, critic=self.critic, batch=batch)
            self._soft_target_update(self.critic, self.critic_target, self.tau)

        if not self.step >= self.start_train_actors:
            return

        if self.step % self.actor_train_interval == 0:
            for a_id, actor in self.actors.items():
                # Sample independent batches for the agents
                batch = self._get_batch()
                self._train_actor(a_id, actor, batch)
                actor_target = self.actor_targets[a_id]
                self._soft_target_update(actor, actor_target, self.tau)

    def _train_actor(self, agent_id, actor, batch):
        obss, next_obss, actions, states, next_states, rewards, dones = batch
        actor.optimizer.zero_grad()
        concat_acts = self._get_acts(obss, target=False)
        agent_idx = self.a_ids.index(agent_id)
        actor_loss = -self.critic(states, concat_acts)[agent_idx].mean()
        actor_loss.backward(retain_graph=True)
        actor.optimizer.step()

    @ torch.no_grad()
    def _compute_targets(self, next_acts, next_states, dones, rewards, a_id):
        target_values = self.critic_target(next_states, next_acts)
        dones = {a_id: torch.reshape(
            dones[a_id], ([self.batch_size, 1])) for a_id in self.a_ids}
        concat_dones = self._torch_dict_to_full_tensor(dones)
        target_values[concat_dones == 1.0] = 0.0
        concat_rewards = self._torch_dict_to_full_tensor(rewards)
        return (concat_rewards + (self.gamma * target_values)).flatten()


class M3ddpg(Maddpg):
    """ Extension for MADDPG from: 10.1609/aaai.v33i01.33014213
    Idea: Assume that all competetive agents act adversarially and optimize
    both actor/critic in a minimax way. """

    def __init__(self, env, alpha_team=1e-5, alpha_adversary=1e-3,
                 teams: dict='mpe', exclude_team: int=-1, use_action_norm=True,
                 *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        # Define for each agent to what "team" it belongs
        # Agent act non adversary vs their teammates
        if teams == 'mpe':
            # Use nomanclature from MPE environments
            self.teams = {a_id: (0 if 'adversary' in a_id else 1)
                          for a_id in self.a_ids}
        elif teams == 'adversary':
            # All agents in different teams
            self.teams = {a_id: idx for idx, a_id in enumerate(self.a_ids)}
        elif teams == 'cooperative':
            # All agents in the same team
            self.teams = {a_id: 0 for a_id in self.a_ids}
        else:
            self.teams = teams

        # Train one team normally to compare performance of M3DDPG with MADDPG
        self.exclude_team = exclude_team

        # Perturbation rate = "Learning rate" of the gradient descent step
        self.alpha_team = alpha_team
        self.alpha_adversary = alpha_adversary

        # Use equation 17 oder 16 from the original paper?
        self.use_action_norm = use_action_norm

        # Define which action (in a concatenated array) belongs to which agent
        self.agent_action_mapping = {}
        counter = 0
        for a_id in self.a_ids:
            n_act = self.env.action_space(a_id).shape[0]
            idxs = np.arange(counter, counter + n_act)
            self.agent_action_mapping[a_id] = idxs
            counter += n_act

    def _train_critic(self, agent_id, critic, batch):
        obss, next_obss, acts, states, next_states, rewards, dones = batch

        # TODO: is this double zero_grad unnecessary?
        critic.optimizer.zero_grad()

        # Use worst case actions for target calculation
        concat_next_acts = self._get_worst_acts(
            next_obss, next_states, target=True, agent_id=agent_id)

        critic.optimizer.zero_grad()
        concat_acts = self._torch_dict_to_full_tensor(acts)
        q_values = critic(states, concat_acts).flatten()
        target_q_values = self._compute_targets(
            concat_next_acts, next_states, dones, rewards, agent_id)
        critic_loss = critic.loss(target_q_values, q_values)
        critic_loss.backward(retain_graph=True)
        critic.optimizer.step()

    def _get_worst_acts(self, obss, states, target: bool, agent_id: str):
        """ Minimax strategy: Assume that all other agents act in the worst
        possible way for our own long term reward. """
        # Train one team only with normal MADDPG
        if self.teams[agent_id] == self.exclude_team:
            return self._get_acts(obss, target)

        actors = self.actor_targets if target else self.actors
        critics = self.critic_targets if target else self.critics

        acts_dict = {a_id: actor(obss[a_id]) for a_id, actor in actors.items()}
        acts = self._torch_dict_to_full_tensor(acts_dict)
        acts.retain_grad()

        # Objective: minimize Q of the current agent
        # TODO: Minus or plus????
        loss = critics[agent_id](states, acts).mean()

        # Perform a single step of gradient descent for the actions
        loss.backward(retain_graph=True)
        grad_l2_normalized = functional.normalize(acts.grad, dim=0, p=2)
        # Equation 16 from the original paper
        eps_team = -self.alpha_team * grad_l2_normalized
        eps_adversary = -self.alpha_adversary * grad_l2_normalized
        if self.use_action_norm:
            # For difficult problems: Equation 17 from the original paper
            eps_team *= torch.linalg.vector_norm(acts, ord=2, dim=0)
            eps_adversary *= torch.linalg.vector_norm(acts, ord=2, dim=0)

        worse_acts = []
        with torch.set_grad_enabled(not target):
            for a_id in self.a_ids:
                act_idxs = self.agent_action_mapping[a_id]
                if agent_id == a_id:
                    # This agent itself does not act adversarially to itself
                    worse_acts.append(acts[:, act_idxs])
                    continue
                elif self.teams[agent_id] == self.teams[a_id]:
                    # Teammates act slighly adversarially (think: random noise)
                    eps = eps_team
                else:
                    # Adversaries act strongly adversarially
                    eps = eps_adversary

                # Add perturbations to the original actions
                worse_acts.append(acts[:, act_idxs] + eps[:, act_idxs])

        return torch.clip(torch.cat(worse_acts, dim=1), -1, 1).to(self.device)

    def _train_actor(self, agent_id, actor, batch):
        obss, next_obss, actions, states, next_states, rewards, dones = batch

        actor.optimizer.zero_grad()
        concat_acts = self._get_worst_acts(
            obss, states, target=False, agent_id=agent_id)
        actor.optimizer.zero_grad()
        actor_loss = -self.critics[agent_id](states, concat_acts).mean()
        actor_loss.backward(retain_graph=True)
        actor.optimizer.step()
