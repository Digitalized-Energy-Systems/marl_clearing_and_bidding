""" These are essentially copies of MADDPG with some adjustments to be
applicable for my energy market problems. """

import copy
import csv
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from drl.maddpg import MaddpgSingleCritic, Maddpg, M3ddpg
from general_market_maddpg import test_for_ne_shared


class MarketMaddpgSingleCritic(MaddpgSingleCritic):
    # Delete all the following stuff
    def learn(self, obs, act, reward, next_obs, done, state=None,
              next_state=None, env_idx=0):
        if np.isnan(state).any() or np.isnan(next_state).any():
            # Data is poisoned -> throw away data
            return

        super().learn(obs, act, reward, next_obs, done, state,
                      next_state, env_idx)

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
        agent_idx = self.a_ids.index(agent_id)
        actor_loss = -self.critic(states, concat_acts)[agent_idx].mean()
        actor_loss.backward(retain_graph=True)
        actor.optimizer.step()

        if agent_id == self.a_ids[3]:
            print('mean acts:', concat_acts.mean(dim=0).cpu().detach().numpy())
            with open(f'{self.path}bidding.csv', 'a') as f:
                wr = csv.writer(f, quoting=csv.QUOTE_ALL)
                wr.writerow(
                    [self.step] + list(concat_acts.mean(dim=0).cpu().detach().numpy()))


class MarketMaddpg(Maddpg):
    def __init__(self, env, memory_size=200000,
                 gamma=0.97, batch_size=256, tau=0.001, start_train=1000,
                 start_train_actors=0, actor_train_interval=1,
                 critic_train_interval=1,
                 actor_fc_dims=(128,), critic_fc_dims=(512,),
                 actor_learning_rate=0.001, critic_learning_rate=0.001,
                 noise_std_dev=0.2, optimizer='RMSprop', grad_clip=None,
                 data_std=0.4, data_mean=None, *args, **kwargs):
        super().__init__(env, memory_size,
                         gamma, batch_size, tau, start_train,
                         start_train_actors, actor_train_interval,
                         critic_train_interval,
                         list(actor_fc_dims), list(critic_fc_dims),
                         actor_learning_rate, critic_learning_rate,
                         noise_std_dev, optimizer, *args, **kwargs)
        # TODO: Activation?!
        self.grad_clip = grad_clip
        self.total_test_time = 0

        self.data_std = data_std
        self.data_mean = data_mean

        with open(self.path + 'removed_gens.txt', 'w') as f:
            f.write(str(list(env.internal_env.remove_gen_idxs)))

    def act(self, obs_dict):
        if len(self.memory) < self.start_train_actors:
            # No training happened yet -> use random actions
            if self.data_std is None or self.data_mean is None:
                return {a_id: space.sample()
                        for a_id, space in self.nn_action_spaces.items()}
            actions = torch.clamp(
                torch.normal(mean=self.data_mean, std=self.data_std, size=(len(self.a_ids), )), -1, 1).detach().numpy()
            return {a_id: actions[idx]
                    for idx, a_id in enumerate(self.a_ids)}

        return self.test_act(obs_dict, noisy=True)

    def learn(self, obs, act, reward, next_obs, done, state=None,
              next_state=None, env_idx=0):
        print(f'step {self.step}')
        if np.isnan(state).any() or np.isnan(next_state).any() or np.isnan(np.array(list(reward.values()))).any():
            # Data is poisoned -> throw away data
            return

        super().learn(obs, act, reward, next_obs, done, state,
                      next_state, env_idx)

    def _train_actor(self, agent_id, actor, batch):
        """ Add grad clip and store some data. """
        obss, next_obss, actions, states, next_states, rewards, dones = batch
        actor.optimizer.zero_grad()
        concat_acts = self._get_acts(obss, target=False)
        actor_loss = -self.critics[agent_id](states, concat_acts).mean()
        actor_loss.backward(retain_graph=True)
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(actor.parameters(), self.grad_clip)
        actor.optimizer.step()

        if agent_id == self.a_ids[0] and self.step % 20 == 0:
            bids = self._get_acts(
                obss, target=False).cpu().detach().numpy() / 2 + 0.5
            bids_mean = bids.mean(axis=0)
            bids_min = bids.min(axis=0)
            bids_max = bids.max(axis=0)
            print('bids: ', bids_mean)
            bid_list = list(np.concatenate((bids_mean, bids_min, bids_max)))
            bid_list += list(np.std(bids, axis=0))
            with open(f'{self.path}bidding.csv', 'a') as f:
                wr = csv.writer(f, quoting=csv.QUOTE_ALL)
                wr.writerow([self.step] + bid_list)

    def test(self, test_episodes=None, test_steps=50):
        start_time = time.time()

        def get_original_bids(self, env, obs_dict):
            return self.test_act(obs_dict)
        test_for_ne_shared(self, self.env, get_original_bids, test_steps)

        self.total_test_time += time.time() - start_time
        print('current total test time: ', round(
            self.total_test_time, 3), ' s')

        with open(f'{self.path}training_time.txt', 'w') as f:
            train_time = time.time() - self.start_time - self.total_test_time
            f.write(
                f'Train time: {round(train_time/3600, 3)} h \n Test time: {round(self.total_test_time/3600, 3)} h')


class MarketM3ddpg(M3ddpg):
    def learn(self, obs, act, reward, next_obs, done, state=None,
              next_state=None, env_idx=0):
        if np.isnan(state).any() or np.isnan(next_state).any() or np.isnan(np.array(list(reward.values()))).any():
            # Data is poisoned -> throw away data
            return

        super().learn(obs, act, reward, next_obs, done, state,
                      next_state, env_idx)

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

    def _train_actor(self, agent_id, actor, batch):
        super()._train_actor(agent_id, actor, batch)

        if agent_id == self.a_ids[3]:
            concat_acts = self._get_acts(batch[0], target=False).mean(
                dim=0).cpu().detach().numpy()
            print('mean acts:', concat_acts)
            with open(f'{self.path}bidding.csv', 'a') as f:
                wr = csv.writer(f, quoting=csv.QUOTE_ALL)
                wr.writerow([self.step] + list(concat_acts))
