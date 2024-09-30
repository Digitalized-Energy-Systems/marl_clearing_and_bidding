"""
An implementation of my market MADDPG algo for learning to bid on some kind of
energy market (arbitrary rules, but based on OPF). The algo learns not only the
bidding, but also the clearing of the market to then use the gradient
information from the market to optimize bidding.

The clearing of the market is learned with some internal DDPG. The agents are
trained with MADDPG where the critic is replaced by the internal DDPG.

"""


import copy
import csv
import importlib
import random
import time

import gym
from gym.wrappers import RescaleAction
import matplotlib.pyplot as plt
import numpy as np
import torch

from drl.ddpg import Ddpg1Step, Ddpg
from drl.agent import DrlAgent, PettingZooActScaler
from drl.networks import DDPGActorNet
from drl.shared_code.exploration import GaussianNoise
from drl.shared_code.processing import batch_to_tensors
from drl.experiment import create_environment

from market_ddpg import PabMarketDdpg

import pdb

"""
TODOs:

- Idea: Maybe use same data for all agents so that all move to same direction (and use same approach as normal MADDPG)
-> Implementation in pp-env as pwl cost functions (not separate gens)
- acts richtig scaled? (besonders mit internal ddpg)

Unnecessary
- Lernrate der Agents druch P_max teilen, weil sonst große Gens schneller lernen
- Use pettingzoo interface to make it better comparable to MADDPG?!
- Idea: Split into multiple bidding blocks to create multi-dim action space?! -> also nice proof that it works & better exploration for agents

Solved
- Add marginal costs. How? -> actually necessary for paper?
- Deal with vanishing and exploding gradient (replace sigmoid?, clip gradient?)
- Make OPF testing work
- Idea: DDPG ensembles to get more variance/stability in gradient signal
- Add time obs
- Warum act immer 1?????????????????? -> write actual act to csv (maybe they exploit their critic)
- replay buffer correctly used?!
 - Add agent bids to replay buffer?!
- obss scaled?! ja
- Add noise to bids
"""


class MarketMaddpg(DrlAgent):
    """ TODO """

    def __init__(self, env, memory_size=200000,
                 gamma=0.99, batch_size=256, tau=0.001,
                 agents_fc_dims=(128,),
                 actor_fc_dims=(512, 512), critic_fc_dims=(512, 512),
                 agents_learning_rate=0.001, start_train=1000,
                 actor_learning_rate=0.0002, critic_learning_rate=0.001,
                 start_train_agents=5000, early_start_train_agents=None,
                 agent_update_frequency=1,
                 independent_agent_training=True, n_ddpgs=1,
                 noise_std_dev=0.2, optimizer='RMSprop', activation='tanh',
                 full_agent_obs=False, grad_clip=None, ddpg_update_frequency=1,
                 # 'market_ddpg:PabMarketEnsembleDdpg'
                 model_rl_algo='market_ddpg:PabMarketDdpg',
                 test_opf=False, test_ne=True, data_std=0.4, data_mean=None,
                 current_bidding_data=False,
                 test_env_name='ml_opf.envs.energy_market_bidding:BiddingEcoDispatchEnv',
                 *args, **kwargs):

        agents_fc_dims = list(agents_fc_dims)
        actor_fc_dims = list(actor_fc_dims)
        critic_fc_dims = list(critic_fc_dims)

        if not early_start_train_agents:
            self.early_start_train_agents = start_train_agents
        else:
            self.early_start_train_agents = early_start_train_agents

        if activation == 'tanh':
            # Actor only outputs tanh action space [-1, 1]
            # Also clips to action space
            env = RescaleAction(env, -1, 1)
            self.min_range = -1
        elif activation == 'sigmoid':
            env = RescaleAction(env, 0, 1)
            self.min_range = 0
        # The wrapper destroys seeding of the action space -> re-seed
        env.action_space.seed(kwargs['seed'])

        super().__init__(env, gamma, *args, **kwargs)

        self.agents_fc_dims = agents_fc_dims

        self.n_agents = env.n_agents
        self.independent_agent_training = independent_agent_training
        self.full_agent_obs = full_agent_obs
        self.start_train_agents = start_train_agents
        self.agent_update_frequency = agent_update_frequency
        self.grad_clip = grad_clip
        self.ddpg_update_frequency = ddpg_update_frequency
        self.test_ne = test_ne
        self.test_opf = test_opf
        self.current_bidding_data = current_bidding_data
        if current_bidding_data is True:
            assert isinstance(self, MarketMaddpgPab)

        self.data_std = data_std
        self.data_mean = data_mean

        # TODO: Maybe use an internal env as well?!
        ddpg_env = copy.deepcopy(env)
        low = env.action_space.low[0:self.env.n_gens]
        high = env.action_space.high[0:self.env.n_gens]
        ddpg_env.action_space = gym.spaces.Box(low, high, seed=kwargs['seed'])

        module_name, class_name = model_rl_algo.split(':')
        module = importlib.import_module(module_name)
        rl_algo = getattr(module, class_name)
        self.internal_ddpgs = [rl_algo(ddpg_env,
                                       batch_size=128, tau=tau,
                                       start_train=start_train,
                                       actor_fc_dims=actor_fc_dims,
                                       critic_fc_dims=critic_fc_dims,
                                       actor_learning_rate=actor_learning_rate,
                                       critic_learning_rate=critic_learning_rate,
                                       noise_std_dev=noise_std_dev,
                                       optimizer=optimizer,
                                       activation=activation,
                                       grad_clip=grad_clip,
                                       seed=kwargs['seed'])
                               for _ in range(n_ddpgs)]

        self._init_networks(agents_fc_dims, agents_learning_rate, optimizer)
        self.bid_noise = GaussianNoise((self.n_agents,), noise_std_dev)
        self.noise_std_dev = noise_std_dev

        self.total_test_time = 0

        params = {'n_agents': env.n_agents,
                  'remove_gen_idxs': env.unwrapped.remove_gen_idxs,
                  'seed': kwargs['seed']}
        with open(self.path + 'removed_gens.txt', 'w') as f:
            f.write(str(list(env.unwrapped.remove_gen_idxs)))

        test_env = create_environment(test_env_name, params)
        nn_action_spaces = {
            a_id: gym.spaces.Box(
                -np.ones(test_env.action_space(a_id).shape),
                np.ones(test_env.action_space(a_id).shape),
                seed=kwargs['seed'])
            for a_id in test_env.possible_agents}
        self.test_env = PettingZooActScaler(test_env, nn_action_spaces)

    def _init_networks(self, agents_fc_dims, agents_learning_rate, optimizer):
        if not self.full_agent_obs:
            self.n_agent_obs = 6  # TODO: Which observations for agents?
        else:
            self.n_agent_obs = self.n_obs
        self.n_agent_act = 1  # Assumption: only one bid per agent
        self.agent_actors = [DDPGActorNet(
            self.n_agent_obs, agents_fc_dims, self.n_agent_act,
            agents_learning_rate, 'tanh',
            optimizer=optimizer) for _ in range(self.n_agents)]
        # TODO: Add target nets

    def act(self, obs):
        """ Only the internal ddpg (the grid operator/market) interacts with
        environment. """
        with torch.no_grad():
            bids = self._get_bids(obs).cpu().numpy()
            bids += self.bid_noise()
            bids = np.clip(bids, -1, 1)
        # Overwrite bid observations from env with agent bids
        obs[-self.n_agents:] = bids
        act = random.choice(self.internal_ddpgs).act(obs)
        act = np.append(act, bids)

        if self.step % 20 == 0:  # TODO: Delete
            with open(f'{self.path}actual_acts.csv', 'a') as f:
                wr = csv.writer(f, quoting=csv.QUOTE_ALL)
                wr.writerow([self.step] + list(act))
        return act

    @ torch.no_grad()
    def test_act(self, obs, noisy=False, random_bids=False):
        with torch.no_grad():
            bids = self._get_bids(obs, random_bids).cpu().numpy()
        obs[-self.n_agents:] = bids

        act = self.internal_ddpgs[0].test_act(obs)
        return np.append(act, bids)

    def learn(self, obs, act, reward, next_obs, done, state=None,
              next_state=None, env_idx=0):
        if np.isnan(obs).any() or np.isnan(next_obs).any():
            # Data is poisoned -> throw away data point
            return

        # Train internal DDPG (and store memory data there)
        act = act[:self.env.n_gens]  # Throw away bids here
        self.internal_ddpgs[0].remember(obs, act, reward, next_obs, done)
        if len(self.internal_ddpgs[0].memory) < self.internal_ddpgs[0].start_train:
            return

        # After agent training started, only train occasionally
        if (self.start_train_agents > len(self.memory)
                or self.step % self.ddpg_update_frequency == 0):
            for ddpg in self.internal_ddpgs:
                batch = self.internal_ddpgs[0].memory.sample_random_batch(
                    self.batch_size)
                batch = batch_to_tensors(batch, self.device, continuous=True)
                obss, acts, rewards, next_obss, dones = batch

                # Train on current bidding (not old bidding data)
                if (self.current_bidding_data is True
                        and len(self.memory) > self.start_train_agents):
                    with torch.no_grad():
                        bids = self._get_bids(copy.copy(obss)).detach()
                        bids += np.random.normal(scale=self.noise_std_dev,
                                                 size=(self.batch_size, self.n_agents))
                        bids = np.clip(bids, -1, 1)
                        obss[:, -self.n_agents:] = bids

                ddpg._learn(obss, acts, rewards, next_obss, dones)

        # Currently: Train with same batch as critic/model
        # batch = self.memory.sample_random_batch(self.batch_size)
        # batch = batch_to_tensors(batch, self.device, continuous=True)
        # obss, acts, rewards, next_obss, dones = batch

        # Create and store some data for evaluation
        if self.step % int(20) == 0:
            r = np.random.randint(len(self.internal_ddpgs))
            acts = self.internal_ddpgs[r].actor.forward(obss)
            q_values = self.internal_ddpgs[r].critic(obss, acts)
            if len(self.memory) >= self.early_start_train_agents:
                bids = self._get_bids(copy.copy(obss))
            else:
                bids = None
            self._store_data(bids, acts, q_values)

        if len(self.memory) > self.start_train_agents:
            self._learn(obss, acts, rewards, next_obss, dones)
        elif len(self.memory) > self.early_start_train_agents and self.step % 100 == 0:
            self._learn(obss, acts, rewards, next_obss, dones)

    @property
    def memory(self):
        return self.internal_ddpgs[0].memory

    @property
    def batch_size(self):
        return self.internal_ddpgs[0].batch_size

    @property
    def device(self):
        return self.internal_ddpgs[0].device

    def _learn(self, obss, acts, rewards, next_obss, dones):
        """ Only train the agents here. """

        # Train the agents in random order to treat them equally
        agent_idxs = random.sample(
            range(len(self.agent_actors)), len(self.agent_actors))

        for actor_idx in agent_idxs:
            actor = self.agent_actors[actor_idx]
            actor.optimizer.zero_grad()

            losses = []
            for ddpg in self.internal_ddpgs:
                # Sample extra data for each agent?!
                if self.independent_agent_training:
                    batch = self.memory.sample_random_batch(self.batch_size)
                    obss_ = batch_to_tensors(
                        batch, self.device, continuous=True)[0]
                else:
                    obss_ = copy.copy(obss)

                # TODO: Why is copy required??? Is this a hint for a bug?
                bids = self._get_bids(copy.copy(obss_))
                obss_[:, -self.n_agents:] = bids
                ddpg.actor.optimizer.zero_grad()  # TODO: Store these grads?
                ddpg.critic.optimizer.zero_grad()

                acts = ddpg.actor.forward(obss_)
                q_values = ddpg.critic(obss_, acts)

                agent_actor_loss = self._calc_loss(
                    q_values, bids, acts, actor_idx)
                losses.append(agent_actor_loss)
            # TODO: Where does this sqrt idea come from? (model-based papers?)
            (sum(losses) / np.sqrt(len(losses))).backward(retain_graph=True)
            # Clip gradients
            # TODO: Maybe clip during backward already (per DDPG, not afterwards)
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    actor.parameters(), self.grad_clip)
            actor.optimizer.step()

    def _calc_loss(self, q_values, bids, acts, actor_idx):
        # Each agent maximizes only its own profit
        # Note the absence of minus sign!
        # Because q = costs for grid operator = profit for agents
        agent_actor_loss = q_values[:, actor_idx].mean()
        return agent_actor_loss

    def _get_bids(self, obss, random_bids=False):
        # TODO: Refactor this method!
        if len(self.memory) < self.early_start_train_agents or random_bids:
            # Use random bids, if training has not started yet
            if self.data_mean is None or self.data_std is None:
                return (torch.rand((self.n_agents, )) * 2) - 1
            # Make educated guess: Agents bid somewhat higher than their costs
            return torch.clamp(torch.normal(
                mean=self.data_mean, std=self.data_std, size=(self.n_agents, )), -1, 1)

        # The agents know only current time (always first 6 observations)
        if not self.full_agent_obs:
            if len(obss.shape) > 1:
                agent_obss = obss[:, 0:6]
            else:
                agent_obss = torch.tensor(obss[0:6], dtype=torch.float)
        else:
            if len(obss.shape) == 1:
                agent_obss = torch.tensor(obss, dtype=torch.float)
            else:
                agent_obss = obss

        bids = [actor(agent_obss.to(self.device))
                for actor in self.agent_actors]

        return torch.cat(bids, dim=len(obss.shape) - 1)

    def _concat_bids_to_obs(self, obs, bids):
        return torch.cat((obs, bids), dim=len(obs.shape) - 1)

    def _store_data(self, bids, acts, q_values):
        if bids is not None:
            bids = (bids.detach().cpu().numpy() + 1) / 2
            bids_mean = bids.mean(axis=0)
            bids_min = bids.min(axis=0)
            bids_max = bids.max(axis=0)
            print('bids: ', bids_mean)
            bid_list = list(np.concatenate((bids_mean, bids_min, bids_max)))
            bid_list += list(np.std(bids, axis=0))
            with open(f'{self.path}bidding.csv', 'a') as f:
                wr = csv.writer(f, quoting=csv.QUOTE_ALL)
                wr.writerow([self.step] + bid_list)

        setpoints_mean = acts.mean(dim=0).detach().cpu().numpy()
        with open(f'{self.path}mean_relative_setpoints.csv', 'a') as f:
            wr = csv.writer(f, quoting=csv.QUOTE_ALL)
            wr.writerow([self.step] + list(setpoints_mean))

        exp_profit_mean = q_values.mean(dim=0).detach().cpu().numpy()
        with open(f'{self.path}mean_exp_profits.csv', 'a') as f:
            wr = csv.writer(f, quoting=csv.QUOTE_ALL)
            wr.writerow([self.step] + list(exp_profit_mean))

    def test(self, test_episodes=None, test_steps=10):
        start_time = time.time()
        if self.test_opf is True:
            self.test_opf_baseline(test_episodes, test_steps=10 * test_steps)
        if self.step >= self.start_train_agents - 2000 and self.test_ne is True:
            self.test_for_ne(n_samples=test_steps, bid_range=(-1.0, 1.0))

        self.total_test_time += time.time() - start_time
        print('current total test time: ', round(
            self.total_test_time, 3), ' s')

        with open(f'{self.path}training_time.txt', 'w') as f:
            train_time = time.time() - self.start_time - self.total_test_time
            f.write(
                f'Train time: {round(train_time/3600, 3)} h \n Test time: {round(self.total_test_time/3600, 3)} h')

    def test_opf_baseline(self, test_episodes=None, test_steps=30):
        """ Evaluate the trained agent on a specified number of test
        steps/episodes. If both episodes and steps are specified, the first
        condition that is met ends the evaluation. Use `None` to set no limit,
        but one of them must always be defined to prevent endless loop.
        Testing is done on the original (not distributed) environment to ensure
        no interference with training environments. """
        assert (test_episodes is not None or test_steps is not None)
        print('Start testing: ')
        if test_episodes is None:
            test_episodes = np.inf
        if test_steps is None:
            test_steps = np.inf
        count_episodes = 0
        count_steps = 0

        total_rewards = []
        total_rewards_without_penalty = []
        # Some reference reward, e.g. human-performance or optimal actions
        total_baseline_rewards = []

        while True:
            count_episodes += 1
            obs = self.env.reset()
            done = False
            total_rewards.append(0)
            total_baseline_rewards.append(0)
            total_rewards_without_penalty.append(0)
            while not done:
                count_steps += 1
                if self.autoscale_obs:
                    obs = self.scale_obs(obs)
                act = self.test_act(obs, random_bids=True)
                obs, reward, done, info = self.env.step(act)
                print('current acts:', self.env.get_current_actions())
                total_rewards[-1] += sum(reward)
                total_rewards_without_penalty[-1] += sum(
                    reward[:self.n_agents]) + reward[-1]  # Only agent costs and soft constraint

                if hasattr(self.env, 'baseline_reward'):
                    reward = self.env.baseline_reward()
                    try:
                        total_baseline_rewards[-1] += sum(reward)
                    except TypeError:
                        total_baseline_rewards[-1] += reward

                if total_baseline_rewards[-1] + 0.1 < total_rewards[-1]:
                    # Check if there is an error!
                    import pdb
                    pdb.set_trace()

            if count_episodes >= test_episodes or count_steps >= test_steps:
                break

        if hasattr(self.env, 'baseline_reward'):
            # Compute mse and mape
            total_rewards = np.array(total_rewards)
            print('rewards: ', total_rewards)
            total_rewards_without_penalty = np.array(
                total_rewards_without_penalty)
            print('rewards without penalty: ', total_rewards_without_penalty)
            total_baseline_rewards = np.array(total_baseline_rewards)
            print('Baseline rewards: ', total_baseline_rewards)
            print('Mine better: ', total_rewards > total_baseline_rewards)
            # Sometimes, there is no baseline -> Filter nan entries
            # Sometimes, the baseline is also zero -> MAPE compute impossible
            wrong_entries = np.isnan(
                total_baseline_rewards) + (total_baseline_rewards >= -0.1)
            total_baseline_rewards = total_baseline_rewards[~wrong_entries]
            total_rewards = total_rewards[~wrong_entries]
            total_rewards_without_penalty = total_rewards_without_penalty[~wrong_entries]

            rmse = np.sqrt(np.square(total_rewards -
                                     total_baseline_rewards).mean())
            mape = np.mean(np.abs(total_baseline_rewards - total_rewards)
                           / np.abs(total_baseline_rewards)) * 100
            rmse_ = np.sqrt(np.square(total_rewards_without_penalty -
                                      total_baseline_rewards).mean())
            mape_ = np.mean(np.abs(total_baseline_rewards - total_rewards_without_penalty)
                            / np.abs(total_baseline_rewards)) * 100
            mean_reward = np.mean(total_rewards)

            print('_________________________________')
            print('Test completed')
            print('Mean reward: ', mean_reward)
            print('Deviation from baseline:')
            print('rmse: ', rmse)
            print('mape: ', mape, '%')
            print('rmse without penalty: ', rmse_)
            print('mape without penalty: ', mape_, '%')
            print('')

            with open(f'{self.path}test_opf.csv', 'a') as f:
                wr = csv.writer(f, quoting=csv.QUOTE_ALL)
                wr.writerow([self.step, rmse, mape, rmse_, mape_, mean_reward])

    def test_for_ne(self, n_samples=3, bid_range=(-1.0, 1.0), max_error=0.02):
        def get_original_bids(self, env, obs_dict):
            a_ids = env.possible_agents
            # All agents have same obss anyway
            obs = obs_dict[a_ids[0]]
            bs = self._get_bids(obs).detach().cpu().numpy()
            return {a_id: np.array([bid]) for a_id, bid in zip(a_ids, bs)}

        test_for_ne_shared(self, self.test_env, get_original_bids,
                           n_samples, bid_range, max_error)


class MarketMaddpgPab(MarketMaddpg):
    """ Hard-code pay-as-bid (PAB) market rules into the algorithm. """

    def _calc_loss(self, q_values, bids, acts, actor_idx):
        # Each agent maximizes only its own profit
        # TODO: Maybe simply use sigmoid for actors?!
        abs_bid = ((bids[:, actor_idx] / 2 + 0.5) -
                   self.env.rel_marginal_costs) * self.env.max_price
        rel_setpoint = acts[:, actor_idx] / 2 + 0.5
        abs_setpoint = rel_setpoint * self.env.max_power[actor_idx]
        agent_actor_loss = (-abs_bid * abs_setpoint) * self.env.reward_scaling
        if self.env.bid_as_reward is True:
            # Setpoint zero -> bid too high -> use bid as loss (but only in this case)
            agent_actor_loss[rel_setpoint <= 0.05] += (
                abs_bid[rel_setpoint <= 0.05] * self.env.reward_scaling)
        return agent_actor_loss.mean()


def test_for_ne_shared(self, env, get_original_bids, n_samples=3, bid_range=(-1.0, 1.0), max_error=0.05):
    """ Test bidding strategies of all agents for Nash equilibrium and compute
    error relative to NE. """
    print('----- Test for Nash Equilibrium -----')
    if hasattr(env, 'possible_agents'):
        a_ids = env.possible_agents
    else:
        a_ids = [f'a{i}' for i in range(self.n_agents)]
    actual_rewards = np.zeros((n_samples, len(a_ids)))
    best_rewards = np.zeros((n_samples, len(a_ids)))
    successful = []

    step = 10
    for i in range(n_samples):
        print('Sample ', i, ' | Env step , ', step)
        obs_dict = env.unwrapped.reset(step=step)
        step += 500
        if step >= 35136:
            step -= 35136

        original_bids = get_original_bids(self, env, obs_dict)
        print('original_bids: ', [bid[0] for bid in original_bids.values()])

        # test_bids = list(np.concatenate(
        #     [original_bids[a_id] for a_id in a_ids]))
        # with open(f'{self.path}test_bids.csv', 'a') as f:
        #     wr = csv.writer(f, quoting=csv.QUOTE_ALL)
        #     wr.writerow(test_bids)
        # print('test bids: ', test_bids)
        # print('average test bids: ', sum(test_bids) / len(test_bids))

        for agent_idx, a_id in enumerate(a_ids):

            best_reward = 0
            guesses = np.array([])
            rewards = np.array([])
            new_guesses = np.append(
                np.array([original_bids[a_id]]), np.arange(
                    bid_range[0] + 0.01, bid_range[1] + 0.01, 1 / 3))
            # Educated guess: Slightly above marginal costs
            new_guesses = np.append(
                new_guesses, np.array([-0.79, -0.72]))
            new_guesses[new_guesses > 1.0] = 1.0

            for m in range(20):
                for bid in new_guesses:
                    bids = copy.deepcopy(original_bids)
                    bids[a_id] = bid
                    result = env.step(bids)[1]
                    if np.isnan(np.array(list(result.values()))).any():
                        break
                    rewards = np.append(rewards, result[a_id])
                    guesses = np.append(guesses, bid)

                if np.isnan(np.array(list(result.values()))).any():
                    break

                best_bid_idx = np.argmax(rewards)
                best_bid = guesses[best_bid_idx]

                # Did the best reward improve significantly?
                improve = abs(
                    abs(best_reward - max(rewards)) / best_reward)

                # Stop if absolute or relative improvement only marginal
                if (((max(rewards) - best_reward) < 0.001) or (improve < max_error and sum(rewards) != max(rewards))) and m > 3:
                    # Finished search
                    print(
                        f'Finish search after {len(guesses)-1} trials (m={m})')
                    best_reward = rewards[best_bid_idx]
                    break

                best_reward = rewards[best_bid_idx]

                # We need the two bids that are closest to best bid
                try:
                    high = min(guesses[guesses > best_bid])
                except ValueError:
                    # Probably there is no higher bid
                    high = max(best_bid + 0.01, bid_range[1])
                try:
                    low = max(guesses[guesses < best_bid])
                except ValueError:
                    low = min(best_bid - 0.01, bid_range[0])
                new_guesses = [(high + best_bid) / 2, (low + best_bid) / 2]

            if np.isnan(np.array(list(result.values()))).any():
                # Probably complete failure of the OPF
                print('Nan result')
                break

            print('guesses: ', guesses)
            print('rewards: ', rewards)

            best_rewards[i, agent_idx] = max(rewards)
            actual_rewards[i, agent_idx] = rewards[0]

            actual_bid = round(original_bids[a_id][0], 3)

            if not i > 20:
                plt.plot(guesses, rewards, '.')
                plt.plot(guesses[0], rewards[0], 'r.')
                plt.savefig(self.path + f'ne_test{i}_{agent_idx}' + '.png')
                plt.clf()

            print(f'{a_id}: Actual bid: {actual_bid} (r={rewards[0]})')
            print(
                f'Best bid: {guesses[np.argmax(rewards)]} (r={best_rewards[i, agent_idx]})')
            print('Error: ', best_rewards[i, agent_idx] - rewards[0])
        else:
            successful.append(int(i))

    actual_rewards = actual_rewards[np.array(successful), :]
    best_rewards = best_rewards[np.array(successful), :]

    # The simple error (=regret)
    simple_error = np.mean(best_rewards - actual_rewards, axis=0)
    total_simple_error = sum(simple_error)

    # Mean absolute percentage error (MAPE) calculation
    reward_ape = np.abs(
        (best_rewards - actual_rewards) / best_rewards) * 100
    # Mean over samples
    reward_mape = np.mean(reward_ape, axis=0)
    # Also mean over all agents
    reward_mmape = np.mean(reward_mape.flatten())

    # Root Mean squared error (RMSE) calculation
    # TODO: MSE not really suited because smaller gens can make less error
    # Maybe normalize with generator size?
    reward_rmse = np.sqrt(
        np.mean((best_rewards - actual_rewards)**2, axis=0))
    reward_mrmse = np.mean(reward_rmse.flatten())

    print('----------------------------------')
    print(f'Agent Error: {simple_error}')
    print(f'Total Error: {total_simple_error}')
    print(f'Ape: {reward_ape} in %')
    print(f'Mape: {reward_mape} in %')
    print(f'Mean mape: {reward_mmape}%')
    print(f'Rmse: {reward_rmse}')
    print(f'Mean rmse: {reward_mrmse}')

    with open(f'{self.path}test_for_ne.csv', 'a') as f:
        wr = csv.writer(f, quoting=csv.QUOTE_ALL)
        wr.writerow([self.step, reward_mmape, reward_mrmse, total_simple_error]
                    + list(reward_mape) + list(reward_rmse) + list(simple_error))

    np.savetxt(f'{self.path}final_regret.csv', best_rewards -
               actual_rewards, delimiter=",")
