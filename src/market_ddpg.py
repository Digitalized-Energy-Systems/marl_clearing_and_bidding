"""
Implementations of DDPG for clearing of a smart market with some fixed market
rules (e.g. economic dispatch in the energy system with uniform pricing or
pay-as-bid pricing), where the market rules are directly hard-coded in the
loss function and do not need to be learned.

Assumption: The market costs are not part of the reward!

"""

import copy
import csv
import random
import time

import gym
import numpy as np
import torch

from drl.ddpg import Ddpg1Step
from drl.unpublished.ensemble_ddpg import EnsembleDdpg
from drl.networks import DDPGCriticNet, DDPGActorNet
from drl.shared_code.exploration import GaussianNoise
from drl.shared_code.memory import ReplayMemory
from ml_opf.penalties import ext_grid_overpower

import pdb


class PabMarketDdpg(Ddpg1Step):
    def __init__(self, env, activation='tanh', *args, **kwargs):
        # Only consider penalties as reward (market stuff is hardcoded)
        low = env.reward_space.low[env.n_agents:]
        high = env.reward_space.high[env.n_agents:]
        env.reward_space = gym.spaces.Box(low, high, seed=kwargs['seed'])

        super().__init__(env, activation=activation, *args, **kwargs)

        self.activation = activation
        self.max_price = env.max_price
        self.max_power = torch.tensor(
            env.max_power, requires_grad=False).to(self.device)
        self.learn_bids = env.learn_bids
        self.reward_scaling = env.reward_scaling
        self.delayed_actor_training = 100

    def act(self, obs):
        """ Use actor to create actions and add noise for exploration. """
        if self.memory.memory_counter < self.start_train + self.delayed_actor_training:
            act = self.env.action_space.sample()
            return act
        action = self.test_act(obs)
        action += self.noise()
        return np.clip(action, self.min_range, 1)

    def remember(self, obs, action, reward, next_obs, done):
        # Only consider penalties as reward (market stuff is hardcoded)
        reward = reward[self.env.n_agents:]
        self.memory.store_transition(obs, action, reward, next_obs, done)

    def _train_actor(self, obss, acts, rewards, next_obss, dones):
        """ Use not only the critic to train the actor, but add the known
        market costs to the loss. """

        self.actor.optimizer.zero_grad()
        acts = self.actor(obss)

        # This is the loss from the environment (constraints mainly)
        env_loss = -self.critic(obss, acts).sum(axis=1).mean()

        if len(self.memory) < self.start_train + self.delayed_actor_training:
            # Start training of actor only, when critic is trained a bit
            return

        # Assumption: The last n obs are bids (1 bid per setpoint/action)
        bids = obss[:, -self.n_act:]  # .detach()
        if self.activation == 'tanh':
            # Convert from range [-1, 1] to range [0, 1]
            setpoints = (acts / 2) + 0.5
            if self.learn_bids:
                # Only apply to bids, if generated by neural net
                bids = (bids / 2) + 0.5
        else:
            assert self.activation == 'sigmoid'
            setpoints = acts
        assert bids.shape == setpoints.shape
        # This is the loss that comes from accepting agent bids
        bidding_loss = ((bids * self.max_price * setpoints).mean(axis=0) *
                        self.max_power).sum() * self.reward_scaling

        # import pdb
        # pdb.set_trace()
        total_loss = env_loss + bidding_loss
        # print(env_loss.item(), '|', (bidding_loss).item(),
        #       '|', total_loss.item())

        total_loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.grad_clip)
        self.actor.optimizer.step()


class PabMarketEnsembleDdpg(EnsembleDdpg):
    def __init__(self, env, activation='tanh', n_actors=2, *args, **kwargs):
        # Only consider penalties as reward (market stuff is hardcoded)
        low = env.reward_space.low[env.n_agents:]
        high = env.reward_space.high[env.n_agents:]
        env.reward_space = gym.spaces.Box(low, high, seed=kwargs['seed'])

        super().__init__(env, activation=activation, n_actors=n_actors, *args, **kwargs)
        self.activation = activation
        self.max_price = env.max_price
        self.max_power = torch.tensor(
            env.max_power, requires_grad=False).to(self.device)
        self.learn_bids = env.learn_bids
        self.reward_scaling = env.reward_scaling
        self.delayed_actor_training = 100

    def act(self, obs):
        """ Use actor to create actions and add noise for exploration. """
        if self.memory.memory_counter < self.start_train + self.delayed_actor_training:
            act = self.env.action_space.sample()
            return act
        action = self.test_act(obs)
        action += self.noise()
        return np.clip(action, self.min_range, 1)

    def remember(self, obs, action, reward, next_obs, done):
        # Only consider penalties as reward (market stuff is hardcoded)
        reward = reward[self.env.n_agents:]
        self.memory.store_transition(obs, action, reward, next_obs, done)

    def _train_actor(self, critic_idx, actor_idx, obss, acts, rewards, next_obss, dones):
        """ Use not only the critic to train the actor, but add the known
        market costs to the loss. """

        if len(self.memory) < self.start_train + 100:
            # Start training of actor only, when critic is trained a bit
            return

        actor = self.actors[actor_idx]
        critic = self.critics[critic_idx]

        actor.optimizer.zero_grad()
        acts = actor(obss)

        # This is the loss from the environment (constraints mainly)
        env_loss = -critic(obss, acts).sum(axis=1).mean()

        if len(self.memory) < self.start_train + 100:
            # Start training of actor only, when critic is trained a bit
            return

        # Assumption: The last n obs are bids (1 bid per setpoint/action)
        bids = obss[:, -self.n_act:]  # .detach()
        if self.activation == 'tanh':
            # Convert from range [-1, 1] to range [0, 1]
            setpoints = (acts / 2) + 0.5
            if self.learn_bids:
                # Only apply to bids, if generated by neural net
                bids = (bids / 2) + 0.5
        else:
            assert self.activation == 'sigmoid'
            setpoints = acts
        assert bids.shape == setpoints.shape
        # This is the loss that comes from accepting agent bids
        bidding_loss = ((bids * self.max_price * setpoints).mean(axis=0) *
                        self.max_power).sum() * self.reward_scaling

        # import pdb
        # pdb.set_trace()
        total_loss = env_loss + bidding_loss
        # print(env_loss.item(), '|', (bidding_loss).item(),
        #       '|', total_loss.item())

        total_loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(actor.parameters(), self.grad_clip)
        actor.optimizer.step()

    @property
    def actor(self):
        return np.random.choice(self.actors)

    @property
    def critic(self):
        return np.random.choice(self.critics)


class UniformMarketDdpg(Ddpg1Step):
    def __init__(self, env, memory_size=200000,
                 gamma=0.99, batch_size=64, tau=0.001, start_train=2000,
                 agents_fc_dims=[128, 128],
                 actor_fc_dims=[512, 512], critic_fc_dims=[512, 512],
                 actor_learning_rate=0.0001, critic_learning_rate=0.001,
                 start_train_intermediate=2500, stop_train_intermediate=120000,
                 intermediate_update_frequency=1, actor_update_frequency=1,
                 noise_std_dev=0.1, optimizer='Adam',
                 *args, **kwargs):
        assert start_train_intermediate >= start_train

        self.agents_fc_dims = agents_fc_dims

        self.n_actors = len(env.gen_idxs)
        self.n_agents = len(env.gen_idxs)
        if self.n_agents == 0:
            # Use sgens instead
            self.n_actors = len(env.sgen_idxs)
            self.n_agents = len(env.sgen_idxs)

            assert len(env.net.gen.index) == 0  # TODO: allow for EHV grids!
            self.n_units = len(env.net.sgen.index)  # + len(env.net.gen.index)

        super().__init__(env, memory_size,
                         gamma, batch_size, tau, start_train,
                         actor_fc_dims, critic_fc_dims,
                         actor_learning_rate, critic_learning_rate,
                         noise_std_dev=0.1, optimizer=optimizer,
                         autoscale_obs=False, activation='sigmoid',
                         *args, **kwargs)

        self.max_price = env.max_price
        # Attention: Technically possible values, not currently possible!
        max_power = np.concatenate(
            [np.array(env.net.sgen.max_max_p_mw),
             np.array(env.net.gen.max_max_p_mw)])
        # Normalize the power values (otherwise far too big in big systems)
        self.high_value = sum(max_power)
        max_power /= self.high_value
        self.max_power = torch.tensor(
            max_power, requires_grad=False).to(self.device)

        self.start_train_intermediate = start_train_intermediate
        self.stop_train_intermediate = stop_train_intermediate
        # TD3 feature: Delayed update of the actor layers
        self.intermediate_update_frequency = intermediate_update_frequency
        self.actor_update_frequency = actor_update_frequency

        self.nn_action_space = gym.spaces.Box(
            -np.zeros(self.env.action_space.shape),
            np.ones(self.env.action_space.shape),
            seed=kwargs['seed'])

    def _init_networks(self, actor_fc_dims, actor_learning_rate,
                       critic_fc_dims, critic_learning_rate, optimizer):
        """ Assumption: Single-step scenarios only -> no target nets required.
        """
        self.intermediate = DDPGActorNetUniform(
            self.n_obs + self.n_agents, actor_fc_dims, self.n_units,
            actor_learning_rate, 'sigmoid', optimizer=optimizer)
        self.critic = DDPGCriticNet(
            self.n_obs + self.n_agents, self.n_units + 1, critic_learning_rate, critic_fc_dims)

    def _init_noise(self, std_dev):
        self.setpoints_noise = GaussianNoise((self.n_units,), std_dev)
        self.price_noise = GaussianNoise((1,), std_dev)

    def _init_memory(self, memory_size: int):
        # Additional obs: one bid per agents
        # Additional action: resulting uniform market price
        self.memory = ReplayMemory(
            memory_size, self.n_obs + self.n_agents, self.n_act + 1)

    def act(self, obs):
        """ Use actor to create actions and add noise for exploration. """

        # Assume some random bids (in absence of actual bidding agents)
        self.last_bids = self._random_bids(obs).to(self.device)

        if self.memory.memory_counter < self.start_train_intermediate:
            self.last_price = np.random.random((1,))
            self.last_setpoints = self.nn_action_space.sample()
        else:
            setpoints, price = self.test_act(obs)
            setpoints += self.setpoints_noise()
            self.last_price = np.clip(price + self.price_noise(), 0, 1)
            self.last_setpoints = np.clip(
                setpoints, self.nn_action_space.low, self.nn_action_space.high)

        # Uniform pricing: set all setpoints to 0, where bid>market_price
        # TODO: Maybe better do this in env?!
        last_setpoints = copy.copy(self.last_setpoints)
        last_setpoints[self.last_bids.cpu().numpy() > self.last_price] = 0.0
        return last_setpoints

    @ torch.no_grad()
    def test_act(self, obs, noisy=False):
        obs = torch.tensor(obs, dtype=torch.float).to(self.device)
        # TODO: Store the assumed bids!
        self.last_bids = self._random_bids(obs).to(self.device)
        full_obs = self._concat_bids_to_obs(obs, self.last_bids)
        setpoints, price = self.intermediate(full_obs)
        return setpoints.cpu().numpy(), price.cpu().numpy()

    def _random_bids(self, obs):
        if len(obs.shape) == 2:
            # For a batch of observations -> return batch of bids
            return torch.rand(obs.shape[0], self.n_agents)
        else:
            # During application: A single observation
            return torch.rand(self.n_agents)

    def _concat_bids_to_obs(self, obs, bids):
        return torch.cat((obs, bids), dim=len(obs.shape) - 1)

    def remember(self, obs, action, reward, next_obs, done):
        print('reward: ', reward)
        obs = np.concatenate((obs, self.last_bids))
        next_obs = np.concatenate((next_obs, self.last_bids))
        # Use the original actions that the nn outputted and the price
        action = np.concatenate((self.last_setpoints, self.last_price))
        self.memory.store_transition(obs, action, reward, next_obs, done)

    def _train_actor(self, obss, acts, rewards, next_obss, dones):
        if (self.step < self.start_train_intermediate
                or self.step % self.intermediate_update_frequency != 0):
            return

        # Train intermediate network first
        self.intermediate.optimizer.zero_grad()

        # Completely random bids for training of intermediate
        # Idea: prevent any overfitting to agent bidding
        # drawback: grid operator cannot react to bidding strategies, eg market manipulation
        # Also: The intermediate wont be trained on the actual distribution but some uniform distribution
        bids = self._random_bids(obss).to(self.device)
        obss[:, -self.n_agents:] = bids
        setpoints, market_price = self.intermediate(obss)

        # Uniform pricing
        # Simple implementation: Minimize all (!) setpoints, even if not accepted (converges to set=0, if the bid is not accepted, which is intuitive)
        # market_costs = (market_price *
        #                 (setpoints * self.max_power).sum(dim=0)).mean()
        # Alternative implementation: Only use the accepted setpoints
        # From a theoritical perspective this should be better, because less to learn (but same result)
        market_costs = ((market_price * setpoints * self.max_power)[bids < market_price]
                        ).sum() / self.batch_size

        # Note: Setpoints where the bid was lower than the market price get cut out before action anyway
        env_costs = -torch.mean(
            self.critic(obss, self._concat_bids_to_obs(setpoints, market_price)))

        # pdb.set_trace()
        print(f'market loss: {market_costs} | env loss: {env_costs}')

        market_loss = (market_costs + env_costs).to(self.device)
        market_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.intermediate.parameters(), self.grad_clip)
        self.intermediate.optimizer.step()

    def run(self, n_steps, test_interval=999999):
        next_test = test_interval
        if hasattr(self, 'start_train'):
            next_test += self.start_train

        start_step = self.step
        start_time = time.time()
        self.n_train_steps = n_steps
        n_env = self.n_envs

        self.envs = [copy.deepcopy(self.env) for _ in range(n_env)]
        [env.seed(random.randint(0, 100000)) for env in self.envs]

        dones = [True] * n_env
        obss = [None] * n_env
        total_rewards = [None] * n_env
        while True:
            for idx, env in enumerate(self.envs):
                # TODO: Create extra function/methods for these MarlAgent special stuff
                if (dones[idx] is True):
                    # Environment is done -> reset & collect episode data
                    if total_rewards[idx] is not None:
                        t = time.time() - start_time
                        self.evaluator.step(
                            {self.name: total_rewards[idx]}, self.step, t)
                    obss[idx] = self.envs[idx].reset()
                    total_rewards[idx] = 0
                    dones[idx] = False
                self.step += 1
                if self.autoscale_obs:
                    obss[idx] = self.scale_obs(obss[idx])
                act = self.act(obss[idx])

                next_obs, reward, done, info = self.envs[idx].step(act)

                # Penalty for power flow over slack
                c_slack = sum(
                    self.envs[idx].net.res_ext_grid.p_mw) / self.high_value * 10
                if c_slack < 0.0:
                    c_slack = 0.0
                # TODO: Maybe do not allow values>0 (see as power plant instead of slack)
                reward -= c_slack

                self.learn(
                    obss[idx], act, reward, next_obs, done, env_idx=idx)

                market_costs = -sum(self.last_price *
                                    (act * self.max_power.cpu().numpy()))
                with open(f'{self.path}rewards.csv', 'a') as f:
                    wr = csv.writer(f, quoting=csv.QUOTE_ALL)
                    wr.writerow(
                        [self.step, reward + info['penalty'] + c_slack, market_costs, -c_slack, -info['penalty']])
                with open(f'{self.path}market_price.csv', 'a') as f:
                    wr = csv.writer(f, quoting=csv.QUOTE_ALL)
                    wr.writerow(self.last_price)
                obss[idx] = next_obs
                dones[idx] = done

                total_rewards[idx] += reward + market_costs

            if self.step >= next_test:
                self.test(test_steps=30)
                next_test = self.step + test_interval

            if self.step - start_step >= self.n_train_steps:
                # TODO: Store results of testing instead of training
                self.evaluator.plot_reward()
                return self.evaluator

    def _train_critic(self, obss, acts, rewards, next_obss, dones):
        targets = super()._train_critic(obss, acts, rewards, next_obss, dones)
        # Store critic loss
        with torch.no_grad():
            q_values = self.critic(obss, acts).flatten()
            critic_loss = self.critic.loss(
                targets, q_values).cpu().numpy().sum()
        with open(f'{self.path}critic_loss.csv', 'a') as f:
            wr = csv.writer(f, quoting=csv.QUOTE_ALL)
            wr.writerow([self.step, critic_loss])


class DDPGActorNetUniform(DDPGActorNet):
    def _init_linear_layers(self, n_obs, fc_dims, n_act):
        super()._init_linear_layers(n_obs, fc_dims, n_act)

        # Network return a single uniform market price
        self.uniform_price = torch.nn.Linear(fc_dims[-1], 1)

    def _init_weights(self):
        super()._init_weights()

        f3 = 0.003
        self.uniform_price.weight.data.uniform_(-f3, f3)
        self.uniform_price.bias.data.uniform_(-f3, f3)

    def forward(self, obs):
        output = obs
        for fc, ln in zip(self.fcs, self.lns):
            output = fc(output)
            output = ln(output)
            output = torch.nn.functional.relu(output)

        setpoints = getattr(torch, self.output_activation)(self.value(output))
        price = getattr(torch, self.output_activation)(
            self.uniform_price(output))

        return setpoints, price
