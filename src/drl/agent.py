
from abc import ABC, abstractmethod
import copy
import csv
import random
import time

import gym
from gym.spaces import Box, Discrete
import numpy as np
import pettingzoo

from .util import evaluation

import pdb


class DrlAgent(ABC):
    def __init__(self, env, gamma, n_envs=3, path='temp/', name='Unnamed Agent',
                 autoscale_obs=True, *args, **kwargs):
        # Must be a gym env or a parallel env from pettingzoo (same API as gym)
        assert isinstance(env, gym.Env)
        self.env = env
        self.n_obs = self.env.observation_space.shape[0]
        if isinstance(self.env.action_space, Discrete):
            self.n_act = self.env.action_space.n
        elif isinstance(self.env.action_space, Box):
            assert len(self.env.action_space.shape) == 1
            self.n_act = self.env.action_space.shape[0]
        else:
            raise NotImplementedError(
                'Only discrete and continuous action spaces possible')
        self.gamma = gamma
        self.n_envs = n_envs
        self.step = 0
        self.name = name
        self.path = path

        self.autoscale_obs = autoscale_obs
        # TODO: Better use an gym wrapper to do this. Otherwise potential of forgetting it somewhere
        self.scale_obs = ScalerObs(self.env.observation_space.low,
                                   self.env.observation_space.high)

        self.evaluator = evaluation.Eval(
            agent_names=[name], path=path, name=name)

    # TODO: Maybe move all of this to experiment?!
    # Does not work for custom algorithms (eg reward array stuff or MARL)
    def run(self, n_steps, test_interval=999999, test_steps=10):
        next_test = test_interval

        start_step = self.step
        self.start_time = time.time()
        self.n_train_steps = n_steps

        self.envs = [copy.deepcopy(self.env) for _ in range(self.n_envs)]
        [env.seed(random.randint(0, 100000)) for env in self.envs]

        dones = [True] * self.n_envs
        obss = [None] * self.n_envs
        states = [None] * self.n_envs
        total_rewards = [None] * self.n_envs
        while True:
            for idx, env in enumerate(self.envs):
                # TODO: Create extra function/methods for these MarlAgent special stuff
                if (dones[idx] is True or
                        (isinstance(self, MarlAgent) and all(dones[idx].values()))):
                    # Environment is done -> reset & collect episode data
                    if total_rewards[idx] is not None:
                        t = time.time() - self.start_time
                        self.evaluator.step(total_rewards[idx], self.step, t)
                    obss[idx] = self.envs[idx].reset()
                    try:
                        states[idx] = self.envs[idx].state()
                    except TypeError:
                        states[idx] = self.envs[idx].state
                    except AttributeError:
                        states[idx] = None

                    if isinstance(self, MarlAgent):
                        # TODO: Create extra method for MarlAgents!
                        # Multi-agent case: more than one reward
                        total_rewards[idx] = {a_id: 0 for a_id in self.a_ids}
                    else:
                        total_rewards[idx] = 0
                    dones[idx] = False
                self.step += 1
                if self.autoscale_obs:
                    obss[idx] = self.scale_obs(obss[idx])
                act = self.act(obss[idx])

                next_obs, reward, done, info = self.envs[idx].step(act)

                if 'TimeLimit.truncated' in info and info['TimeLimit.truncated'] is True and done is True:
                    actually_done = False
                else:
                    actually_done = done

                try:
                    next_state = self.envs[idx].state()
                except TypeError:
                    next_state = self.envs[idx].state
                except AttributeError:
                    next_state = None

                self.learn(
                    obss[idx], act, reward, next_obs, actually_done, states[idx], next_state, env_idx=idx)
                # TODO: Put this somewhere else + do it again for returns!
                with open(f'{self.path}rewards.csv', 'a') as f:
                    if isinstance(reward, np.ndarray):
                        row = [self.step] + list(reward)
                    elif isinstance(reward, dict):
                        row = [self.step] + list(reward.values())
                    else:
                        row = [self.step, reward]
                    wr = csv.writer(f, quoting=csv.QUOTE_ALL)
                    wr.writerow(row)
                obss[idx] = next_obs
                states[idx] = next_state
                dones[idx] = done

                if isinstance(self, MarlAgent):
                    total_rewards[idx] = {
                        a_id: r + reward[a_id]
                        for a_id, r in total_rewards[idx].items()}
                else:
                    total_rewards[idx] += reward

            if self.step >= next_test:
                self.test(test_steps=test_steps)
                next_test = self.step + test_interval

            if self.step - start_step >= self.n_train_steps:
                # TODO: Store results of testing instead of training
                self.evaluator.plot_reward()
                return self.evaluator

    def test(self, test_episodes=None, test_steps=30):
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
        # Some reference reward, e.g. human-performance or optimal actions
        total_baseline_rewards = []

        while True:
            count_episodes += 1
            obs = self.env.reset()
            done = False
            if isinstance(self, MarlAgent):
                total_rewards.append({a_id: 0 for a_id in self.a_ids})
            else:
                total_rewards.append(0)
            total_baseline_rewards.append(0)
            while not done:
                count_steps += 1
                # Use `test_act()` for testing instead of the noisy `act()`
                if self.autoscale_obs:
                    obs = self.scale_obs(obs)
                act = self.test_act(obs)
                if hasattr(self.env, 'test_step'):
                    obs, reward, done, _ = self.env.test_step(act)
                else:
                    obs, reward, done, info = self.env.step(act)
                if isinstance(self, MarlAgent):
                    total_rewards[-1] = {
                        a_id: r + reward[a_id]
                        for a_id, r in total_rewards[-1].items()}
                else:
                    total_rewards[-1] += reward

                if hasattr(self.env, 'baseline_reward'):
                    total_baseline_rewards[-1] += self.env.baseline_reward()

            if count_episodes >= test_episodes or count_steps >= test_steps:
                break

        print('Average episode return:', sum(
            total_rewards) / len(total_rewards))
        print('')  # TODO: delete

        if hasattr(self.env, 'baseline_reward'):
            # Compute rmse and mape
            total_rewards = np.array(total_rewards)
            print('rewards: ', total_rewards)
            total_baseline_rewards = np.array(total_baseline_rewards)
            print('Baseline rewards: ', total_baseline_rewards)
            # Sometimes, there is no baseline -> Filter nan entries
            wrong_entries = np.isnan(total_baseline_rewards)
            total_baseline_rewards = total_baseline_rewards[~wrong_entries]
            total_rewards = total_rewards[~wrong_entries]

            rmse = np.sqrt(np.square(total_rewards -
                                     total_baseline_rewards).mean())
            mape = np.mean(np.abs((total_baseline_rewards - total_rewards)
                                  / total_baseline_rewards)) * 100

            print('_________________________________')
            print('Test completed')
            print('Deviation from baseline:')
            print('rmse: ', rmse)
            print('mape: ', mape, '%')
            print('')
            # TODO: store somewhere instead

        with open(self.path + 'test_results.txt', 'w') as f:
            # TODO: Use csv instead and track every test(!) instead of overwriting
            f.write(f'Amount of test steps: {count_steps}\n')
            f.write(f'Amount of test episodes: {count_episodes}\n')
            f.write(f'Return per test episode: {total_rewards}\n')
            f.write(
                f'Average episode return: {sum(total_rewards) / len(total_rewards)}\n')
            f.write(f'Total sum of rewards: {sum(total_rewards)}\n')
            if hasattr(self.env, 'baseline_reward'):
                f.write(f'MAPE: {mape}\n')
                f.write(f'RMSE: {rmse}\n')

    @abstractmethod
    def act(obs):
        pass

    @abstractmethod
    def test_act(obs):
        """ Act without exploration for testing, e.g. remove noise. """
        pass

    @abstractmethod
    def learn(obs, act, reward, next_obs, done):
        # TODO: Maybe rename to agent.step()
        pass


class MarlAgent(DrlAgent):
    def __init__(self, env, gamma, n_envs=1, path='temp/', name='Unnamed Agent',
                 autoscale_obs=True, *args, **kwargs):
        # TODO: Autoscale not possible currently!
        self.env = env
        self.a_ids = env.possible_agents
        self.action_spaces = env.action_spaces
        self.observation_spaces = env.observation_spaces
        self.state_space = env.state_space

        self.n_obs = sum(len(s.low) for s in env.observation_spaces.values())
        self.n_act = sum(len(a.low) for a in env.action_spaces.values())
        self.n_states = len(env.state_space.low)
        self.n_agents = len(self.a_ids)

        self.gamma = gamma
        self.n_envs = n_envs
        self.step = 0
        self.name = name
        self.path = path
        self.evaluator = evaluation.Eval(
            agent_names=self.a_ids, path=path, name=name)

        self.autoscale_obs = False  # Currently not possible!

        # Use the sequential API of the pettingzoo environment
        if isinstance(env, pettingzoo.AECEnv):
            self.run = self.run_sequential
            self.test = self.test_sequential

    def run_sequential(self, n_steps, test_interval=99999999):
        """ When the pettingzoo environment is sequential (agents act one after
        the other), we need a new run method. """
        # TODO: This whole method is still WIP
        dones = {a_id: False for a_id in self.a_ids}
        obss = {}
        next_obss = {}
        acts = {}
        rewards = {}
        returns = {a_id: 0 for a_id in self.a_ids}

        self.start_time = time.time()
        self.env.reset()
        for a_id in self.env.agent_iter():
            next_obss[a_id], rewards[a_id], dones[a_id], _ = self.env.last()
            returns[a_id] += rewards[a_id]

            if dones[a_id]:
                obss = {}
                self.env.reset()
                dones = {a_id: False for a_id in self.a_ids}
                t = time.time() - self.start_time
                self.evaluator.step(returns, self.step, t)
                returns = {a_id: 0 for a_id in self.a_ids}
                continue

            assert self.env.agent_selection == a_id

            if self.step > self.start_train:
                acts[a_id] = self.single_act(a_id, next_obss[a_id], noisy=True)
            else:
                # Sample randomly from action space to collect data
                acts[a_id] = self.nn_action_spaces[a_id].sample()

            # Scale actions to action space of environment
            act = space_scaler(acts[a_id],
                               from_space=self.nn_action_spaces[a_id],
                               to_space=self.action_spaces[a_id])
            self.env.step(act)

            if a_id == self.a_ids[0]:
                self.step += 1
                # Agents went full circle -> store data and learn
                next_state = self.env.state()
                # TODO: Is this really correct? All agents get trained with state_1 this way. However, state changes with every action. So maybe we should actually store the state per agent (then the other actions are actually not required anymore...)
                if obss:
                    self.learn(obss, acts, rewards, next_obss, dones,
                               state, next_state)
                obss = next_obss
                state = next_state

            if self.step >= n_steps:
                break

    def test_sequential(self, test_episodes=None, test_steps=50):
        return None  # TODO before merge

    @abstractmethod
    def single_act(self, agent_id: str, obs, noisy: bool):
        """ Only the single agent "agent_id" acts. """
        pass


class TargetNetMixin():
    """ Mixin for agents that have a target net, which needs to be updated. """
    # TODO: This way, these are only functions and a mixin is not really required...

    def _hard_target_update(self, net, target_net):
        """ Set parameters of target net equal to q net. """
        # TODO: Add more efficient implementation here
        self._soft_target_update(net, target_net, tau=1)

    def _soft_target_update(self, net, target_net, tau=0.001):
        params = dict(net.state_dict())
        target_params = dict(target_net.state_dict())

        for name in params:
            params[name] = (
                tau * params[name].clone()
                + (1 - tau) * target_params[name].clone()
            )

        target_net.load_state_dict(params)


# TODO: Always use env wrappers instead (create for pettingzoo?!)
def space_scaler(action, from_space, to_space):
    return to_space.low + (to_space.high - to_space.low) * (
        (action - from_space.low) / (from_space.high - from_space.low))


# TODO: Again use wrappers instead
class ScalerObs():
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def __call__(self, obs):
        # import pdb
        # pdb.set_trace()
        # Minor offset to prevent rounding errors to do harm
        # eps = 1e-6
        # try:
        #     assert not ((obs + eps) < self.low).any()
        #     assert not ((obs - eps) > self.high).any()
        # except AssertionError:
        #     import pdb
        #     pdb.set_trace()
        # (2 * (obs - self.low) / (self.high - self.low)) - 1
        return (obs - self.low) / (self.high - self.low)


class PettingZooActScaler(pettingzoo.utils.wrappers.BaseParallelWraper):
    def __init__(self, env, nn_action_spaces: dict):
        super().__init__(env)
        self.nn_action_spaces = nn_action_spaces
        assert all(isinstance(self.action_space(a_id), Box) for a_id in getattr(
            self, 'possible_agents', [])), "should only use Scaling for Box spaces"

    def step(self, action: dict):
        scaled_act = {}
        for a_id in self.possible_agents:
            from_space = self.nn_action_spaces[a_id]
            to_space = self.action_space(a_id)
            new_act = to_space.low + (to_space.high - to_space.low) * (
                (action[a_id] - from_space.low) / (from_space.high - from_space.low))

            scaled_act[a_id] = np.clip(new_act, to_space.low, to_space.high)

        return super().step(scaled_act)

    def __str__(self):
        return str(self.env)
