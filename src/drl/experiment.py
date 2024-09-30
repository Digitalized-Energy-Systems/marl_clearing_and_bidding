#!/usr/bin/env python3
"""
Simple way of running experiments to compare different implementations and
or hyperparameter against each other. The reward curve gets plotted in the
end.

"""

import argparse
import ast
from datetime import datetime
import importlib
import os
import random

import gym
import matplotlib.pyplot as plt
import numpy as np
import pybullet_envs
import torch

from drl.ddpg import *
from drl.reinforce import *
from drl.dqn import *
from drl.maddpg import *
from drl.sac import *


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '--environment-name',
        type=str,
        default='CartPole-v1',
        help="Either provide the name of the openai gym environment class or "
        "provide import path to a custom creator function that generates your "
        "environment. For example: 'path.to:function_def'"
    )
    argparser.add_argument(
        '--num-experiments',
        type=int,
        default=3,
        help="Repeat experiment how many times with different seeds?"
    )
    argparser.add_argument(
        '--steps-to-train',
        type=int,
        default=1e5,
        help="Number of training steps per agent"
    )
    argparser.add_argument(
        '--agent-classes',
        help="Give a list of agent class names as string separated by comma.",
        type=str,
        default='Reinforce'
    )
    argparser.add_argument(
        '--hyperparams',
        help="Give a list of hyperparams settings as dicts separated by ';',"
        """for example "{'n_envs': 1}; {'n_envs': 2}" """,
        type=str,
        default=''
    )
    argparser.add_argument(
        '--env-hyperparams',
        help="Give a dict of hyperparams settings ,"
        """for example "{'load_scaling': 2.0}" """,
        type=str,
        default=''
    )
    argparser.add_argument(
        '--store-results',
        help="Store the results in a separate folder?",
        action='store_true'
    )
    argparser.add_argument(
        '--seed',
        help="Which seed to use to make experiment reproducible?",
        type=int
    )
    argparser.add_argument(
        '--test-interval',
        help="Perform tests at which interval?",
        type=int,
        default=9999999
    )
    argparser.add_argument(
        '--test-steps',
        # TODO: Maybe better test episodes?!
        help="How many test steps per test?",
        type=int,
        default=1000
    )
    argparser.add_argument(
        '--path',
        help="Store results to which path?",
        type=str,
        default='data/'
    )

    args = argparser.parse_args()

    agent_names = args.agent_classes
    if '[' in agent_names and ']' in agent_names:
        agent_names = agent_names[1:-1]
    agent_names = agent_names.replace(" ", "").replace("\t", "")
    agent_names = agent_names.split(',')

    n_steps = args.steps_to_train
    n_experiments = args.num_experiments
    env_name = args.environment_name
    store = args.store_results

    # Convert agent hyperparameters to list of dicts
    hyperparams = args.hyperparams
    if hyperparams:
        if '[' in hyperparams and ']' in hyperparams:
            hyperparams = hyperparams[1:-1]
        hyperparams = hyperparams.split(';')
        # TODO: This seems to be wrong, because dicts get split as well!
        hyperparams = [str_to_dict(hp) for hp in hyperparams]
    else:
        hyperparams = [dict()]

    # Convert environment hyperparameters to dict
    env_hyperparams = args.env_hyperparams
    if env_hyperparams:
        env_hyperparams = str_to_dict(env_hyperparams)
    else:
        env_hyperparams = dict()

    experiment(agent_names, env_name, n_experiments,
               n_steps, store, args.path, args.seed, hyperparams,
               env_hyperparams, args.test_interval, args.test_steps)


def experiment(
        agent_names: list,
        env_name: str,
        n_experiments: int,
        n_steps: int,
        store: bool,
        base_path: str,
        main_seed: int,
        agent_hyperparams: list=[{}],
        env_hyperparams: dict=dict(),
        test_interval=99999999,
        test_steps=10):

    names = []
    results = []
    colors = []

    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # TODO: Allow for list of hyperparams that are done in all combinations
    n = 0
    idx = 0
    for agent_name in agent_names:
        for hyperparams in agent_hyperparams:
            idx += 1
            color = color_cycle[idx]

            if ':' not in agent_name:
                agent_class = eval(agent_name)
            else:
                module_name, function_name = agent_name.split(':')
                module = importlib.import_module(module_name)
                agent_class = getattr(module, function_name)

            print('Start experiments with agent class: ', agent_class)
            for _ in range(n_experiments):
                if not main_seed:
                    seed = generate_seed()
                else:
                    # TODO: Currently always same seed is used. Useless for multiple experiments.
                    seed = main_seed
                apply_seed(seed)

                env_hyperparams['seed'] = seed
                env = create_environment(env_name, env_hyperparams)

                name = f'{env_name}_{agent_name}_{n}'.replace(':', '.')

                if store:
                    path = create_experiment_folder(
                        name, env_name, agent_name, seed, hyperparams,
                        env_hyperparams, base_path, n_steps, test_interval,
                        test_steps)
                else:
                    path = 'temp/'

                print('Start experiment: ', name)
                print(f'Seed is: {seed}')
                print(hyperparams)
                hyperparams['seed'] = seed
                agent = agent_class(
                    env, name=name, path=path, **hyperparams)
                agent.run(n_steps=n_steps, test_interval=test_interval,
                          test_steps=test_steps)
                agent.test(test_steps=test_steps)

                n += 1


def create_environment(env_name, env_hyperparams: dict):
    if ':' not in env_name:
        env = gym.make(env_name)
    elif 'pettingzoo' in env_name:
        module_name = env_name.replace(':', '.')
        module = importlib.import_module(module_name)
        env = module.env(**env_hyperparams)
    else:
        module_name, class_name = env_name.split(':')
        module = importlib.import_module(module_name)
        env = getattr(module, class_name)(**env_hyperparams)
    return env


def str_to_dict(string):
    string = string.replace(" {", "{").replace("\t", " ")
    dictionary = ast.literal_eval(string)
    return dictionary


def generate_seed():
    return int.from_bytes(os.urandom(3), byteorder='big')


def apply_seed(seed):
    # TODO: Apply seed to gym env!!!
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def create_experiment_folder(name, env_name, agent_name, seed, hyperparams,
                             env_hyperparams, path_dir, n_steps, test_interval,
                             test_steps):
    # Create path folder
    now = datetime.now().isoformat()
    full_name = '%s_%s_%s' % (now, seed, name)
    path = os.path.join(path_dir, full_name.replace(':', '.'), '')
    os.makedirs(os.path.dirname(path))

    # Store some meta-data
    with open(os.path.join(path, 'meta-data.txt'), 'w') as f:
        f.write(f'Description: {name}\n')
        f.write(f'Environment: {env_name}\n')
        f.write(f'DRL Algorithm: {agent_name}\n')
        f.write(f'Seed: {seed}\n')
        f.write(f'Training steps: {n_steps}\n')
        f.write(f'Test interval/steps: {test_interval}/{test_steps}\n')
        if hyperparams:
            f.write(f'DRL-agent hyperparams: {hyperparams}\n')
        if env_hyperparams:
            f.write(f'Environment hyperparams: {env_hyperparams}\n')

    return path


if __name__ == '__main__':
    main()
