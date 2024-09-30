
import argparse
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '--paths',
        default="20220822_maddpg_40_small_net,20220822_maddpg_30_small_net,20220822_maddpg_20_small_net,20220822_maddpg_10_small_net,20220822_mmaddpg_10,20220822_mmaddpg_20,20220822_mmaddpg_30,20220822_mmaddpg_40,20220824_maddpg_10_10k,20220824_maddpg_20_10k,20220824_maddpg_30_10k,20220824_maddpg_40_10k,20220824_mmaddpg_10_10k,20220824_mmaddpg_20_10k,20220824_mmaddpg_30_10k,20220824_mmaddpg_40_10k",
        help="Path(s) where experiment results are stored. Separate with ','.")
    argparser.add_argument(
        '--store',
        help="Store the plot in current directory instead of showing it?",
        action='store_true'
    )
    argparser.add_argument(
        '--plot',
        help="Plot results instead of printing them?",
        action='store_true'
    )
    argparser.add_argument(
        '--directory',
        default="data/DGX/",
        help="Parent directory where all paths are located.")
    args = argparser.parse_args()

    train_times = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list)))
    n_steps_set = set()

    for sub_path in args.paths.split(','):
        if sub_path[-1] != '/':
            sub_path += '/'
        if args.directory not in sub_path:
            path = args.directory + sub_path
        print('/n', path)
        for run_path in os.listdir(path):
            print(run_path)

            # Get meta-data
            with open(path + run_path + '/meta-data.txt') as f:
                lines = f.readlines()
            steps = int(lines[4].split(' ')[-1])
            n_steps_set.add(steps)
            n_agents = int(lines[7].split(':')[2][1:3])
            algo = lines[2].replace('\n', '').split(':')[-1]
            if 'GPU' in sub_path:
                algo += '_GPU'

            try:
                with open(path + run_path + '/training_time.txt') as f:
                    lines = f.readlines()
                train_time = float(lines[0].split(
                    ':')[1].replace('\n', '').replace(' ', '').replace('h', ''))
                train_times[algo][n_agents][steps].append(train_time)
            except FileNotFoundError:
                pass

            print('')

    print(train_times)

    n_agents = [10, 20, 30, 40]
    for n_steps in n_steps_set:
        algo1, algo2 = ('MarketMaddpg', 'MarketMaddpgPab')
        print(train_times[algo1][40][n_steps])
        print(train_times[algo2][40][n_steps])

        speedup = [(np.mean(train_times[algo1][n][n_steps]) /
                    np.mean(train_times[algo2][n][n_steps]))
                   for n in n_agents]
        plt.plot(speedup, label=f'{n_steps//1000}k CPU')

        if len(train_times[algo1 + '_GPU']) > 0 and len(train_times[algo2 + '_GPU']):

            speedup = [(np.mean(train_times[algo1 + '_GPU'][n][n_steps]) /
                        np.mean(train_times[algo2 + '_GPU'][n][n_steps]))
                       for n in n_agents]
            plt.plot(speedup, label=f'{n_steps//1000}k GPU')

    plt.ylabel('Mean speed-up')
    plt.xlabel('Number of agents')

    plt.legend()

    if args.store:
        filename = f'speed_up.pdf'
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    else:
        plt.show()


if __name__ == '__main__':
    main()
