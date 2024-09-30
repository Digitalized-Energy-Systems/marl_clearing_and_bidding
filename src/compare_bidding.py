
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
        default="20220822_maddpg_40_small_net,20220822_maddpg_30_small_net,20220822_maddpg_20_small_net,20220822_maddpg_10_small_net,20220822_mmaddpg_10,20220822_mmaddpg_20,20220822_mmaddpg_30,20220822_mmaddpg_40",
        # "20220822_maddpg_40_small_net,20220822_maddpg_30_small_net,20220822_maddpg_20_small_net,20220822_maddpg_10_small_net,20220822_mmaddpg_10,20220822_mmaddpg_20,20220822_mmaddpg_30,20220822_mmaddpg_40"
        # "20220824_mmaddpg_40_10k,20220824_mmaddpg_30_10k,20220824_mmaddpg_20_10k,20220824_mmaddpg_10_10k,20220824_maddpg_40_10k,20220824_maddpg_30_10k,20220824_maddpg_20_10k,20220824_maddpg_10_10k"
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
        '--bidding',
        help="Show bidding data? (costs more time)",
        action='store_true'
    )
    argparser.add_argument(
        '--directory',
        default="data/DGX/",
        help="Parent directory where all paths are located.")
    args = argparser.parse_args()

    bidding = defaultdict(list)
    std_bidding = defaultdict(list)
    opf = defaultdict(list)
    regret = defaultdict(list)
    train_times = defaultdict(list)
    n_steps_set = set()

    regret_new = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    train_times_new = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list)))
    # structure: r[algo][n_agents][step][run_idx] = float

    for sub_path in args.paths.split(','):
        if sub_path[-1] != '/':
            sub_path += '/'
        if args.directory not in sub_path:
            path = args.directory + sub_path
        print('/n', path)
        for run_path in os.listdir(path):
            print(run_path)

            # Get hyperparams
            with open(path + run_path + '/meta-data.txt') as f:
                lines = f.readlines()

            steps = int(lines[4].split(' ')[-1])
            n_steps_set.add(steps)
            hyperparams = lines[6][23:].replace('\n', '')
            env_hyperparams = lines[7][23:].replace('\n', '')

            # Remove seed from hyperparams to cluster
            hyperparams = ''.join(
                [hp for hp in hyperparams.split(',') if 'seed' not in hp])
            if hyperparams[-1] != '}':
                hyperparams += '}'

            algo = lines[2].replace('\n', '').split(':')[-1]
            hyperparams += '_' + str(steps)  # + '_' + env_hyperparams

            # Get regrets
            try:
                df = pd.read_csv(path + run_path +
                                 '/test_for_ne.csv', header=None)
                total_regret = df.iloc[-1:, 3].mean()
                n_agents = (len(df.columns) - 4) // 3
                hyperparams += '_' + str(n_agents)
                if not np.isnan(total_regret):
                    regret[hyperparams].append(total_regret)
                    regret_new[algo][n_agents][steps].append(total_regret)
                    print('print to new regret: ', algo,
                          n_agents, steps, total_regret)
            except FileNotFoundError as e:
                print('NE-test not found -> Experiment probably not finished yet \n')
                continue

            # Get bidding results
            if args.bidding:
                df = pd.read_csv(path + run_path + '/bidding.csv', header=None)
                mean_bid = df.iloc[-20:,
                                   list(range(1, n_agents + 1))].mean().mean()
                std_bid = df.iloc[-20:,
                                  list(range(3 * n_agents + 1, 4 * n_agents + 1))].mean().mean()
                bidding[hyperparams].append(mean_bid)
                std_bidding[hyperparams].append(std_bid)

            # Get OPF results
            try:
                df = pd.read_csv(path + run_path +
                                 '/test_opf.csv', header=None)
                mean_opf_mape = df.iloc[-3:, 2].mean()
                opf[hyperparams].append(mean_opf_mape)
            except FileNotFoundError:
                pass

            with open(path + run_path + '/training_time.txt') as f:
                lines = f.readlines()
            train_time = float(lines[0].split(
                ':')[1].replace('\n', '').replace(' ', '').replace('h', ''))
            train_times[hyperparams].append(train_time)
            train_times_new[algo][n_agents][steps].append(train_time)

            print('')

    for hp, _ in regret.items():
        print('')
        print(hp, ':', f'({len(regret[hp])} experiments)')
        if args.bidding:
            print('Bidding: ', sum(bidding[hp]) /
                  len(bidding[hp]), bidding[hp])
            print('Std Bidding: ', sum(
                std_bidding[hp]) / len(std_bidding[hp]), std_bidding[hp])
        if opf[hp]:
            print('OPF MAPE:', sum(opf[hp]) / len(opf[hp]), opf[hp])
        print('Total Regret: ', sum(
            regret[hp]) / len(regret[hp]), regret[hp])
        print('Train time: ', sum(
            train_times[hp]) / len(train_times[hp]), ' h')

    if args.plot:
        n_agents = [10, 20, 30, 40]

        for n_steps in n_steps_set:
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True,
                                           gridspec_kw={'height_ratios': [2.2, 1]})
            c1 = 'pink'
            regrets = [regret_new['MarketMaddpgPab'][n][n_steps]
                       for n in n_agents]
            bplot1 = plt.boxplot(regrets, positions=n_agents, showmeans=True,
                                 patch_artist=True)  # labels=labels,
            for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
                plt.setp(bplot1[item], color=c1)
            plt.setp(bplot1["boxes"], facecolor=c1)
            plt.setp(bplot1["fliers"], markeredgecolor=c1)

            try:
                c2 = 'blue'
                regrets = [regret_new['MarketMaddpg'][n][n_steps]
                           for n in n_agents]
                bplot2 = plt.boxplot(regrets, positions=n_agents, showmeans=True,
                                     patch_artist=True)
                for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
                    plt.setp(bplot2[item], color=c2)
                plt.setp(bplot2["boxes"], facecolor=c2)
                plt.setp(bplot2["fliers"], markeredgecolor=c2)
            except:
                pass

            # Plot random bidding as baseline
            c1 = 'green'
            regrets = [regret_new['MarketMaddpgPab'][n][10]
                       for n in n_agents]
            bplot3 = ax1.boxplot(regrets, positions=n_agents, showmeans=True,
                                 patch_artist=True)  # labels=labels,
            for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
                plt.setp(bplot3[item], color=c1)
            plt.setp(bplot3["boxes"], facecolor=c1)
            plt.setp(bplot3["fliers"], markeredgecolor=c1)

            ax1.legend([bplot1["boxes"][0], bplot2["boxes"][0], bplot3["boxes"][0]],
                       [f'M-MADDPG_{n_steps//1000}k',
                           f'MADDPG_{n_steps//1000}k', f'Baseline: Random'],
                       loc='upper right')
            fig.supylabel('Total regret', fontsize='medium', x=0.04)
            plt.xlabel('Number of agents')

            # Broken y-axis
            ax1.spines['bottom'].set_visible(False)
            ax2.spines['top'].set_visible(False)
            ax1.set_ylim(ymax=10.0)
            ax1.set_ylim(ymin=7.5)
            ax2.set_ylim(ymax=1.25)
            ax1.xaxis.tick_top()
            ax1.tick_params(labeltop=False)  # don't put tick labels at the top
            ax2.xaxis.tick_bottom()

            d = .005  # how big to make the diagonal lines in axes coordinates
            # arguments to pass to plot, just so we don't keep repeating them
            kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
            ax1.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
            ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

            kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
            # bottom-left diagonal
            ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
            ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **
                     kwargs)  # bottom-right diagonal
            fig.subplots_adjust(hspace=0.2)

            stepsize = 0.5
            ax1.yaxis.set_ticks(np.arange(7.5, 10.0, stepsize))
            ax2.yaxis.set_ticks(np.arange(0.1, 1.25, stepsize))

            if args.store:
                filename = f'regret_comparison{n_steps//1000}.pdf'
                plt.savefig(filename, bbox_inches='tight', pad_inches=0)
            else:
                plt.show()


if __name__ == '__main__':
    main()
