"""
Open a bidding.csv file and plot mean, min and max bids of all agent over
training. """

import argparse

import pandas as pd
import matplotlib.pyplot as plt


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '--experiment-path',
        default="2021-10-01T13:40:02.279359_3990776_ml_opf.thesis_envs:qmarket_env_market_ddpg:MarketDdpg_{'absolute_qvalues': True}_0/",
        help="Experiment path file where the results are stored")
    argparser.add_argument(
        '--store',
        help="Store the plot in 'path' instead of showing it?",
        action='store_true'
    )
    argparser.add_argument(
        '--directory',
        help="Directory where the files are (Default: 'data')",
        default='data',
    )
    argparser.add_argument(
        '--from-idx',
        help="Integer: Start plotting from this point on",
        default=None,
        type=int
    )
    argparser.add_argument(
        '--to-idx',
        help="Integer: Start plotting from this point on",
        default=None,
        type=int
    )
    argparser.add_argument(
        '--running-average',
        help="Integer: Window of the running average plot",
        default=5,
        type=int
    )
    argparser.add_argument(
        '--mean',
        help="Mean bid over all agents (good for comparison of algorithms)",
        action='store_true'
    )

    args = argparser.parse_args()

    if args.directory[-1] != '/':
        args.directory += '/'
    if args.experiment_path[-1] != '/':
        args.experiment_path += '/'

    experiment_paths = args.experiment_path.split(',')
    for path in experiment_paths:
        path = args.directory + path

        plot_bidding(path, store=args.store, running_average=args.running_average,
                     from_=args.from_idx, to_=args.to_idx, mean=args.mean)

    if args.store:
        filename = f'{path}bidding.png'
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    else:
        plt.show()


def plot_bidding(path: str, plot_bid_range=True, plot_only_idxs=[], store=False,
                 running_average=5, from_=None, to_=None, mean=False):
    bid_df = pd.read_csv(path + 'bidding.csv', index_col=0)

    n_gens = int(len(bid_df.columns) / 4)
    column_names = [[f'gen{i}_{name}' for i in range(n_gens)]
                    for name in ('mean', 'min', 'max', 'std')]
    bid_df.columns = [item for sublist in column_names for item in sublist]

    if n_gens == 4:  # "1-LV-rural1--0-sw"
        labels = ['24.1 kW', '46.1 kW', '11.3 kW', '13.9 kW']
    elif n_gens == 5:
        labels = ['10.4 kW', '15.6 kW', '29.4 kW',
                  '50 kW', '8.2 kW']  # ,'75 kW'
    elif n_gens == 6:
        labels.append('75 kW')
    else:
        labels = [f'Gen_{i}' for i in range(n_gens)]

    if not plot_only_idxs:
        plot_only_idxs = range(n_gens)

    if from_ is None:
        from_ = bid_df.index[0]
    if to_ is None:
        to_ = bid_df.index[-1]
    x = bid_df.loc[from_:to_].index

    if plot_bid_range and not mean:
        for i in plot_only_idxs:
            # min_ = bid_df[f'gen{i}_min'].rolling(window=running_average).mean()
            # max_ = bid_df[f'gen{i}_max'].rolling(window=running_average).mean()
            bids = bid_df[f'gen{i}_mean'].loc[from_:to_].rolling(
                window=running_average).mean()
            std = bid_df[f'gen{i}_std'].loc[from_:to_].rolling(
                window=running_average).mean()
            plt.fill_between(x, y1=bids - std, y2=bids + std,
                             color='lightskyblue', alpha=0.4)

    if mean:
        bids_list = [bid_df[f'gen{i}_mean'].loc[from_:to_].rolling(
            window=running_average).mean()
            for i in plot_only_idxs]
        mean_bids = sum(bids_list) / len(bids_list)
        plt.plot(x, mean_bids, label='mean bids')  # , label=f'gen{i}'
    else:
        for i in plot_only_idxs:
            bids = bid_df[f'gen{i}_mean'].loc[from_:to_].rolling(
                window=running_average).mean()
            plt.plot(x, bids, label=labels[i])  # , label=f'gen{i}'

            # show_resulting_prices = True
            # if show_resulting_prices:
            #     plt.hlines(bid_df[f'gen{i}_mean'][-10:].mean(), x[0], x[-1])

    plt.ylabel('Mean bid and std deviation')
    plt.xlabel('Training step')
    plt.legend(loc='upper right')
    plt.grid()


if __name__ == '__main__':
    main()
