
import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '--paths',
        default="20220711_opf_test/2022-07-11T09:46:24.122443_3450771_ml_opf.envs.energy_market_bidding:OpfAndBiddingEcoDispatchEnv_general_market_maddpg:MarketMaddpgPab_0",
        help="Experiment path file where the results are stored")
    argparser.add_argument(
        '--store',
        help="Store the plot in directory instead of showing it?",
        action='store_true'
    )
    argparser.add_argument(
        '--directory',
        default="data/DGX/",
        help="Directory where the files are and where the plot will be stored",
    )
    argparser.add_argument(
        '--running-average',
        help="Integer: Window of the running average plot",
        default=1,
        type=int
    )
    argparser.add_argument(
        '--mean',
        help="Plot only the mean of multiple runs with same hyperparams",
        action='store_true'
    )
    argparser.add_argument(
        '--mse',
        help="Plot not only MAPE but als the MSE",
        action='store_true'
    )

    args = argparser.parse_args()

    directory = args.directory
    if directory[-1] != '/':
        directory += '/'

    for idx, path in enumerate(args.paths.split(',')):

        if args.directory not in path:
            path = directory + path
        if path[-1] != '/':
            path += '/'

        mapes = []
        mses = []
        for run_path in os.listdir(path):

            run_path = path + run_path + '/'
            print(run_path)

            with open(run_path + '/meta-data.txt') as f:
                lines = f.readlines()
            n_agents = lines[7].split(':')[2][1:3]

            opf_df = pd.read_csv(run_path + 'test_opf.csv',
                                 index_col=0, header=None)

            mape = opf_df[opf_df.columns[1]]
            # Compute running average
            mape = mape.rolling(window=args.running_average).mean()

            if not args.mean:
                plt.plot(opf_df.index, mape, label='MAPE in % ' + n_agents)
            else:
                mapes.append(mape)

            if args.mse:
                mse = opf_df[opf_df.columns[0]]
                # Compute running average
                mse = mse.rolling(window=args.running_average).mean()

                if not args.mean:
                    plt.plot(opf_df.index, 100 * mse,
                             label=f'100xMSE (n={n_agents})')
                else:
                    mses.append(mse)

        if args.mean:
            plt.plot(opf_df.index, np.sum(mapes, axis=0) / len(mapes),
                     label=f'MAPE in % (n={n_agents})')
            if args.mse:
                plt.plot(opf_df.index, 100 * np.sum(mses, axis=0) / len(mses),
                         label=f'100xMSE (n={n_agents})')

    plt.ylabel('Error')
    plt.xlabel('Training step')
    plt.legend(loc='upper right')
    plt.grid()

    if args.store:
        filename = f'{directory}opf_course.pdf'
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    else:
        plt.show()


if __name__ == '__main__':
    main()
