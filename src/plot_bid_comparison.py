
import argparse

import pandas as pd
import matplotlib.pyplot as plt


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '--maddpg-paths',
        default="2022-07-06T07:36:18.285689_8279324_ml_opf.envs.energy_market_bidding:BiddingEcoDispatchEnv_maddpg:MarketMaddpg_0,2022-07-06T07:36:16.453374_6513876_ml_opf.envs.energy_market_bidding:BiddingEcoDispatchEnv_maddpg:MarketMaddpg_0",
        help="Experiment path file where the MADDPG results are stored")
    argparser.add_argument(
        '--mmaddpg-paths',
        default="2022-07-06T07:35:58.310648_1794314_ml_opf.envs.energy_market_bidding:OpfAndBiddingEcoDispatchEnv_general_market_maddpg:MarketMaddpgPab_0,2022-07-06T07:35:57.237941_11069599_ml_opf.envs.energy_market_bidding:OpfAndBiddingEcoDispatchEnv_general_market_maddpg:MarketMaddpgPab_0",
        help="Experiment path file where the M-MADDPG results are stored")
    argparser.add_argument(
        '--store',
        help="Store the plot in directory instead of showing it?",
        action='store_true'
    )
    argparser.add_argument(
        '--directory',
        default="data/DGX/20220701_shared_ne_test",
        help="Directory where the files are",
    )
    argparser.add_argument(
        '--running-average',
        help="Integer: Window of the running average plot",
        default=5,
        type=int
    )

    args = argparser.parse_args()

    directory = args.directory
    if directory[-1] != '/':
        directory += '/'

    maddpg_paths = args.maddpg_paths.split(',')
    mmaddpg_paths = args.mmaddpg_paths.split(',')
    print('')
    print(maddpg_paths)

    for label, paths in zip(('MADDPG', 'M-MADDPG'),
                            (maddpg_paths, mmaddpg_paths)):
        for idx, path in enumerate(paths):
            if args.directory not in path:
                path = directory + path
            if path[-1] != '/':
                path += '/'
            print('')
            print(path)

            bid_df = pd.read_csv(path + 'bidding.csv',
                                 index_col=0, header=None)

            n_agents = len(bid_df.columns) // 4
            bid_df = bid_df[bid_df.columns[0:n_agents]]
            bids = bid_df.mean(axis=1)
            # Compute running average
            bids = bids.rolling(window=args.running_average).mean()
            std_dev = bid_df.std(axis=1)  # Standard deviation over agents (!)
            std_dev = std_dev.rolling(window=args.running_average).mean()

            plt.plot(bid_df.index, bids, label=label + '_' + str(idx))
            # Plot standard dev as area around bids
            plt.fill_between(bid_df.index, y1=bids - std_dev, y2=bids + std_dev,
                             color='lightskyblue', alpha=0.4)

    plt.ylabel('Average bid')
    plt.xlabel('Training step')
    plt.legend(loc='upper right')
    plt.grid()

    if args.store:
        filename = f'{directory}bidding_course.pdf'
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    else:
        plt.show()


if __name__ == '__main__':
    main()
