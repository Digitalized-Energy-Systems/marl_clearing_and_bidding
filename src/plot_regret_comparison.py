
import argparse

import pandas as pd
import matplotlib.pyplot as plt


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '--maddpg-paths',
        default="20220817_maddpg_10_regret_course/2022-08-17T08:07:04.227354_12642597_ml_opf.envs.energy_market_bidding.BiddingEcoDispatchEnv_maddpg.MarketMaddpg_0,20220817_maddpg_20_regret_course/2022-08-17T08:01:57.033474_4666463_ml_opf.envs.energy_market_bidding.BiddingEcoDispatchEnv_maddpg.MarketMaddpg_0,20220817_maddpg_30_regret_course/2022-08-17T08:06:51.899801_12930239_ml_opf.envs.energy_market_bidding.BiddingEcoDispatchEnv_maddpg.MarketMaddpg_0,20220817_maddpg_40_regret_course/2022-08-17T08:01:36.593127_12085161_ml_opf.envs.energy_market_bidding.BiddingEcoDispatchEnv_maddpg.MarketMaddpg_0",
        help="Experiment path file where the MADDPG results are stored")
    argparser.add_argument(
        '--mmaddpg-paths',
        default="20220819_mmaddpg_regret_course/2022-08-19T13:22:50.332595_279382_ml_opf.envs.energy_market_bidding.OpfAndBiddingEcoDispatchEnv_general_market_maddpg.MarketMaddpgPab_0,20220819_mmaddpg_regret_course/2022-08-19T13:22:41.342543_3900209_ml_opf.envs.energy_market_bidding.OpfAndBiddingEcoDispatchEnv_general_market_maddpg.MarketMaddpgPab_0,20220819_mmaddpg_regret_course/2022-08-19T13:22:59.179834_12244614_ml_opf.envs.energy_market_bidding.OpfAndBiddingEcoDispatchEnv_general_market_maddpg.MarketMaddpgPab_0,20220819_mmaddpg_regret_course/2022-08-19T13:23:17.090999_16698035_ml_opf.envs.energy_market_bidding.OpfAndBiddingEcoDispatchEnv_general_market_maddpg.MarketMaddpgPab_0",
        help="Experiment path file where the M-MADDPG results are stored")
    argparser.add_argument(
        '--store',
        help="Store the plot in directory instead of showing it?",
        action='store_true'
    )
    argparser.add_argument(
        '--directory',
        default="data/DGX/",
        help="Directory where the files are",
    )
    argparser.add_argument(
        '--running-average',
        help="Integer: Window of the running average plot",
        default=1,
        type=int
    )

    args = argparser.parse_args()

    directory = args.directory
    if directory[-1] != '/':
        directory += '/'

    maddpg_paths = args.maddpg_paths.split(',')
    mmaddpg_paths = args.mmaddpg_paths.split(',')

    for label, paths in zip(('MADDPG', 'M-MADDPG'), (maddpg_paths, mmaddpg_paths)):
        linestyle = '-' if label == 'MADDPG' else '--'
        for idx, path in enumerate(paths):
            if args.directory not in path:
                path = directory + path
            if path[-1] != '/':
                path += '/'
            print(path)
            regret_df = pd.read_csv(path + 'test_for_ne.csv',
                                    index_col=0, header=None)

            regret = regret_df[regret_df.columns[2]]
            # Compute running average
            regret = regret.rolling(window=args.running_average).mean()

            n_agents = (len(regret_df.columns) - 3) // 3
            # Standard deviation over agents (!)
            std_dev = regret_df[regret_df.columns[-n_agents:]].std(axis=1)
            std_dev = std_dev.rolling(window=args.running_average).mean()

            plt.plot(regret_df.index, regret,
                     label=label + '_' + str(n_agents),
                     linestyle=linestyle)
            # Plot standard dev as area around bids
            plt.fill_between(regret_df.index, y1=regret - std_dev, y2=regret + std_dev,
                             color='lightskyblue', alpha=0.4)

    plt.ylabel('Total Regret')
    plt.xlabel('Training step')
    plt.legend(loc='upper right')
    plt.grid()

    if args.store:
        filename = 'regret_course.pdf'
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    else:
        plt.show()


if __name__ == '__main__':
    main()
