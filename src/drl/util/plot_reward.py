
import argparse
import os
from collections import defaultdict
import pandas as pd


def main():
    # TODO: Print test returns instead to enable actual performcance comparison
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '--directories',
        type=str,
        help="Paths to one or multiple experiment folders"
    )

    args = argparser.parse_args()

    mean_reward = defaultdict(list)

    print(args.directories)
    directories = args.directories.split(',')
    for directory in directories:
        print(directory)
        for run_path in os.listdir(directory):
            # TODO: Find out how many training steps were performed (messy this way)
            steps = pd.read_csv(directory + run_path +
                                '/rewards.csv', header=None).index[-1]

            # Get hyperparams
            with open(directory + run_path + '/meta-data.txt') as f:
                lines = f.readlines()

            hyperparams = lines[6][23:].replace('\n', '')

            hyperparams += '_' + '_' + str(steps)

            print(hyperparams)


if __name__ == '__main__':
    main()
