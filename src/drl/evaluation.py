""" Automatically evaluate all experiment runs at a give path. """

import argparse
from collections import defaultdict
import os

"""
TODOs
- Allow for multiple envs
- Consider computation time
- Allow for multiple experiments at once (--level 2 as arg?!)

"""


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '--path',
        type=str,
        default='data/',
        help="Path to one or multiple experiment runs that should be evaluated."
        "Use the same path as you used for --path in the experiment definition"
    )
    argparser.add_argument(
        '--must-contain',
        type=str,
        default='',
        help="Define some string that must be included in directory name"
        "(can be used for filtering)."
    )

    args = argparser.parse_args()

    results = defaultdict(lambda: defaultdict(list))
    environment = None
    for run_path in os.listdir(args.path):
        if args.must_contain not in run_path:
            continue
        try:
            with open(args.path + run_path + '/meta-data.txt') as f:
                lines = f.readlines()
            algorithm = strip_string(lines[2].split(':')[-1])
            environment = strip_string(lines[1].split(':')[-1])
            hyperparams = lines[4][23:].replace('\n', '')
            with open(args.path + run_path + '/test_results.txt') as f:
                performance = strip_string(f.readlines()[3].split(':')[-1])
                computation_time = None
        except (NotADirectoryError, FileNotFoundError) as e:
            # Ignore all non-directories or not-finished experiments
            continue
        key = algorithm + ': ' + hyperparams
        results[environment][key].append(float(performance))

    print('')
    print('Average test return of each algorithm:')
    print('')
    for env, algo_results in results.items():
        print('\n', 'Environment: ', env, '\n')
        for algo, performance in algo_results.items():
            n_runs = len(performance)
            mean_perf = round(sum(performance) / n_runs, 4)
            print(f'{algo}: {mean_perf} (over {n_runs} runs)')
            print('')


def strip_string(string):
    return string.replace(' ', '').replace('\n', '')


if __name__ == '__main__':
    main()
