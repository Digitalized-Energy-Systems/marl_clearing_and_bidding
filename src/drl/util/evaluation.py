
import os

import matplotlib.pyplot as plt
import numpy as np

# TODO: rename to store or something?!


class Eval:
    def __init__(self, agent_names=['Unnamed agent'], average_last_eps=10,
                 plot_interval_steps=300, path='temp/', name=''):
        # TODO: Average over last n step, not episoded (too high variance in eps)
        self.average_last_eps = average_last_eps
        self.plot_interval_steps = plot_interval_steps

        self.path = f'{path}{name}'
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        self.plot = True

        self.steps = []
        self.times = []
        self.returns = {a_id: list() for a_id in agent_names}
        self.avrg_returns = {a_id: list() for a_id in agent_names}
        self.next_plot_counter = plot_interval_steps

    def step(self, return_, step: int, time: float):
        self.steps.append(step)
        self.times.append(time)

        for a_id in self.returns.keys():
            # TODO: Use scalar or dict, but only one of them, not both!
            if isinstance(return_, float):
                if np.isnan(return_):
                    return_ = self.avrg_returns[a_id][-1]
                self.returns[a_id].append(return_)
            elif isinstance(return_, dict):
                self.returns[a_id].append(return_[a_id])
                # Prevent NAN values
                if np.isnan(self.returns[a_id][-1]):
                    if len(self.avrg_returns[a_id]) == 0:
                        self.returns[a_id][-1] = 0
                    else:
                        # Replace with average of last returns
                        self.returns[a_id][-1] = self.avrg_returns[a_id][-1]
            else:
                for i, entry in enumerate(return_):
                    if np.isnan(entry):
                        return_[i] = self.avrg_returns[a_id][-1]
                self.returns[a_id].append(return_)

            self.avrg_returns[a_id].append(
                np.mean(self.returns[a_id][-self.average_last_eps:]))

        if step >= self.next_plot_counter and self.plot:
            self.next_plot_counter = step + self.plot_interval_steps
            self.plot_reward()
            print('Step: ', step, '| Episode: ', len(self.steps) - 1)

    def plot_reward(self):
        for a_id in self.avrg_returns.keys():
            plt.plot(self.steps[int(self.average_last_eps / 5):],
                     self.avrg_returns[a_id][int(self.average_last_eps / 5):],
                     label=a_id)
        plt.xlabel('Step')
        plt.ylabel(f'Mean return of last {self.average_last_eps} eps')
        plt.legend()
        plt.grid(visible=True, which='major', axis='both')
        plt.savefig(self.path + '.png')
        plt.clf()

        for a_id in self.avrg_returns.keys():
            plt.plot(self.times[int(self.average_last_eps / 2):],
                     self.avrg_returns[a_id][int(self.average_last_eps / 2):],
                     label=a_id)
        plt.xlabel('Time in s')
        plt.ylabel(f'Mean return of last {self.average_last_eps} eps')
        plt.legend()
        plt.grid(visible=True, which='major', axis='both')
        plt.savefig(self.path + '_timed' + '.png')
        plt.clf()
