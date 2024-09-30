"""
All standalone exploration methods.

"""

import numpy as np


class EpsilonGreedy():
    # TODO: Probably a superclass is a good idea, if there are other exploration methods
    def __init__(self, epsilon_start=1, epsilon_min=0.0, epsilon_end_step=0.5):
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_end_step = epsilon_end_step

    def update_epsilon(self, n_steps: int):
        """ Assumption: One update per step is done. """
        # TODO: try with eps *= 0.99 instead
        self.epsilon = max(self.epsilon_min, self.epsilon -
                           1 / n_steps / self.epsilon_end_step)

    def explore(self, n_act, action_function, obs):
        """ Apply the exploration to the action function. """
        if np.random.random() <= self.epsilon:
            return np.random.randint(n_act)
        return action_function(obs)


class GaussianNoise():
    def __init__(self, shape, std_dev=0.1):
        self.shape = shape
        self.std_dev = std_dev

    def __call__(self):
        return np.random.normal(scale=self.std_dev, size=self.shape)
