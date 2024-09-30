
from collections import deque

import numpy as np

import pdb


class ReplayMemory(object):
    def __init__(self, max_size, n_inputs, n_actions, n_rewards=1):
        self.mem_size = max_size
        self.memory_counter = 0
        self.state_memory = np.zeros((self.mem_size, n_inputs))
        self.next_state_memory = np.zeros((self.mem_size, n_inputs))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros((self.mem_size, n_rewards))
        self.done_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(
            self, state: list, action: list, reward, next_state: list, done):
        index = self.memory_counter % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.next_state_memory[index] = next_state
        self.done_memory[index] = done

        self.memory_counter += 1

        return index

    def sample_random_batch(self, batch_size: int):
        """ Sample from the memory randomly and uniformly. """
        batch_idxs = np.random.choice(len(self), batch_size, replace=False)
        return self._get_batch(batch_idxs)

    def _get_batch(self, batch_idxs):
        states = self.state_memory[batch_idxs]
        next_states = self.next_state_memory[batch_idxs]
        rewards = self.reward_memory[batch_idxs]
        actions = self.action_memory[batch_idxs]
        done = self.done_memory[batch_idxs]

        return states, actions, rewards, next_states, done

    def __len__(self):
        return min(self.memory_counter, self.mem_size)


class MarlReplayMemory():
    def __init__(self, max_size, a_ids, observation_spaces: dict, action_spaces: dict,
                 state_space, n_rewards=1):
        self.a_ids = a_ids
        self.mem_size = max_size
        self.memory_counter = 0
        if state_space:
            self.state_memory = np.zeros((self.mem_size, state_space.shape[0]))
            self.next_state_memory = np.zeros(
                (self.mem_size, state_space.shape[0]))
            self.store_states = True
        else:
            # Sometimes, there is no differentiation between states and obs
            self.state_memory = None
            self.next_state_memory = None
            self.store_states = False

        self.obs_memory = {  # TODO: is this really required?!
            a_id: np.zeros((self.mem_size, observation_spaces[a_id].shape[0]))
            for a_id in self.a_ids}
        self.next_obs_memory = {
            a_id: np.zeros((self.mem_size, observation_spaces[a_id].shape[0]))
            for a_id in self.a_ids}
        self.act_memory = {
            a_id: np.zeros((self.mem_size, action_spaces[a_id].shape[0]))
            for a_id in self.a_ids}
        self.reward_memory = {a_id: np.zeros((self.mem_size, 1))
                              for a_id in self.a_ids}
        self.done_memory = {a_id: np.zeros(self.mem_size, dtype=np.float32)
                            for a_id in self.a_ids}

    def store_transition(
            self, obs, state, action, reward, next_obs, next_state, done):
        index = self.memory_counter % self.mem_size
        for a_id in self.a_ids:
            self.obs_memory[a_id][index] = obs[a_id]
            self.next_obs_memory[a_id][index] = next_obs[a_id]
            self.act_memory[a_id][index] = action[a_id]
            self.reward_memory[a_id][index] = reward[a_id]
            self.done_memory[a_id][index] = done[a_id]
        # There is only one state, independent of the agents' perspective
        self.state_memory[index] = state
        self.next_state_memory[index] = next_state

        self.memory_counter += 1

        return index

    def sample_random_batch(self, batch_size: int):
        """ Sample from the memory randomly and uniformly. """
        batch_idxs = np.random.choice(len(self), batch_size, replace=False)
        return self._get_batch(batch_idxs)

    def _get_batch(self, batch_idxs):
        obs = {a: self.obs_memory[a][batch_idxs] for a in self.a_ids}
        next_obs = {a: self.next_obs_memory[a][batch_idxs] for a in self.a_ids}
        rewards = {a: self.reward_memory[a][batch_idxs] for a in self.a_ids}
        actions = {a: self.act_memory[a][batch_idxs] for a in self.a_ids}
        dones = {a: self.done_memory[a][batch_idxs] for a in self.a_ids}
        if self.store_states:
            states = self.state_memory[batch_idxs]
            next_states = self.next_state_memory[batch_idxs]
        else:
            # TODO: Test if this is working
            states = np.concatenate([obs[a] for a in self.a_ids], axis=1)
            next_states = np.concatenate(
                [next_obs[a] for a in self.a_ids], axis=1)

        return obs, next_obs, actions, states, next_states, rewards, dones

    def __len__(self):
        return min(self.memory_counter, self.mem_size)


class PrioritizedReplayMemory(ReplayMemory):
    """ Schaul et al: Prioritized Experience Replay
    (https://arxiv.org/pdf/1511.05952.pdf) """

    def __init__(self, max_size, n_inputs, n_actions, alpha=0.6, beta_start=0.4, beta_update_time=100000, *args, **kwargs):
        super().__init__(max_size, n_inputs, n_actions, *args, **kwargs)
        self.prio_memory = np.zeros(self.mem_size)
        self.current_max_prio = 1.0
        self.alpha = alpha
        self.beta = beta_start
        self.beta_increment = (1 - beta_start) / beta_update_time
        assert self.beta_increment >= 0

    def store_transition(self, state, action, reward, next_state, done):
        index = super().store_transition(
            state, action, reward, next_state, done)
        # New samples do not have a priority yet
        # -> highest existing prio to make sure they are sampled early
        self.prio_memory[index] = self.current_max_prio
        # IDEA: why set to max? Seems arbitrary
        # Why not give higher priority to support this further?
        # Why not always manually add it to next batch to determine actual prio as fast as possible? (prio = inf)
        # Why not a smaller value? Too high prio has the risk that the next batch mainly consists of samples from a single episode, which breaks i.d.d property

        # IDEA: This way, the oldest transition is deleted, when the buffer is full. Why not delete the one with lowest prio?

        self._update_beta()

    def _update_max_prio(self, new_prios):
        """ We need to determine the current highest prio after prio
        updates to give it to new samples. """
        self.current_max_prio = max(self.current_max_prio, new_prios.max())
        # TODO: What if our current highest prio is deleted from the buffer? -> error!!!! because we do not check the rest of the buffer prios

    def _update_beta(self):
        self.beta += self.beta_increment
        self.beta = min(1.0, self.beta)

    def sample_random_batch(self, batch_size: int):
        """ Sample from the memory with higher priority samples sampled
        more often. """
        probabilities = self.prio_memory[:len(self)] ** self.alpha
        probabilities /= probabilities.sum()
        # IDEA: Always take newest sample here (test if it works better)
        batch_idxs = np.random.choice(
            len(self), batch_size, p=probabilities, replace=False)
        batch = self._get_batch(batch_idxs)

        batch_weights = (len(self) * probabilities[batch_idxs]) ** (-self.beta)
        # For stability reasons scaled downwards (instead of using the mean)
        # Essentially slows down learning!
        batch_weights /= batch_weights.max()  # TODO max!

        return batch, batch_idxs, batch_weights

    def update_priorities(self, new_batch_prios, batch_idxs):
        """ After training, we can update the new priorities of the batch
        (normally computed from the loss). """
        new_batch_prios += 1e-5
        # IDEA: Offset shall prevent a prio of zero. But why not use min(self.prio) or similar? Otherwise, we have the risk that the offset is to big

        self.prio_memory[batch_idxs] = new_batch_prios
        self._update_max_prio(new_prios=new_batch_prios)


class NStepBuffer():
    """ Buffer for squashing multiple transitions into one to speed up
    training. Used for example in DQN or REINFORCE. """

    def __init__(self, n_steps: int, gamma: float):
        self.n_steps = n_steps
        self._reset_deque()
        self.gamma = gamma

    def append_and_squash(self, obs, action, reward, next_obs, done):
        self.n_step_buffer.append((obs, action, reward, next_obs, done))
        if len(self) < self.n_steps and not done:
            # Not enough samples are collected
            return None

        if not done:
            return [self._squash_transition(from_idx=0)]

        # When the environment is done, we can calculate the exact q values
        # for all transitions
        transitions = [self._squash_transition(i) for i in
                       range(len(self.n_step_buffer))]
        self._reset_deque()

        return transitions

    def _squash_transition(self, from_idx: int):
        obs, action = self.n_step_buffer[from_idx][0:2]
        next_obs, done = self.n_step_buffer[-1][3:5]
        reward = self._discounting(from_idx=from_idx)
        return obs, action, reward, next_obs, done

    def _discounting(self, from_idx: int):
        """ Discount the intermediate rewards. """
        discounted_reward = sum(
            [self.n_step_buffer[idx][2] * (self.gamma**n)
             for n, idx in enumerate(range(from_idx, len(self.n_step_buffer)))])
        return discounted_reward

    def _reset_deque(self):
        self.n_step_buffer = deque(maxlen=self.n_steps)

    def __len__(self):
        return len(self.n_step_buffer)


class Episode():
    def __init__(self, gamma: float):
        self.gamma = gamma
        self.is_done = False
        self.obss = []
        self.actions = []
        self.rewards = []
        self.q_values = None
        self.last_obs = None

    def step(self, obs, action, reward, next_obs, done):
        self.obss.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        if done:
            self.last_obs = next_obs
            self.is_done = True
            self.q_values = self._calc_q_values()

    def __len__(self):
        return len(self.rewards)

    # def get_transition(self, idx):
    #     done = True if (self.is_done and idx == len(self)) else False
    #     obs = self.obss[idx]
    #     next_obs = self.obss[idx+1] if done else self.last_obs
    #     action = self.actions[idx]
    #     reward = self.rewards[idx]
    #     return obs, action, reward, next_obs, done

    def _calc_q_values(self):
        # Calculate q values for the episode
        discount_vector = self._discount_vector(len(self.rewards))
        reward_memory = np.array(self.rewards)

        # TODO: There must be a better solution than this
        # Create first entry manually
        q_values = [sum(reward_memory * discount_vector)]
        for idx in range(1, len(self.rewards)):
            q = sum(reward_memory[idx:] * discount_vector[:-idx])
            q_values.append(q)
        return q_values

    def _discount_vector(self, length: int):
        return self.gamma ** np.array(range(length))
