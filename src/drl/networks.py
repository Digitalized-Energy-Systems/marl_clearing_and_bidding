"""
Standard and simple neural net definition for DQN. Uses no convolutional layers, layer norm and relu activation.

"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal


class DqNet(nn.Module):
    def __init__(self, n_obs: int, fc_dims: list, n_act: int, learning_rate: float,
                 optimizer='Adam', layer_norm=True):
        super().__init__()
        self.n_obs = n_obs
        self.fc_dims = fc_dims
        self.n_act = n_act
        self.loss = nn.MSELoss()

        self._init_linear_layers(n_obs, fc_dims, n_act)

        self.layer_norm = layer_norm
        self.lns = nn.ModuleList(
            [nn.LayerNorm(n_neurons) for n_neurons in fc_dims])

        self._init_weights()

        self.optimizer = getattr(optim, optimizer)(
            self.parameters(), lr=learning_rate)

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

        self.to(self.device)

    def _init_linear_layers(self, n_obs, fc_dims, n_act):
        fcs = [nn.Linear(n_obs, fc_dims[0])]
        fcs += [nn.Linear(fc_dims[i], fc_dims[i + 1])
                for i in range(len(fc_dims) - 1)]
        self.fcs = nn.ModuleList(fcs)

        self.value = nn.Linear(fc_dims[-1], n_act)

    def _init_weights(self):
        for idx, fc in enumerate(self.fcs):
            f = 1.0 / np.sqrt(fc.weight.data.size()[0])
            self.fcs[idx].weight.data.uniform_(-f, f)
            self.fcs[idx].bias.data.uniform_(-f, f)

        f = 1.0 / np.sqrt(self.value.weight.data.size()[0])
        self.value.weight.data.uniform_(-f, f)
        self.value.bias.data.uniform_(-f, f)

    def forward(self, obs):
        output = obs
        for fc, ln in zip(self.fcs, self.lns):
            output = fc(output)
            if self.layer_norm:
                output = ln(output)
            output = F.relu(output)

        output = self.value(output)

        return output


class NoisyDqNet(DqNet):
    def _init_linear_layers(self, n_obs, fc_dims, n_act):
        fcs = [NoisyLinear(n_obs, fc_dims[0])]
        fcs += [NoisyLinear(fc_dims[i], fc_dims[i + 1])
                for i in range(len(fc_dims) - 1)]
        self.fcs = nn.ModuleList(fcs)

        self.value = NoisyLinear(fc_dims[-1], n_act)


class NoisyLinear(nn.Linear):
    """ Linear layer with some learnable added noise for exploration.
    https://arxiv.org/pdf/1706.10295.pdf
    Code partly taken from: Lapan - Deep Reinforcement Learning Hands on """

    def __init__(self, n_in, n_out, sigma_start=0.017, bias=True):
        super().__init__(n_in, n_out, bias=bias)

        weight = torch.full(size=(n_out, n_in), fill_value=sigma_start)
        self.sigma_weight = nn.Parameter(weight)

        # Epsilon: Some random vector/matrix
        self.register_buffer('epsilon_weight', torch.zeros(n_out, n_in))

        if bias:
            bias = torch.full(size=(n_out,), fill_value=sigma_start)
            self.sigma_bias = nn.Parameter(bias)
            self.register_buffer('epsilon_bias', torch.zeros(n_out))

        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, obs):
        self.epsilon_weight.normal_()
        weight = self.weight + self.sigma_weight * self.epsilon_weight.data

        bias = self.bias
        if bias is not None:
            self.epsilon_bias.normal_()
            bias = self.bias + self.sigma_bias * self.bias.data

        return F.linear(obs, weight, bias)


class DuelingNet(DqNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO


class ReinforceNet(DqNet):
    """ The same net for DQN can be reused. Inheritance is simply used for
    renaming to avoid confusion. """
    pass


class DiscreteA2CNet(nn.Module):
    def __init__(self, n_obs: int, n_act: int, learning_rate: float,
                 layer_norm=True):
        super().__init__()
        # TODO: use soft-coded net dimensions!
        self.body = nn.Sequential(
            StandardLayer(n_obs, 512), layer_norm)
        self.actor = nn.Sequential(
            StandardLayer(512, 512), nn.Linear(512, n_act), layer_norm)
        self.critic = nn.Sequential(
            StandardLayer(512, 512), nn.Linear(512, 1), layer_norm)

        self.critic_loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

        self.to(self.device)

    def _get_intermediate(self, obs):
        return self.body(obs)

    def forward(self, obs):
        obs = self._get_intermediate(obs)
        return self.actor(obs)

    def full_forward(self, obs):
        obs = self._get_intermediate(obs)
        return self.actor(obs), self.critic(obs)


class ContinuousA2CNet(nn.Module):
    def __init__(self, n_obs: int, n_act: int, learning_rate: float, layer_norm=True):
        super().__init__()
        # TODO: use soft-coded net dimensions!
        self.body = nn.Sequential(
            StandardLayer(n_obs, 512, layer_norm))
        self.actor_mu = nn.Sequential(
            StandardLayer(512, 512, layer_norm), nn.Linear(512, 1), nn.Tanh())
        self.actor_var = nn.Sequential(
            StandardLayer(512, 512, layer_norm), nn.Linear(512, 1), nn.Softplus())
        self.critic = nn.Sequential(
            StandardLayer(512, 512, layer_norm), nn.Linear(512, 1))

        self.critic_loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

        self.to(self.device)

    def _get_intermediate(self, obs):
        return self.body(obs)

    def forward(self, obs):
        obs = self._get_intermediate(obs)
        return self.actor_mu(obs), self.actor_var(obs)

    def full_forward(self, obs):
        obs = self._get_intermediate(obs)
        return (self.actor_mu(obs), self.actor_var(obs)), self.critic(obs)


class DDPGActorNet(DqNet):
    def __init__(self, n_obs, fc_dims, n_act, learning_rate,
                 output_activation='tanh', optimizer='Adam',
                 *args, **kwargs):
        self.output_activation = output_activation
        super().__init__(
            n_obs, fc_dims, n_act, learning_rate, optimizer, *args, **kwargs)

    def _init_weights(self):
        super()._init_weights()
        f3 = 0.003
        self.value.weight.data.uniform_(-f3, f3)
        self.value.bias.data.uniform_(-f3, f3)

    def forward(self, obs):
        output = super().forward(obs)
        # TODO: How to deal with unbounded action spaces?
        return getattr(torch, self.output_activation)(output)


class DDPGCriticNet(nn.Module):
    def __init__(self, n_obs: int, n_act: int, learning_rate: float,
                 fc_dims: list=[300], n_rewards=1, obs_neurons=400,
                 layer_norm=True):
        super().__init__()

        self.learning_rate = learning_rate
        self.loss = nn.MSELoss()

        self.obs_layer = StandardLayer(n_obs, obs_neurons, layer_norm)

        fc_dims = [obs_neurons + n_act] + fc_dims + [n_rewards]

        self.layers = nn.Sequential(
            *([StandardLayer(fc_dims[i], fc_dims[i + 1], layer_norm)
               for i in range(len(fc_dims) - 2)]
              + [nn.Linear(fc_dims[-2], fc_dims[-1])]))

        f3 = 0.003
        self.layers[-1].weight.data.uniform_(-f3, f3)
        self.layers[-1].bias.data.uniform_(-f3, f3)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

        self.to(self.device)

    def forward(self, obs, action):
        obs = self.obs_layer(obs)
        return self.layers(torch.cat([obs, action], dim=1))


class SACActorNet(DDPGActorNet):
    def __init__(self, n_obs, fc_dims, n_act, learning_rate,
                 output_activation='tanh', optimizer='Adam', layer_norm=True,
                 *args, **kwargs):

        super().__init__(
            n_obs, fc_dims, n_act, learning_rate, optimizer, *args, **kwargs)
        self.normal = Normal(0, 1)
        self.log_std_out = StandardLayer(fc_dims[-1], n_act, layer_norm)

    def inference(self, obs):
        obs = obs
        for fc, ln in zip(self.fcs, self.lns):
            obs = fc(obs)
            obs = ln(obs)
            obs = F.relu(obs)

        mean = self.value(obs)
        log_std = torch.clamp(self.log_std_out(obs), -20,
                              20)  # TODO: A bit random?!

        return mean, log_std

    def forward(self, obs, act_only=False, deterministic=False):
        mean, log_std = self.inference(obs)
        if deterministic and act_only:
            # For testing
            return torch.tanh(mean)

        std = log_std.exp()
        normal_sample = self.normal.sample().to(self.device)
        act_ = mean + std * normal_sample
        act = torch.tanh(act_)

        if act_only:
            # For interaction with environment
            return act

        # For training
        log_prob = (Normal(mean, std).log_prob(act_)
                    - torch.log(1 - act**2 + 1e-6))

        return log_prob, act


class StandardLayer(nn.Module):
    def __init__(self, n_in, n_out, layer_norm=True):
        super().__init__()
        if layer_norm:
            self.layer = nn.Sequential(
                nn.Linear(n_in, n_out), nn.LayerNorm(n_out), nn.ReLU())
        else:
            self.layer = nn.Sequential(nn.Linear(n_in, n_out), nn.ReLU())

    def forward(self, input):
        return self.layer(input)
