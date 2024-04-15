# -*- coding: utf-8 -*-
from __future__ import division
import math
import torch
from torch import nn
from torch.nn import functional as F


# Factorised NoisyLinear layer with bias
class NoisyLinear(nn.Module):
  def __init__(self, in_features, out_features, std_init=0.5):
    super(NoisyLinear, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.std_init = std_init
    self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
    self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
    self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
    self.bias_mu = nn.Parameter(torch.empty(out_features))
    self.bias_sigma = nn.Parameter(torch.empty(out_features))
    self.register_buffer('bias_epsilon', torch.empty(out_features))
    self.reset_parameters()
    self.reset_noise()

  def reset_parameters(self):
    mu_range = 1 / math.sqrt(self.in_features)
    self.weight_mu.data.uniform_(-mu_range, mu_range)
    self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
    self.bias_mu.data.uniform_(-mu_range, mu_range)
    self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

  def _scale_noise(self, size):
    x = torch.randn(size, device=self.weight_mu.device)
    return x.sign().mul_(x.abs().sqrt_())

  def reset_noise(self):
    epsilon_in = self._scale_noise(self.in_features)
    epsilon_out = self._scale_noise(self.out_features)
    self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
    self.bias_epsilon.copy_(epsilon_out)

  def forward(self, input):
    if self.training:
      return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
    else:
      return F.linear(input, self.weight_mu, self.bias_mu)


class C51(nn.Module):
  def __init__(self, args, action_space):
    super(C51, self).__init__()
    self.atoms = args.atoms
    self.action_space = action_space

    if args.architecture == 'canonical':
      self.convs = nn.Sequential(nn.Conv2d(args.history_length, 32, 8, stride=4, padding=0), nn.ReLU(),
                                 nn.Conv2d(32, 64, 4, stride=2, padding=0), nn.ReLU(),
                                 nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU())
      self.conv_output_size = 3136
    elif args.architecture == 'data-efficient':
      self.convs = nn.Sequential(nn.Conv2d(args.history_length, 32, 5, stride=5, padding=0), nn.ReLU(),
                                 nn.Conv2d(32, 64, 5, stride=5, padding=0), nn.ReLU())
      self.conv_output_size = 576
    self.fc_h_v = NoisyLinear(self.conv_output_size, args.hidden_size, std_init=args.noisy_std)
    self.fc_h_a = NoisyLinear(self.conv_output_size, args.hidden_size, std_init=args.noisy_std)
    self.fc_z_v = NoisyLinear(args.hidden_size, self.atoms, std_init=args.noisy_std)
    self.fc_z_a = NoisyLinear(args.hidden_size, action_space * self.atoms, std_init=args.noisy_std)

  def forward(self, x, log=False):
    x = self.convs(x)
    x = x.view(-1, self.conv_output_size)
    v = self.fc_z_v(F.relu(self.fc_h_v(x)))  # Value stream
    a = self.fc_z_a(F.relu(self.fc_h_a(x)))  # Advantage stream
    v, a = v.view(-1, 1, self.atoms), a.view(-1, self.action_space, self.atoms)
    q = v + a - a.mean(1, keepdim=True)  # Combine streams
    if log:  # Use log softmax for numerical stability
      q = F.log_softmax(q, dim=2)  # Log probabilities with action over second dimension
    else:
      q = F.softmax(q, dim=2)  # Probabilities with action over second dimension
    return q

  def reset_noise(self):
    for name, module in self.named_children():
      if 'fc' in name:
        module.reset_noise()


class DQN(nn.Module):
  def __init__(self, args, action_space):
    super(DQN, self).__init__()
    self.action_space = action_space

    if args.architecture == 'canonical':
      self.convs = nn.Sequential(nn.Conv2d(args.history_length, 32, 8, stride=4, padding=0), nn.ReLU(),
                                 nn.Conv2d(32, 64, 4, stride=2, padding=0), nn.ReLU(),
                                 nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU())
      self.conv_output_size = 3136
    elif args.architecture == 'data-efficient':
      self.convs = nn.Sequential(nn.Conv2d(args.history_length, 32, 5, stride=5, padding=0), nn.ReLU(),
                                 nn.Conv2d(32, 64, 5, stride=5, padding=0), nn.ReLU())
      self.conv_output_size = 576
    self.fc_h_v = NoisyLinear(self.conv_output_size, args.hidden_size, std_init=args.noisy_std)
    self.fc_h_a = NoisyLinear(self.conv_output_size, args.hidden_size, std_init=args.noisy_std)
    self.fc_z_v = NoisyLinear(args.hidden_size, 1, std_init=args.noisy_std)
    self.fc_z_a = NoisyLinear(args.hidden_size, action_space, std_init=args.noisy_std)

  def forward(self, x):
    x = self.convs(x)
    x = x.view(-1, self.conv_output_size)
    v = self.fc_z_v(F.relu(self.fc_h_v(x)))  # Value stream
    a = self.fc_z_a(F.relu(self.fc_h_a(x)))  # Advantage stream
    v, a = v.view(-1, 1), a.view(-1, self.action_space)
    q = v + a - a.mean(1, keepdim=True)  # Combine streams
    return q   # [batch, action_space]

  def reset_noise(self):
    for name, module in self.named_children():
      if 'fc' in name:
        module.reset_noise()


class mean_var_DQN(nn.Module):
  def __init__(self, args, action_space):
    super(mean_var_DQN, self).__init__()
    self.action_space = action_space

    if args.architecture == 'canonical':
      self.convs = nn.Sequential(nn.Conv2d(args.history_length, 32, 8, stride=4, padding=0), nn.ReLU(),
                                 nn.Conv2d(32, 64, 4, stride=2, padding=0), nn.ReLU(),
                                 nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU())
      self.conv_output_size = 3136
    elif args.architecture == 'data-efficient':
      self.convs = nn.Sequential(nn.Conv2d(args.history_length, 32, 5, stride=5, padding=0), nn.ReLU(),
                                 nn.Conv2d(32, 64, 5, stride=5, padding=0), nn.ReLU())
      self.conv_output_size = 576
    self.fc_h_v = NoisyLinear(self.conv_output_size, args.hidden_size, std_init=args.noisy_std)
    self.fc_h_a = NoisyLinear(self.conv_output_size, args.hidden_size, std_init=args.noisy_std)
    self.fc_z_v = NoisyLinear(args.hidden_size, 1, std_init=args.noisy_std)
    self.fc_z_a = NoisyLinear(args.hidden_size, action_space, std_init=args.noisy_std)

    self.fc_h_var_v = NoisyLinear(self.conv_output_size, args.hidden_size, std_init=args.noisy_std)
    self.fc_h_var_a = NoisyLinear(self.conv_output_size, args.hidden_size, std_init=args.noisy_std)
    self.fc_z_var_v = NoisyLinear(args.hidden_size, 1, std_init=args.noisy_std)
    self.fc_z_var_a = NoisyLinear(args.hidden_size, action_space, std_init=args.noisy_std)

  def forward(self, x):
    x = self.convs(x)
    x = x.view(-1, self.conv_output_size)
    v = self.fc_z_v(F.relu(self.fc_h_v(x)))  # Value stream
    a = self.fc_z_a(F.relu(self.fc_h_a(x)))  # Advantage stream
    v, a = v.view(-1, 1), a.view(-1, self.action_space)
    q = v + a - a.mean(1, keepdim=True)  # Combine streams

    var_v = self.fc_z_var_v(F.relu(self.fc_h_var_v(x)))  # Value stream
    var_a = self.fc_z_var_a(F.relu(self.fc_h_var_a(x)))  # Advantage stream
    var_v, var_a = var_v.view(-1, 1), var_a.view(-1, self.action_space)
    var_q = var_v + var_a - var_a.mean(1, keepdim=True)  # Combine streams

    return q, var_q   # [batch, action_space]

  def reset_noise(self):
    for name, module in self.named_children():
      if 'fc' in name:
        module.reset_noise()


class mean_var_DQN2(nn.Module):
  def __init__(self, args, action_space):
    super(mean_var_DQN2, self).__init__()
    self.action_space = action_space

    if args.architecture == 'canonical':
      self.convs = nn.Sequential(nn.Conv2d(args.history_length, 32, 8, stride=4, padding=0), nn.ReLU(),
                                 nn.Conv2d(32, 64, 4, stride=2, padding=0), nn.ReLU(),
                                 nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU())
      self.conv_output_size = 3136
    elif args.architecture == 'data-efficient':
      self.convs = nn.Sequential(nn.Conv2d(args.history_length, 32, 5, stride=5, padding=0), nn.ReLU(),
                                 nn.Conv2d(32, 64, 5, stride=5, padding=0), nn.ReLU())
      self.conv_output_size = 576
    self.fc_h_v = NoisyLinear(self.conv_output_size, args.hidden_size, std_init=args.noisy_std)
    self.fc_h_a = NoisyLinear(self.conv_output_size, args.hidden_size, std_init=args.noisy_std)
    self.fc_z_v = NoisyLinear(args.hidden_size, 1, std_init=args.noisy_std)
    self.fc_z_a = NoisyLinear(args.hidden_size, action_space, std_init=args.noisy_std)

    self.fc_z_var_v = NoisyLinear(args.hidden_size, 1, std_init=args.noisy_std)
    self.fc_z_var_a = NoisyLinear(args.hidden_size, action_space, std_init=args.noisy_std)

  def forward(self, x):
    x = self.convs(x)
    x = x.view(-1, self.conv_output_size)
    v = self.fc_z_v(F.relu(self.fc_h_v(x)))  # Value stream
    a = self.fc_z_a(F.relu(self.fc_h_a(x)))  # Advantage stream
    v, a = v.view(-1, 1), a.view(-1, self.action_space)
    q = v + a - a.mean(1, keepdim=True)  # Combine streams

    var_v = self.fc_z_var_v(F.relu(self.fc_h_v(x)))  # Value stream
    var_a = self.fc_z_var_a(F.relu(self.fc_h_a(x)))  # Advantage stream
    var_v, var_a = var_v.view(-1, 1), var_a.view(-1, self.action_space)
    var_q = var_v + var_a - var_a.mean(1, keepdim=True)  # Combine streams

    return q, var_q   # [batch, action_space]

  def reset_noise(self):
    for name, module in self.named_children():
      if 'fc' in name:
        module.reset_noise()


class vector_DQN(C51):
  '''Random slopes and output biases.'''
  def __init__(self, args, action_space, source_net=None):
    super(vector_DQN, self).__init__(args, action_space)
    self.Vmax = args.V_max

    if source_net is None:
      slope_min, slope_max = args.tnh_slope
      assert slope_max > slope_min, "Max slope must be larger than min slope."
      assert slope_min > 0, "All slopes must be strictly positive."
      # log_uniform sampling to ensure more even sampling between 1 > slope > slope_min
      slope_min, slope_max = math.log(slope_min), math.log(slope_max)
      tnh_slopes_v = torch.rand((1, self.atoms)) * (slope_max - slope_min) + slope_min
      tnh_slopes_a = torch.rand((1, self.atoms * self.action_space)) * (slope_max - slope_min) + slope_min
      tnh_slopes_v = torch.exp(tnh_slopes_v)
      tnh_slopes_a = torch.exp(tnh_slopes_a)

      if args.max_tnh_shift is not None:
        shifts_v = torch.rand((1, self.atoms)) * 2 * args.max_tnh_shift - args.max_tnh_shift
        shifts_a = torch.rand((1, self.atoms * self.action_space)) * 2 * args.max_tnh_shift - args.max_tnh_shift
      else:
        shifts_v = torch.zeros((1, self.atoms), dtype=torch.float16)
        shifts_a = torch.zeros((1, self.atoms * self.action_space), dtype=torch.float16)

      if args.Q_bias is not None:
        Q_biases = torch.rand((1, self.action_space, self.atoms)) * 2 * args.Q_bias - args.Q_bias
      else:
        Q_biases = torch.zeros((1, self.action_space, self.atoms))

    else:
      source_dict = source_net.state_dict()
      tnh_slopes_v = source_dict['tnh_slopes_v'].detach().clone()
      tnh_slopes_a = source_dict['tnh_slopes_a'].detach().clone()
      shifts_v = source_dict['shifts_v'].detach().clone()
      shifts_a = source_dict['shifts_a'].detach().clone()
      Q_biases = source_dict['Q_biases'].detach().clone()

    self.register_buffer('tnh_slopes_v', tnh_slopes_v)
    self.register_buffer('tnh_slopes_a', tnh_slopes_a)
    self.register_buffer('shifts_v', shifts_v)
    self.register_buffer('shifts_a', shifts_a)
    self.register_buffer('Q_biases', Q_biases)

  def forward(self, x, check_saturation=False):
    x = self.convs(x)
    x = x.view(-1, self.conv_output_size)
    v = self.fc_z_v(F.relu(self.fc_h_v(x)))  # Value stream
    a = self.fc_z_a(F.relu(self.fc_h_a(x)))  # Advantage stream
    v *= self.tnh_slopes_v
    a *= self.tnh_slopes_a

    if check_saturation:
      v = self.Vmax * torch.tanh(v)
      a = self.Vmax * torch.tanh(a)
      max_v = torch.max(torch.abs(v)).detach()
      max_a = torch.max(torch.abs(a)).detach()
      v += self.shifts_v
      a += self.shifts_a
    else:
      v = self.Vmax * torch.tanh(v) + self.shifts_v
      a = self.Vmax * torch.tanh(a) + self.shifts_a

    v, a = v.view(-1, 1, self.atoms), a.view(-1, self.action_space, self.atoms)
    q = v + a - a.mean(1, keepdim=True)  # Combine streams

    if check_saturation:
      return q, max_v, max_a
    else:
      return q
