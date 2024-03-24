# -*- coding: utf-8 -*-
from __future__ import division
import math
import torch
from torch import nn
from torch.nn import functional as F

class Geometric(nn.Module):
  def __init__(self, batch_size, atoms):
    super(Geometric, self).__init__()
    self.p = nn.Parameter(torch.Tensor(batch_size,1))
    p_expanded = self.p.expand(batch_size,atoms)
    sequence = torch.arange(0,atoms,dtype=self.p.dtype)
    self.r = p_expanded ** sequence

  def forward(self):
    return self.r

class fit_Geometric():
  def __init__(self, args):
    self.atoms = args.atoms
    self.Vmin = args.V_min
    self.Vmax = args.V_max
    self.support = torch.linspace(args.V_min, args.V_max, self.atoms).to(device=args.device)  # Support (range) of z
    self.delta_z = (args.V_max - args.V_min) / (self.atoms - 1)
    self.device = args.device
    self.lr = args.learning_rate

  def fit(self, means):
    batch_size = len(means)
    prob_param = Geometric(batch_size).to(self.device)
    optimiser = optim.Adam(prob_param, lr=self.lr)

    for _ in range(5):
      # compute means
      r = prob_param()
      curr_means = r*self.support.expand(batch_size,-1)
      curr_means = curr_means.sum(1)
      curr_means = curr_means/r.sum(1)
      loss = F.mse_loss(means, curr_means)

      prob_param.zero_grad()
      loss.backward()
      optimiser.step()

