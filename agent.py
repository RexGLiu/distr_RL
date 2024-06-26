# -*- coding: utf-8 -*-
from __future__ import division
import os
import numpy as np
import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F

from model import C51, DQN, mean_var_DQN, mean_var_DQN2, mean_var_DQNa, mean_var_skew_DQN, vector_DQN

class BaseAgent():
  # Resets noisy weights in all linear layers (of online net only)
  def reset_noise(self):
    self.online_net.reset_noise()

  # Acts with an ε-greedy policy (used for evaluation only)
  def act_e_greedy(self, state, epsilon=0.001):  # High ε can reduce evaluation scores drastically
    return np.random.randint(0, self.action_space) if np.random.random() < epsilon else self.act(state)

  def update_target_net(self):
    self.target_net.load_state_dict(self.online_net.state_dict())

  # Save checkpoint (model params, target params, optimiser state) on current device (don't move model between devices)
  def save(self, path, name='model.pth'):
    checkpoint = {"online_state_dict" : self.online_net.state_dict(),
                  "optimiser_state_dict" : self.optimiser.state_dict(),
                  "target_state_dict" : self.target_net.state_dict()
                  }

    torch.save(checkpoint, os.path.join(path, name))

  def train(self):
    self.online_net.train()

  def eval(self):
    self.online_net.eval()

class Rainbow(BaseAgent):
  def __init__(self, args, env):
    self.action_space = env.action_space()
    self.atoms = args.atoms
    self.Vmin = args.V_min
    self.Vmax = args.V_max
    self.support = torch.linspace(args.V_min, args.V_max, self.atoms).to(device=args.device)  # Support (range) of z
    self.delta_z = (args.V_max - args.V_min) / (self.atoms - 1)
    self.batch_size = args.batch_size
    self.n = args.multi_step
    self.discount = args.discount
    self.norm_clip = args.norm_clip
    self.track_grads = args.track_grads

    self.online_net = C51(args, self.action_space).to(device=args.device)
    self.target_net = C51(args, self.action_space).to(device=args.device)
    if args.model:  # Load pretrained model if provided
      if os.path.isfile(args.model):
        checkpoint = torch.load(args.model, map_location='cpu')  # Always load tensors onto CPU by default, will shift to GPU if necessary
      else:  # Raise error if incorrect model path provided
        raise FileNotFoundError(args.model)

      state_dict = checkpoint["online_state_dict"]
      if 'conv1.weight' in state_dict.keys():
        for old_key, new_key in (
        ('conv1.weight', 'convs.0.weight'), ('conv1.bias', 'convs.0.bias'), ('conv2.weight', 'convs.2.weight'),
        ('conv2.bias', 'convs.2.bias'), ('conv3.weight', 'convs.4.weight'), ('conv3.bias', 'convs.4.bias')):
          state_dict[new_key] = state_dict[old_key]  # Re-map state dict for old pretrained models
          del state_dict[old_key]  # Delete old keys for strict load_state_dict
      self.online_net.load_state_dict(state_dict)

      state_dict = checkpoint["target_state_dict"]
      if 'conv1.weight' in state_dict.keys():
        for old_key, new_key in (
        ('conv1.weight', 'convs.0.weight'), ('conv1.bias', 'convs.0.bias'), ('conv2.weight', 'convs.2.weight'),
        ('conv2.bias', 'convs.2.bias'), ('conv3.weight', 'convs.4.weight'), ('conv3.bias', 'convs.4.bias')):
          state_dict[new_key] = state_dict[old_key]  # Re-map state dict for old pretrained models
          del state_dict[old_key]  # Delete old keys for strict load_state_dict
      self.target_net.load_state_dict(state_dict)

      print("Loading pretrained model: " + args.model)

    self.online_net.train()
    self.update_target_net()
    self.target_net.train()
    for param in self.target_net.parameters():
      param.requires_grad = False

    self.optimiser = optim.Adam(self.online_net.parameters(), lr=args.learning_rate, eps=args.adam_eps)
    if args.model:
      optimiser_dict = checkpoint["optimiser_state_dict"]
      self.optimiser.load_state_dict(optimiser_dict)

  # Acts based on single state (no batch)
  def act(self, state):
    with torch.no_grad():
      return (self.online_net(state.unsqueeze(0)) * self.support).sum(2).argmax(1).item()

  def learn(self, mem):
    # Sample transitions
    idxs, states, actions, returns, next_states, nonterminals, weights = mem.sample(self.batch_size)

    # Calculate current state probabilities (online network noise already sampled)
    log_ps = self.online_net(states, log=True)  # Log probabilities log p(s_t, ·; θonline)
    log_ps_a = log_ps[range(self.batch_size), actions]  # log p(s_t, a_t; θonline)

    with torch.no_grad():
      # Calculate nth next state probabilities
      pns = self.online_net(next_states)  # Probabilities p(s_t+n, ·; θonline)
      dns = self.support.expand_as(pns) * pns  # Distribution d_t+n = (z, p(s_t+n, ·; θonline))
      argmax_indices_ns = dns.sum(2).argmax(1)  # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]
      self.target_net.reset_noise()  # Sample new target net noise
      pns = self.target_net(next_states)  # Probabilities p(s_t+n, ·; θtarget)
      pns_a = pns[range(self.batch_size), argmax_indices_ns]  # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)

      # Compute Tz (Bellman operator T applied to z)
      Tz = returns.unsqueeze(1) + nonterminals * (self.discount ** self.n) * self.support.unsqueeze(0)  # Tz = R^n + (γ^n)z (accounting for terminal states)
      Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)  # Clamp between supported values
      # Compute L2 projection of Tz onto fixed support z
      b = (Tz - self.Vmin) / self.delta_z  # b = (Tz - Vmin) / Δz
      l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
      # Fix disappearing probability mass when l = b = u (b is int)
      l[(u > 0) * (l == u)] -= 1
      u[(l < (self.atoms - 1)) * (l == u)] += 1

      # Distribute probability of Tz
      m = states.new_zeros(self.batch_size, self.atoms)
      offset = torch.linspace(0, ((self.batch_size - 1) * self.atoms), self.batch_size).unsqueeze(1).expand(self.batch_size, self.atoms).to(actions)
      m.view(-1).index_add_(0, (l + offset).view(-1), (pns_a * (u.float() - b)).view(-1))  # m_l = m_l + p(s_t+n, a*)(u - b)
      m.view(-1).index_add_(0, (u + offset).view(-1), (pns_a * (b - l.float())).view(-1))  # m_u = m_u + p(s_t+n, a*)(b - l)

    loss = -torch.sum(m * log_ps_a, 1)  # Cross-entropy loss (minimises DKL(m||p(s_t, a_t)))
    self.online_net.zero_grad()
    (weights * loss).mean().backward()  # Backpropagate importance-weighted minibatch loss
    pre_clip_norm = clip_grad_norm_(self.online_net.parameters(), self.norm_clip)  # Clip gradients by L2 norm
    post_clip_norm = clip_grad_norm_(self.online_net.parameters(), float('inf'))  # get grad norm before clipping
    self.optimiser.step()

    mem.update_priorities(idxs, loss.detach().cpu().numpy())  # Update priorities of sampled transitions

    losses = {"total loss" : loss.detach().cpu().numpy().mean()}
    if self.track_grads:
      grad_norms = {"pre_clip_norm": pre_clip_norm.detach().cpu().item(), "post_clip_norm" : post_clip_norm.detach().cpu().item()}
    else:
      grad_norms = {}
    return losses, grad_norms

  # Evaluates Q-value based on single state (no batch)
  def evaluate_q(self, state):
    with torch.no_grad():
      return (self.online_net(state.unsqueeze(0)) * self.support).sum(2).max(1)[0].item()

class Rainbow_DQN51(Rainbow):
  def learn(self, mem):
    # Sample transitions
    idxs, states, actions, returns, next_states, nonterminals, weights = mem.sample(self.batch_size)

    # Calculate current state probabilities (online network noise already sampled)
    ps = self.online_net(states)  # probabilities p(s_t, ·; θonline)
    ps_a = ps[range(self.batch_size), actions]  # p(s_t, a_t; θonline)
    ds_a = self.support.expand_as(ps_a) * ps_a  # Weight d_t = (z, p(s_t, ·; θonline))
    Q_a = ds_a.sum(1)  # Q-values for a_t

    with torch.no_grad():
      # Calculate nth next state probabilities
      pns = self.online_net(next_states)  # Probabilities p(s_t+n, ·; θonline)
      dns = self.support.expand_as(pns) * pns  # Distribution d_t+n = (z, p(s_t+n, ·; θonline))
      argmax_indices_ns = dns.sum(2).argmax(1)  # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]
      self.target_net.reset_noise()  # Sample new target net noise
      pns = self.target_net(next_states)  # Probabilities p(s_t+n, ·; θtarget)
      pns_a = pns[range(self.batch_size), argmax_indices_ns]  # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)
      dns_a = self.support.expand_as(pns_a) * pns_a

      Qns_a = dns_a.sum(1)
      Q_target = returns + nonterminals.view(-1) * (self.discount ** self.n) * Qns_a  # Q_target = R^n + (γ^n)Qns_a (accounting for terminal states)

      # Tz = returns.unsqueeze(1) + nonterminals * (self.discount ** self.n) * self.support.unsqueeze(0)  # Tz = R^n + (γ^n)z (accounting for terminal states)
      # dns_a = Tz * pns_a
      # Q_target = dns_a.sum(1)

    loss = F.mse_loss(Q_a, Q_target)
    self.online_net.zero_grad()
    (weights * loss).mean().backward()  # Backpropagate importance-weighted minibatch loss
    pre_clip_norm = clip_grad_norm_(self.online_net.parameters(), self.norm_clip)  # Clip gradients by L2 norm
    post_clip_norm = clip_grad_norm_(self.online_net.parameters(), float('inf'))  # get grad norm before clipping
    self.optimiser.step()

    mem.update_priorities(idxs, loss.detach().cpu().numpy())  # Update priorities of sampled transitions

    losses = {"total loss": loss.detach().cpu().numpy().mean()}
    if self.track_grads:
      grad_norms = {"pre_clip_norm": pre_clip_norm.detach().cpu().item(), "post_clip_norm" : post_clip_norm.detach().cpu().item()}
    else:
      grad_norms = {}
    return losses, grad_norms

class Rainbow_DQN(BaseAgent):
  def __init__(self, args, env):
    self.action_space = env.action_space()
    self.batch_size = args.batch_size
    self.n = args.multi_step
    self.discount = args.discount
    self.norm_clip = args.norm_clip
    self.track_grads = args.track_grads

    self.online_net = DQN(args, self.action_space).to(device=args.device)
    self.target_net = DQN(args, self.action_space).to(device=args.device)
    if args.model:  # Load pretrained model if provided
      if os.path.isfile(args.model):
        checkpoint = torch.load(args.model, map_location='cpu')  # Always load tensors onto CPU by default, will shift to GPU if necessary
      else:  # Raise error if incorrect model path provided
        raise FileNotFoundError(args.model)

      state_dict = checkpoint["online_state_dict"]
      if 'conv1.weight' in state_dict.keys():
        for old_key, new_key in (
        ('conv1.weight', 'convs.0.weight'), ('conv1.bias', 'convs.0.bias'), ('conv2.weight', 'convs.2.weight'),
        ('conv2.bias', 'convs.2.bias'), ('conv3.weight', 'convs.4.weight'), ('conv3.bias', 'convs.4.bias')):
          state_dict[new_key] = state_dict[old_key]  # Re-map state dict for old pretrained models
          del state_dict[old_key]  # Delete old keys for strict load_state_dict
      self.online_net.load_state_dict(state_dict)

      state_dict = checkpoint["target_state_dict"]
      if 'conv1.weight' in state_dict.keys():
        for old_key, new_key in (
        ('conv1.weight', 'convs.0.weight'), ('conv1.bias', 'convs.0.bias'), ('conv2.weight', 'convs.2.weight'),
        ('conv2.bias', 'convs.2.bias'), ('conv3.weight', 'convs.4.weight'), ('conv3.bias', 'convs.4.bias')):
          state_dict[new_key] = state_dict[old_key]  # Re-map state dict for old pretrained models
          del state_dict[old_key]  # Delete old keys for strict load_state_dict
      self.target_net.load_state_dict(state_dict)

      print("Loading pretrained model: " + args.model)

    self.online_net.train()
    self.update_target_net()
    self.target_net.train()
    for param in self.target_net.parameters():
      param.requires_grad = False

    self.optimiser = optim.Adam(self.online_net.parameters(), lr=args.learning_rate, eps=args.adam_eps)
    if args.model:
      optimiser_dict = checkpoint["optimiser_state_dict"]
      self.optimiser.load_state_dict(optimiser_dict)

  # Acts based on single state (no batch)
  def act(self, state):
    with torch.no_grad():
      return self.online_net(state.unsqueeze(0)).argmax(1).item()

  def learn(self, mem):
    # Sample transitions
    idxs, states, actions, returns, next_states, nonterminals, weights = mem.sample(self.batch_size)

    # Calculate current state-action values (online network noise already sampled)
    Qs = self.online_net(states)  # Q(s_t, ·; θonline)
    Qs_a = Qs[range(self.batch_size), actions]  # Q(s_t, a_t; θonline)

    with torch.no_grad():
      # Calculate nth next state probabilities
      Qns = self.online_net(next_states)  # Q(s_t+n, ·; θonline)
      argmax_indices_ns = Qns.argmax(1)  # Perform argmax action selection using online network: argmax_a[Q(s_t+n, a; θonline)]
      self.target_net.reset_noise()  # Sample new target net noise
      Qns = self.target_net(next_states)  # Q(s_t+n, ·; θtarget)
      Qns_a = Qns[range(self.batch_size), argmax_indices_ns]  # Double-Q probabilities Q(s_t+n, argmax_a[Q(s_t+n, a; θonline)]; θtarget)
      Q_target = returns + nonterminals.view(-1) * (self.discount ** self.n) * Qns_a  # Q_target = R^n + (γ^n)Qns_a (accounting for terminal states)

    loss = F.mse_loss(Qs_a, Q_target)
    self.online_net.zero_grad()
    (weights * loss).mean().backward()  # Backpropagate importance-weighted minibatch loss
    pre_clip_norm = clip_grad_norm_(self.online_net.parameters(), self.norm_clip)  # Clip gradients by L2 norm
    post_clip_norm = clip_grad_norm_(self.online_net.parameters(), float('inf'))  # get grad norm before clipping
    self.optimiser.step()

    mem.update_priorities(idxs, loss.detach().cpu().numpy())  # Update priorities of sampled transitions

    losses = {"total loss": loss.detach().cpu().numpy().mean()}
    if self.track_grads:
      grad_norms = {"pre_clip_norm": pre_clip_norm.detach().cpu().item(), "post_clip_norm" : post_clip_norm.detach().cpu().item()}
    else:
      grad_norms = {}

    return losses, grad_norms

  # Evaluates Q-value based on single state (no batch)
  def evaluate_q(self, state):
    with torch.no_grad():
      return self.online_net(state.unsqueeze(0)).max(1)[0].item()

class Rainbow_DQN51_v2(Rainbow):
  '''Encode mean with an entropic uniform distribution.'''
  def __init__(self, args, env):
    super(Rainbow_DQN51_v2, self).__init__(args, env)

    # cached row vectors used in construction of target distributions
    U_singlet_row = torch.zeros_like(self.support)
    L_singlet_row = torch.zeros_like(self.support)
    U_singlet_row[-1] = 1
    L_singlet_row[0] = 1
    self.U_singlet = U_singlet_row.expand((self.batch_size, -1))
    self.L_singlet = L_singlet_row.expand((self.batch_size, -1))

  def learn(self, mem):
    # Sample transitions
    idxs, states, actions, returns, next_states, nonterminals, weights = mem.sample(self.batch_size)

    # Calculate current state probabilities (online network noise already sampled)
    log_ps = self.online_net(states, log=True)  # Log probabilities log p(s_t, ·; θonline)
    log_ps_a = log_ps[range(self.batch_size), actions]  # log p(s_t, a_t; θonline)

    with torch.no_grad():
      # Calculate nth next state probabilities
      pns = self.online_net(next_states)  # Probabilities p(s_t+n, ·; θonline)
      dns = self.support.expand_as(pns) * pns  # Distribution d_t+n = (z, p(s_t+n, ·; θonline))
      argmax_indices_ns = dns.sum(2).argmax(1)  # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]
      self.target_net.reset_noise()  # Sample new target net noise
      pns = self.target_net(next_states)  # Probabilities p(s_t+n, ·; θtarget)
      pns_a = pns[range(self.batch_size), argmax_indices_ns]  # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)
      dns_a = self.support.expand_as(pns_a) * pns_a

      Qns_a = dns_a.sum(1)
      Q_target = returns + nonterminals.view(-1) * (self.discount ** self.n) * Qns_a  # Q_target = R^n + (γ^n)Qns_a (accounting for terminal states)
      Q_target = Q_target.view(-1,1)

      # construct target distribution m with mean Q_target
      m = torch.zeros_like(pns_a)
      support = self.support.expand_as(pns_a)

      u_mask = support > Q_target
      u_count = u_mask.sum(1, keepdim=True)
      U = (support * u_mask).sum(1, keepdim=True) / u_count  # arithmetic mean of support above Q_target

      l_mask = support <= Q_target
      l_count = l_mask.sum(1, keepdim=True)
      L = (support * l_mask).sum(1, keepdim=True) / l_count  # arithmetic mean of support below Q_target

      alpha = (U - Q_target)/(U - L)/l_count
      beta = (Q_target - L)/(U - L)/u_count
      alpha = alpha.expand_as(l_mask)
      beta = beta.expand_as(u_mask)
      m[l_mask] = alpha[l_mask]
      m[u_mask] = beta[u_mask]

      U_singlet = (self.support[-1] == Q_target).view(-1)
      L_singlet = (self.support[0] == Q_target).view(-1)
      m[U_singlet, :] = self.U_singlet[U_singlet, :]
      m[L_singlet, :] = self.L_singlet[L_singlet, :]

      # check = torch.abs( 1 - (m * self.support.expand_as(m)).sum(1)/Q_target.view(-1) ) < 1E-2
      # assert check.all(), f"{check}\n {(m * self.support.expand_as(m)).sum(1)[~check]}\n {Q_target.view(-1)[~check]}\n {m[~check,:]}\n {self.support}"

    loss = -torch.sum(m * log_ps_a, 1)  # Cross-entropy loss (minimises DKL(m||p(s_t, a_t)))
    self.online_net.zero_grad()
    (weights * loss).mean().backward()  # Backpropagate importance-weighted minibatch loss
    pre_clip_norm = clip_grad_norm_(self.online_net.parameters(), self.norm_clip)  # Clip gradients by L2 norm
    post_clip_norm = clip_grad_norm_(self.online_net.parameters(), float('inf'))  # get grad norm before clipping
    self.optimiser.step()

    mem.update_priorities(idxs, loss.detach().cpu().numpy())  # Update priorities of sampled transitions

    losses = {"total loss": loss.detach().cpu().numpy().mean()}
    if self.track_grads:
      grad_norms = {"pre_clip_norm": pre_clip_norm.detach().cpu().item(), "post_clip_norm" : post_clip_norm.detach().cpu().item()}
    else:
      grad_norms = {}
    return losses, grad_norms

class Rainbow_DQN51_v3(Rainbow):
  '''2-hot encoding of mean'''
  def learn(self, mem):
    # Sample transitions
    idxs, states, actions, returns, next_states, nonterminals, weights = mem.sample(self.batch_size)

    # Calculate current state probabilities (online network noise already sampled)
    log_ps = self.online_net(states, log=True)  # Log probabilities log p(s_t, ·; θonline)
    log_ps_a = log_ps[range(self.batch_size), actions]  # log p(s_t, a_t; θonline)

    with torch.no_grad():
      # Calculate nth next state probabilities
      pns = self.online_net(next_states)  # Probabilities p(s_t+n, ·; θonline)
      dns = self.support.expand_as(pns) * pns  # Distribution d_t+n = (z, p(s_t+n, ·; θonline))
      argmax_indices_ns = dns.sum(2).argmax(1)  # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]
      self.target_net.reset_noise()  # Sample new target net noise
      pns = self.target_net(next_states)  # Probabilities p(s_t+n, ·; θtarget)
      pns_a = pns[range(self.batch_size), argmax_indices_ns]  # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)
      dns_a = self.support.expand_as(pns_a) * pns_a

      Qns_a = dns_a.sum(1)
      Q_target = returns + nonterminals.view(-1) * (self.discount ** self.n) * Qns_a  # Q_target = R^n + (γ^n)Qns_a (accounting for terminal states)
      Q_target = Q_target.view(-1,1)

      # construct target distribution m with mean Q_target
      b = (Q_target - self.Vmin) / self.delta_z  # b = (Q_target - Vmin) / Δz
      l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
      # Fix disappearing probability mass when l = b = u (b is int)
      l[(u > 0) * (l == u)] -= 1
      u[(l < (self.atoms - 1)) * (l == u)] += 1

      L = self.support[l]
      U = self.support[u]

      beta = (Q_target - L)/(U-L)
      alpha = 1-beta

      m = torch.zeros_like(pns_a)
      m.scatter_(src=alpha, index=l, dim=1)
      m.scatter_(src=beta, index=u, dim=1)

      if (m.sum(1)!=1).any():
        print(f"Warning: target probabilities unnormalized, {m.sum(1)}")

    loss = -torch.sum(m * log_ps_a, 1)  # Cross-entropy loss (minimises DKL(m||p(s_t, a_t)))
    self.online_net.zero_grad()
    (weights * loss).mean().backward()  # Backpropagate importance-weighted minibatch loss
    pre_clip_norm = clip_grad_norm_(self.online_net.parameters(), self.norm_clip)  # Clip gradients by L2 norm
    post_clip_norm = clip_grad_norm_(self.online_net.parameters(), float('inf'))  # get grad norm before clipping
    self.optimiser.step()

    mem.update_priorities(idxs, loss.detach().cpu().numpy())  # Update priorities of sampled transitions

    losses = {"total loss": loss.detach().cpu().numpy().mean()}
    if self.track_grads:
      grad_norms = {"pre_clip_norm": pre_clip_norm.detach().cpu().item(), "post_clip_norm" : post_clip_norm.detach().cpu().item()}
    else:
      grad_norms = {}
    return losses, grad_norms

class Rainbow_mean_var_51(Rainbow):
  def __init__(self, args, env):
    super(Rainbow_mean_var_51, self).__init__(args, env)
    self.weight = args.weight

  def learn(self, mem):
    # Sample transitions
    idxs, states, actions, returns, next_states, nonterminals, weights = mem.sample(self.batch_size)

    # Calculate current state probabilities (online network noise already sampled)
    ps = self.online_net(states)  # probabilities p(s_t, ·; θonline)
    ps_a = ps[range(self.batch_size), actions]  # p(s_t, a_t; θonline)
    ds_a = self.support.expand_as(ps_a) * ps_a  # Weight d_t = (z, p(s_t, ·; θonline))
    Q_a = ds_a.sum(1)  # Q-values for a_t
    ds_a_var = (self.support.expand_as(ps_a) - Q_a.view(-1,1).expand_as(ps_a))**2 * ps_a  # Weight d_t = (z, p(s_t, ·; θonline))
    var_a = ds_a_var.sum(1)

    with torch.no_grad():
      # Calculate nth next state probabilities
      pns = self.online_net(next_states)  # Probabilities p(s_t+n, ·; θonline)
      dns = self.support.expand_as(pns) * pns  # Distribution d_t+n = (z, p(s_t+n, ·; θonline))
      argmax_indices_ns = dns.sum(2).argmax(1)  # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]
      self.target_net.reset_noise()  # Sample new target net noise
      pns = self.target_net(next_states)  # Probabilities p(s_t+n, ·; θtarget)
      pns_a = pns[range(self.batch_size), argmax_indices_ns]  # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)
      dns_a = self.support.expand_as(pns_a) * pns_a

      Qns_a = dns_a.sum(1)
      Q_target = returns + nonterminals.view(-1) * (self.discount ** self.n) * Qns_a  # Q_target = R^n + (γ^n)Qns_a (accounting for terminal states)

      dns_a_var = (self.support.expand_as(pns_a) - Qns_a.view(-1,1).expand_as(ps_a)) ** 2 * ps_a  # Weight d_t = (z, p(s_t, ·; θonline))
      var_ns_a = dns_a_var.sum(1)
      TD_err2 = (Q_a - Q_target) ** 2
      var_target = TD_err2 + nonterminals.view(-1) * (self.discount ** (2 * self.n)) * var_ns_a

    Q_loss = F.mse_loss(Q_a, Q_target)
    var_loss = F.mse_loss(var_a, var_target)

    self.online_net.zero_grad()
    if self.track_grads:
      (weights * Q_loss).mean().backward(retain_graph=True)
      Q_loss_grad = clip_grad_norm_(self.online_net.parameters(), float('inf'))
      self.online_net.zero_grad()

      (weights * var_loss).mean().backward(retain_graph=True)
      var_loss_grad = clip_grad_norm_(self.online_net.parameters(), float('inf'))
      self.online_net.zero_grad()

      total_grad = Q_loss_grad + var_loss_grad

    loss = Q_loss + self.weight * var_loss
    (weights * loss).mean().backward()  # Backpropagate importance-weighted minibatch loss
    pre_clip_norm = clip_grad_norm_(self.online_net.parameters(), self.norm_clip)  # Clip gradients by L2 norm
    post_clip_norm = clip_grad_norm_(self.online_net.parameters(), float('inf'))  # get grad norm before clipping
    self.optimiser.step()

    mem.update_priorities(idxs, Q_loss.detach().cpu().numpy())  # Update priorities of sampled transitions

    losses = {"total loss": loss.detach().cpu().numpy().mean(), "Q_loss" : Q_loss.detach().cpu().numpy().mean(),
              "var_loss" : self.weight * var_loss.detach().cpu().numpy().mean()}
    if self.track_grads:
      grad_norms = {"total_grad": total_grad.detach().cpu().item(), "Q_loss_grad" : Q_loss_grad.detach().cpu().item(),
                    "var_loss_grad" : var_loss_grad.detach().cpu().item(),
                    "pre_clip_norm": pre_clip_norm.detach().cpu().item(),
                    "post_clip_norm" : post_clip_norm.detach().cpu().item()}
    else:
      grad_norms = {}
    return losses, grad_norms

class Rainbow_mean_var_DQN(BaseAgent):
  '''Noisy two-stream architecture forking from the last conv layer.'''
  def __init__(self, args, env):
    self.action_space = env.action_space()
    self.batch_size = args.batch_size
    self.n = args.multi_step
    self.discount = args.discount
    self.norm_clip = args.norm_clip
    self.weight = args.weight
    self.TD_clip = args.reward_clip
    self.track_grads = args.track_grads

    self.online_net = mean_var_DQN(args, self.action_space).to(device=args.device)
    self.target_net = mean_var_DQN(args, self.action_space).to(device=args.device)
    if args.model:  # Load pretrained model if provided
      if os.path.isfile(args.model):
        checkpoint = torch.load(args.model, map_location='cpu')  # Always load tensors onto CPU by default, will shift to GPU if necessary
      else:  # Raise error if incorrect model path provided
        raise FileNotFoundError(args.model)

      state_dict = checkpoint["online_state_dict"]
      if 'conv1.weight' in state_dict.keys():
        for old_key, new_key in (
        ('conv1.weight', 'convs.0.weight'), ('conv1.bias', 'convs.0.bias'), ('conv2.weight', 'convs.2.weight'),
        ('conv2.bias', 'convs.2.bias'), ('conv3.weight', 'convs.4.weight'), ('conv3.bias', 'convs.4.bias')):
          state_dict[new_key] = state_dict[old_key]  # Re-map state dict for old pretrained models
          del state_dict[old_key]  # Delete old keys for strict load_state_dict
      self.online_net.load_state_dict(state_dict)

      state_dict = checkpoint["target_state_dict"]
      if 'conv1.weight' in state_dict.keys():
        for old_key, new_key in (
        ('conv1.weight', 'convs.0.weight'), ('conv1.bias', 'convs.0.bias'), ('conv2.weight', 'convs.2.weight'),
        ('conv2.bias', 'convs.2.bias'), ('conv3.weight', 'convs.4.weight'), ('conv3.bias', 'convs.4.bias')):
          state_dict[new_key] = state_dict[old_key]  # Re-map state dict for old pretrained models
          del state_dict[old_key]  # Delete old keys for strict load_state_dict
      self.target_net.load_state_dict(state_dict)

      print("Loading pretrained model: " + args.model)

    self.online_net.train()
    self.update_target_net()
    self.target_net.train()
    for param in self.target_net.parameters():
      param.requires_grad = False

    self.optimiser = optim.Adam(self.online_net.parameters(), lr=args.learning_rate, eps=args.adam_eps)
    if args.model:
      optimiser_dict = checkpoint["optimiser_state_dict"]
      self.optimiser.load_state_dict(optimiser_dict)

  # Acts based on single state (no batch)
  def act(self, state):
    with torch.no_grad():
      q, _ = self.online_net(state.unsqueeze(0))
      return q.argmax(1).item()

  def learn(self, mem):
    # Sample transitions
    idxs, states, actions, returns, next_states, nonterminals, weights = mem.sample(self.batch_size)

    # Calculate current state probabilities (online network noise already sampled)
    Qs, var_Qs = self.online_net(states)  # Q(s_t, ·; θonline)
    Qs_a, var_Qs_a = Qs[range(self.batch_size), actions], var_Qs[range(self.batch_size), actions]  # Q(s_t, a_t; θonline)

    with torch.no_grad():
      # Calculate nth next state probabilities
      Qns, _ = self.online_net(next_states)  # Q(s_t+n, ·; θonline)
      argmax_indices_ns = Qns.argmax(1)  # Perform argmax action selection using online network: argmax_a[Q(s_t+n, a; θonline)]
      self.target_net.reset_noise()  # Sample new target net noise
      Qns, var_Qns = self.target_net(next_states)  # Q(s_t+n, ·; θtarget)
      Qns_a, var_Qns_a = Qns[range(self.batch_size), argmax_indices_ns], var_Qns[range(self.batch_size), argmax_indices_ns]  # Double-Q probabilities Q(s_t+n, argmax_a[Q(s_t+n, a; θonline)]; θtarget)
      Q_target = returns + nonterminals.view(-1) * (self.discount ** self.n) * Qns_a  # Q_target = R^n + (γ^n)Qns_a (accounting for terminal states)

      TD_err = (Qs_a - Q_target) ** 2
      TD_err = torch.clamp(TD_err, max=self.TD_clip)  # clipping to control scale of update
      var_target = TD_err + nonterminals.view(-1) * (self.discount ** (2*self.n)) * var_Qns_a

    Q_loss = F.mse_loss(Qs_a, Q_target)
    var_loss = F.mse_loss(var_Qs_a, var_target)
    self.online_net.zero_grad()
    if self.track_grads:
      (weights * Q_loss).mean().backward(retain_graph=True)
      Q_loss_grad = clip_grad_norm_(self.online_net.parameters(), float('inf'))
      self.online_net.zero_grad()

      (weights * self.weight * var_loss).mean().backward(retain_graph=True)
      var_loss_grad = clip_grad_norm_(self.online_net.parameters(), float('inf'))
      self.online_net.zero_grad()

      total_grad = Q_loss_grad + var_loss_grad

    loss = Q_loss + self.weight * var_loss
    (weights * loss).mean().backward()  # Backpropagate importance-weighted minibatch loss
    pre_clip_norm = clip_grad_norm_(self.online_net.parameters(), self.norm_clip)  # Clip gradients by L2 norm
    post_clip_norm = clip_grad_norm_(self.online_net.parameters(), float('inf'))  # get grad norm before clipping
    self.optimiser.step()

    mem.update_priorities(idxs, Q_loss.detach().cpu().numpy())  # Update priorities of sampled transitions

    losses = {"total loss": loss.detach().cpu().numpy().mean(), "Q_loss" : Q_loss.detach().cpu().numpy().mean(),
              "var_loss" : self.weight * var_loss.detach().cpu().numpy().mean()}
    if self.track_grads:
      grad_norms = {"total_grad": total_grad.detach().cpu().item(), "Q_loss_grad" : Q_loss_grad.detach().cpu().item(),
                    "var_loss_grad" : var_loss_grad.detach().cpu().item(),
                    "pre_clip_norm": pre_clip_norm.detach().cpu().item(),
                    "post_clip_norm" : post_clip_norm.detach().cpu().item()}
    else:
      grad_norms = {}
    return losses, grad_norms

  # Evaluates Q-value based on single state (no batch)
  def evaluate_q(self, state):
    with torch.no_grad():
      q, _ = self.online_net(state.unsqueeze(0))
      return q.max(1)[0].item()

class Rainbow_mean_var_DQN2(BaseAgent):
  '''Noisy two-stream architecture forking from final hidden layer.'''
  def __init__(self, args, env):
    self.action_space = env.action_space()
    self.batch_size = args.batch_size
    self.n = args.multi_step
    self.discount = args.discount
    self.norm_clip = args.norm_clip
    self.weight = args.weight
    self.TD_clip = args.reward_clip
    self.track_grads = args.track_grads

    self.online_net = mean_var_DQN2(args, self.action_space).to(device=args.device)
    self.target_net = mean_var_DQN2(args, self.action_space).to(device=args.device)
    if args.model:  # Load pretrained model if provided
      if os.path.isfile(args.model):
        checkpoint = torch.load(args.model, map_location='cpu')  # Always load tensors onto CPU by default, will shift to GPU if necessary
      else:  # Raise error if incorrect model path provided
        raise FileNotFoundError(args.model)

      state_dict = checkpoint["online_state_dict"]
      if 'conv1.weight' in state_dict.keys():
        for old_key, new_key in (
        ('conv1.weight', 'convs.0.weight'), ('conv1.bias', 'convs.0.bias'), ('conv2.weight', 'convs.2.weight'),
        ('conv2.bias', 'convs.2.bias'), ('conv3.weight', 'convs.4.weight'), ('conv3.bias', 'convs.4.bias')):
          state_dict[new_key] = state_dict[old_key]  # Re-map state dict for old pretrained models
          del state_dict[old_key]  # Delete old keys for strict load_state_dict
      self.online_net.load_state_dict(state_dict)

      state_dict = checkpoint["target_state_dict"]
      if 'conv1.weight' in state_dict.keys():
        for old_key, new_key in (
        ('conv1.weight', 'convs.0.weight'), ('conv1.bias', 'convs.0.bias'), ('conv2.weight', 'convs.2.weight'),
        ('conv2.bias', 'convs.2.bias'), ('conv3.weight', 'convs.4.weight'), ('conv3.bias', 'convs.4.bias')):
          state_dict[new_key] = state_dict[old_key]  # Re-map state dict for old pretrained models
          del state_dict[old_key]  # Delete old keys for strict load_state_dict
      self.target_net.load_state_dict(state_dict)

      print("Loading pretrained model: " + args.model)

    self.online_net.train()
    self.update_target_net()
    self.target_net.train()
    for param in self.target_net.parameters():
      param.requires_grad = False

    self.optimiser = optim.Adam(self.online_net.parameters(), lr=args.learning_rate, eps=args.adam_eps)
    if args.model:
      optimiser_dict = checkpoint["optimiser_state_dict"]
      self.optimiser.load_state_dict(optimiser_dict)

  # Acts based on single state (no batch)
  def act(self, state):
    with torch.no_grad():
      q, _ = self.online_net(state.unsqueeze(0))
      return q.argmax(1).item()

  def learn(self, mem):
    # Sample transitions
    idxs, states, actions, returns, next_states, nonterminals, weights = mem.sample(self.batch_size)

    # Calculate current state probabilities (online network noise already sampled)
    Qs, var_Qs = self.online_net(states)  # Q(s_t, ·; θonline)
    Qs_a, var_Qs_a = Qs[range(self.batch_size), actions], var_Qs[range(self.batch_size), actions]  # Q(s_t, a_t; θonline)

    with torch.no_grad():
      # Calculate nth next state probabilities
      Qns, _ = self.online_net(next_states)  # Q(s_t+n, ·; θonline)
      argmax_indices_ns = Qns.argmax(1)  # Perform argmax action selection using online network: argmax_a[Q(s_t+n, a; θonline)]
      self.target_net.reset_noise()  # Sample new target net noise
      Qns, var_Qns = self.target_net(next_states)  # Q(s_t+n, ·; θtarget)
      Qns_a, var_Qns_a = Qns[range(self.batch_size), argmax_indices_ns], var_Qns[range(self.batch_size), argmax_indices_ns]  # Double-Q probabilities Q(s_t+n, argmax_a[Q(s_t+n, a; θonline)]; θtarget)
      Q_target = returns + nonterminals.view(-1) * (self.discount ** self.n) * Qns_a  # Q_target = R^n + (γ^n)Qns_a (accounting for terminal states)

      TD_err = (Qs_a - Q_target) ** 2
      TD_err = torch.clamp(TD_err, max=self.TD_clip)  # clipping to control scale of update
      var_target = TD_err + nonterminals.view(-1) * (self.discount ** (2*self.n)) * var_Qns_a

    Q_loss = F.mse_loss(Qs_a, Q_target)
    var_loss = F.mse_loss(var_Qs_a, var_target)
    self.online_net.zero_grad()
    if self.track_grads:
      (weights * Q_loss).mean().backward(retain_graph=True)
      Q_loss_grad = clip_grad_norm_(self.online_net.parameters(), float('inf'))
      self.online_net.zero_grad()

      (weights * self.weight * var_loss).mean().backward(retain_graph=True)
      var_loss_grad = clip_grad_norm_(self.online_net.parameters(), float('inf'))
      self.online_net.zero_grad()

      total_grad = Q_loss_grad + var_loss_grad

    loss = Q_loss + self.weight * var_loss
    (weights * loss).mean().backward()  # Backpropagate importance-weighted minibatch loss
    pre_clip_norm = clip_grad_norm_(self.online_net.parameters(), self.norm_clip)  # Clip gradients by L2 norm
    post_clip_norm = clip_grad_norm_(self.online_net.parameters(), float('inf'))  # get grad norm before clipping
    self.optimiser.step()

    mem.update_priorities(idxs, Q_loss.detach().cpu().numpy())  # Update priorities of sampled transitions

    losses = {"total loss": loss.detach().cpu().numpy().mean(), "Q_loss" : Q_loss.detach().cpu().numpy().mean(),
              "var_loss" : self.weight * var_loss.detach().cpu().numpy().mean()}
    if self.track_grads:
      grad_norms = {"total_grad": total_grad.detach().cpu().item(), "Q_loss_grad" : Q_loss_grad.detach().cpu().item(),
                    "var_loss_grad" : var_loss_grad.detach().cpu().item(),
                    "pre_clip_norm": pre_clip_norm.detach().cpu().item(),
                    "post_clip_norm" : post_clip_norm.detach().cpu().item()}
    else:
      grad_norms = {}
    return losses, grad_norms

  # Evaluates Q-value based on single state (no batch)
  def evaluate_q(self, state):
    with torch.no_grad():
      q, _ = self.online_net(state.unsqueeze(0))
      return q.max(1)[0].item()

class Rainbow_vec_DQN(BaseAgent):
  def __init__(self, args, env):
    self.action_space = env.action_space()
    self.atoms = args.atoms
    self.Vmin = args.V_min
    self.Vmax = args.V_max
    self.batch_size = args.batch_size
    self.n = args.multi_step
    self.discount = args.discount
    self.norm_clip = args.norm_clip
    self.track_grads = args.track_grads
    self.target_mean = args.vec_target_mean

    self.online_net = vector_DQN(args, self.action_space).to(device=args.device)
    self.target_net = vector_DQN(args, self.action_space, self.online_net).to(device=args.device)
    if args.model:  # Load pretrained model if provided
      if os.path.isfile(args.model):
        checkpoint = torch.load(args.model, map_location='cpu')  # Always load tensors onto CPU by default, will shift to GPU if necessary
      else:  # Raise error if incorrect model path provided
        raise FileNotFoundError(args.model)

      state_dict = checkpoint["online_state_dict"]
      if 'conv1.weight' in state_dict.keys():
        for old_key, new_key in (
        ('conv1.weight', 'convs.0.weight'), ('conv1.bias', 'convs.0.bias'), ('conv2.weight', 'convs.2.weight'),
        ('conv2.bias', 'convs.2.bias'), ('conv3.weight', 'convs.4.weight'), ('conv3.bias', 'convs.4.bias')):
          state_dict[new_key] = state_dict[old_key]  # Re-map state dict for old pretrained models
          del state_dict[old_key]  # Delete old keys for strict load_state_dict
      self.online_net.load_state_dict(state_dict)

      state_dict = checkpoint["target_state_dict"]
      if 'conv1.weight' in state_dict.keys():
        for old_key, new_key in (
        ('conv1.weight', 'convs.0.weight'), ('conv1.bias', 'convs.0.bias'), ('conv2.weight', 'convs.2.weight'),
        ('conv2.bias', 'convs.2.bias'), ('conv3.weight', 'convs.4.weight'), ('conv3.bias', 'convs.4.bias')):
          state_dict[new_key] = state_dict[old_key]  # Re-map state dict for old pretrained models
          del state_dict[old_key]  # Delete old keys for strict load_state_dict
      self.target_net.load_state_dict(state_dict)

      print("Loading pretrained model: " + args.model)

    self.Q_biases = self.online_net.Q_biases
    self.Q_biases_expanded = self.Q_biases.expand(self.batch_size, self.action_space, self.atoms)

    self.online_net.train()
    self.update_target_net()
    self.target_net.train()
    for param in self.target_net.parameters():
      param.requires_grad = False

    self.optimiser = optim.Adam(self.online_net.parameters(), lr=args.learning_rate, eps=args.adam_eps)
    if args.model:
      optimiser_dict = checkpoint["optimiser_state_dict"]
      self.optimiser.load_state_dict(optimiser_dict)

  def act(self, state):
    with torch.no_grad():
      return (self.online_net(state.unsqueeze(0)) - self.Q_biases).mean(2).argmax(1).item()

  def learn(self, mem):
    # Sample transitions
    idxs, states, actions, returns, next_states, nonterminals, weights = mem.sample(self.batch_size)

    # Calculate current state-action values (online network noise already sampled)
    Qs, max_v, max_a = self.online_net(states, check_saturation=True)  # Q(s_t, ·; θonline)
    Qs_a = Qs[range(self.batch_size), actions, :]  # Q(s_t, a_t; θonline)
    max_v, max_a = max_v.cpu().item(), max_a.cpu().item()

    with torch.no_grad():
      # Calculate target Q-values
      Q_vec_target = self.online_net(next_states) - self.Q_biases
      Qns = Q_vec_target.mean(dim=-1)
      argmax_indices_ns = Qns.argmax(1)  # Perform argmax action selection using online network: argmax_a[Q(s_t+n, a; θonline)]
      self.target_net.reset_noise()  # Sample new target net noise

      Q_vec_target = self.target_net(next_states) - self.Q_biases # Q(s_t+n, ·; θtarget)
      Qns_a = Q_vec_target[range(self.batch_size), argmax_indices_ns, :]  # Double-Q vals Q(s_t+n, argmax_a[Q(s_t+n, a; θonline)]; θtarget)
      biases = self.Q_biases_expanded[range(self.batch_size), argmax_indices_ns, :]
      if self.target_mean:
        Qns_a = Qns_a.mean(-1, keepdim=True)
      Q_target = returns.unsqueeze(1) + nonterminals * (self.discount ** self.n) * Qns_a + biases # Q_target = R^n + (γ^n)Qns_a (accounting for terminal states)

    loss = F.mse_loss(Qs_a, Q_target)
    self.online_net.zero_grad()
    (weights * loss).mean().backward()  # Backpropagate importance-weighted minibatch loss
    pre_clip_norm = clip_grad_norm_(self.online_net.parameters(), self.norm_clip)  # Clip gradients by L2 norm
    post_clip_norm = clip_grad_norm_(self.online_net.parameters(), float('inf'))  # get grad norm before clipping
    self.optimiser.step()

    mem.update_priorities(idxs, loss.detach().cpu().numpy())  # Update priorities of sampled transitions

    losses = {"total loss": loss.detach().cpu().numpy().mean(), "max v output": max_v, "max a output": max_a}
    if self.track_grads:
      grad_norms = {"pre_clip_norm": pre_clip_norm.detach().cpu().item(), "post_clip_norm" : post_clip_norm.detach().cpu().item()}
    else:
      grad_norms = {}
    return losses, grad_norms

  # Evaluates Q-value based on single state (no batch)
  def evaluate_q(self, state):
    with torch.no_grad():
      return (self.online_net(state.unsqueeze(0)) - self.Q_biases).mean(2).max(1)[0].item()
