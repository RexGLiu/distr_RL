# -*- coding: utf-8 -*-
from __future__ import division
import argparse
import bz2
from datetime import datetime
import os
import pickle

import atari_py
import numpy as np
import torch
from tqdm import trange

from agent import Rainbow, Rainbow_DQN51, Rainbow_DQN, Rainbow_mean_var_DQN, Rainbow_mean_var_DQN2, \
  Rainbow_mean_var_DQNa, Rainbow_mean_var_51, Rainbow_vec_DQN, \
  Rainbow_DQN51_ent, Rainbow_DQN51_cross_ent, Rainbow_DQN51_v2, Rainbow_DQN51_v3
from env import Env
from memory import ReplayMemory
from test import test

import wandb


# Note that hyperparameters may originally be reported in ATARI game frames instead of agent steps
parser = argparse.ArgumentParser(description='Rainbow')
parser.add_argument('--id', type=str, default='default', help='Experiment ID')
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--game', type=str, default='space_invaders', choices=atari_py.list_games(), help='ATARI game')
parser.add_argument('--algo', type=str, default='Rainbow', choices=['Rainbow', 'Rainbow_DQN51', 'Rainbow_DQN', 'Rainbow_mean_var_DQN', 'Rainbow_mean_var_DQN2',
                                                                    'Rainbow_mean_var_DQNa', 'Rainbow_mean_var_51', 'Rainbow_vec_DQN',
                                                                    'Rainbow_DQN51_ent', 'Rainbow_DQN51_cross_ent', 'Rainbow_DQN51_v2',
                                                                    'Rainbow_DQN51_v3'], help='Which RL algorithm to run')
parser.add_argument('--T-max', type=int, default=int(50e6), metavar='STEPS', help='Number of training steps (4x number of frames)')
parser.add_argument('--max-episode-length', type=int, default=int(108e3), metavar='LENGTH', help='Max episode length in game frames (0 to disable)')
parser.add_argument('--history-length', type=int, default=4, metavar='T', help='Number of consecutive states processed')
parser.add_argument('--architecture', type=str, default='canonical', choices=['canonical', 'data-efficient'], metavar='ARCH', help='Network architecture')
parser.add_argument('--hidden-size', type=int, default=512, metavar='SIZE', help='Network hidden size')
parser.add_argument('--noisy-std', type=float, default=0.1, metavar='σ', help='Initial standard deviation of noisy linear layers')
parser.add_argument('--atoms', type=int, default=51, metavar='C', help='Discretised size of value distribution')
parser.add_argument('--V-min', type=float, default=-10, metavar='V', help='Minimum of value distribution support')
parser.add_argument('--V-max', type=float, default=10, metavar='V', help='Maximum of value distribution support')
parser.add_argument('--model', type=str, metavar='PARAMS', help='Pretrained model (state dict)')
parser.add_argument('--memory-capacity', type=int, default=int(1e6), metavar='CAPACITY', help='Experience replay memory capacity')
parser.add_argument('--replay-frequency', type=int, default=4, metavar='k', help='Frequency of sampling from memory')
parser.add_argument('--priority-exponent', type=float, default=0.5, metavar='ω', help='Prioritised experience replay exponent (originally denoted α)')
parser.add_argument('--priority-weight', type=float, default=0.4, metavar='β', help='Initial prioritised experience replay importance sampling weight')
parser.add_argument('--multi-step', type=int, default=3, metavar='n', help='Number of steps for multi-step return')
parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discount factor')
parser.add_argument('--target-update', type=int, default=int(8e3), metavar='τ', help='Number of steps after which to update target network')
parser.add_argument('--reward-clip', type=int, default=1, metavar='VALUE', help='Reward clipping (0 to disable)')
parser.add_argument('--learning-rate', type=float, default=0.0000625, metavar='η', help='Learning rate')
parser.add_argument('--adam-eps', type=float, default=1.5e-4, metavar='ε', help='Adam epsilon')
parser.add_argument('--batch-size', type=int, default=32, metavar='SIZE', help='Batch size')
parser.add_argument('--norm-clip', type=float, default=10, metavar='NORM', help='Max L2 norm for gradient clipping')
parser.add_argument('--learn-start', type=int, default=int(20e3), metavar='STEPS', help='Number of steps before starting training')
parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
parser.add_argument('--evaluation-interval', type=int, default=100000, metavar='STEPS', help='Number of training steps between evaluations')
parser.add_argument('--evaluation-episodes', type=int, default=10, metavar='N', help='Number of evaluation episodes to average over')
# TODO: Note that DeepMind's evaluation method is running the latest agent for 500K frames ever every 1M steps
parser.add_argument('--evaluation-size', type=int, default=500, metavar='N', help='Number of transitions to use for validating Q')
parser.add_argument('--render', action='store_true', help='Display screen (testing only)')
parser.add_argument('--enable-cudnn', action='store_true', help='Enable cuDNN (faster but nondeterministic)')
parser.add_argument('--checkpoint-interval', type=int, default=0, help='How often to checkpoint the model, defaults to 0 (never checkpoint)')
parser.add_argument('--memory', help='Path to save/load the memory from')
parser.add_argument('--disable-bzip-memory', action='store_true', help='Don\'t zip the memory file. Not recommended (zipping is a bit slower and much, much smaller)')
parser.add_argument('--enable-wandb', action='store_true', help='Enable Weights and Biases for logging')
parser.add_argument('--weight', type=float, default=1, metavar='w', help='Weight for auxiliary loss')
parser.add_argument('--max-tnh-shift', type=float, default=1.5, metavar='K', help='Maximum range for random shifts to apply to tanh outputs in vector DQN.')
parser.add_argument('--tnh-slope', type=float, default=(0.1,5), metavar='b', help='Range of slopes for tanh inputs in vector DQN.')
parser.add_argument('--Q-bias', type=float, default=1.5, metavar='Q_bias', help='Maximum range of random biases to apply to vector Q-values.')
parser.add_argument('--vec-target-mean', action='store_false', help='Whether to construct target Q-value from averaging across all atoms or not in vector_DQN.')
parser.add_argument('--track-grads', action='store_false', help='Track gradients.')

# Setup
args = parser.parse_args()

print(' ' * 26 + 'Options')
for k, v in vars(args).items():
  print(' ' * 26 + k + ': ' + str(v))
results_dir = os.path.join('/users/rliu70/scratch/Rainbow', args.id)
seed_dir = os.path.join(results_dir, f'{args.seed}')
if not os.path.exists(results_dir):
  os.makedirs(results_dir)
if not os.path.exists(seed_dir):
  os.makedirs(seed_dir)

metrics = {'steps': [], 'rewards': [], 'Qs': [], 'best_avg_reward': -float('inf')}
np.random.seed(args.seed)
torch.manual_seed(np.random.randint(1, 10000))
if torch.cuda.is_available() and not args.disable_cuda:
  args.device = torch.device('cuda')
  torch.cuda.manual_seed(np.random.randint(1, 10000))
  torch.backends.cudnn.enabled = args.enable_cudnn
else:
  args.device = torch.device('cpu')


# Simple ISO 8601 timestamped logger
def log(s):
  print('[' + str(datetime.now().strftime('%Y-%m-%dT%H:%M:%S')) + '] ' + s)


def load_memory(memory_path, disable_bzip):
  if disable_bzip:
    with open(memory_path, 'rb') as pickle_file:
      return pickle.load(pickle_file)
  else:
    with bz2.open(memory_path, 'rb') as zipped_pickle_file:
      return pickle.load(zipped_pickle_file)


def save_memory(memory, memory_path, disable_bzip):
  if disable_bzip:
    with open(memory_path, 'wb') as pickle_file:
      pickle.dump(memory, pickle_file)
  else:
    with bz2.open(memory_path, 'wb') as zipped_pickle_file:
      pickle.dump(memory, zipped_pickle_file)

if args.enable_wandb:
  wandb.login()
  params = vars(args)
  params.pop("enable_wandb")
  wandb.init(
    project="Rainbow-"+args.id,
    config=params,
  )
  args.enable_wandb = True

# Environment
env = Env(args)
env.train()
action_space = env.action_space()

#
if args.algo == "Rainbow":
  agent = Rainbow(args, env)
elif args.algo == "Rainbow_DQN":
  agent = Rainbow_DQN(args, env)
elif args.algo == "Rainbow_mean_var_DQN":
  agent = Rainbow_mean_var_DQN(args, env)
elif args.algo == "Rainbow_mean_var_DQN2":
  agent = Rainbow_mean_var_DQN2(args, env)
elif args.algo == "Rainbow_mean_var_DQNa":
  agent = Rainbow_mean_var_DQNa(args, env)
elif args.algo == "Rainbow_vec_DQN":
  agent = Rainbow_vec_DQN(args, env)
elif args.algo == "Rainbow_DQN51":
  agent = Rainbow_DQN51(args, env)
elif args.algo == 'Rainbow_DQN51_v2':
  agent = Rainbow_DQN51_v2(args, env)
elif args.algo == 'Rainbow_DQN51_v3':
  agent = Rainbow_DQN51_v3(args, env)
elif args.algo == "Rainbow_DQN51_ent":
  agent = Rainbow_DQN51_ent(args, env)
elif args.algo == "Rainbow_DQN51_cross_ent":
  agent = Rainbow_DQN51_cross_ent(args, env)
elif args.algo == "Rainbow_mean_var_51":
  agent = Rainbow_mean_var_51(args, env)

# If a model is provided, and evaluate is false, presumably we want to resume, so try to load memory
if args.model is not None and not args.evaluate:
  if not args.memory:
    raise ValueError('Cannot resume training without memory save path. Aborting...')
  elif not os.path.exists(args.memory):
    raise ValueError('Could not find memory file at {path}. Aborting...'.format(path=args.memory))

  mem = load_memory(args.memory, args.disable_bzip_memory)

else:
  mem = ReplayMemory(args, args.memory_capacity)

priority_weight_increase = (1 - args.priority_weight) / (args.T_max - args.learn_start)


# Construct validation memory
val_mem = ReplayMemory(args, args.evaluation_size)
T, done = 0, True
while T < args.evaluation_size:
  if done:
    state = env.reset()

  next_state, _, done = env.step(np.random.randint(0, action_space))
  val_mem.append(state, -1, 0.0, done)
  state = next_state
  T += 1

if args.evaluate:
  agent.eval()  # Set online network to evaluation mode
  avg_reward, avg_Q = test(args, 0, agent, val_mem, metrics, seed_dir, evaluate=True)  # Test
  print('Avg. reward: ' + str(avg_reward) + ' | Avg. Q: ' + str(avg_Q))
  if args.enable_wandb:
    wandb.log({"Avg eval reward": avg_reward, "Avg eval Q": avg_Q})
else:
  # Training loop
  agent.train()
  done = True
  for T in trange(1, args.T_max + 1):
    if done:
      state = env.reset()

    if T % args.replay_frequency == 0:
      agent.reset_noise()  # Draw a new set of noisy weights

    action = agent.act(state)  # Choose an action greedily (with noisy weights)
    next_state, reward, done = env.step(action)  # Step
    if args.reward_clip > 0:
      reward = max(min(reward, args.reward_clip), -args.reward_clip)  # Clip rewards
    mem.append(state, action, reward, done)  # Append transition to memory

    # Train and test
    if T >= args.learn_start:
      mem.priority_weight = min(mem.priority_weight + priority_weight_increase, 1)  # Anneal importance sampling weight β to 1

      if T % args.replay_frequency == 0:
        losses, grad_norms = agent.learn(mem)  # Train with n-step distributional double-Q learning

      if T % args.evaluation_interval == 0:
        agent.eval()  # Set online network to evaluation mode
        avg_reward, avg_Q = test(args, T, agent, val_mem, metrics, seed_dir)  # Test
        log('T = ' + str(T) + ' / ' + str(args.T_max) + ' | Avg. reward: ' + str(avg_reward) + ' | Avg. Q: ' + str(avg_Q))
        if args.enable_wandb:
          log_dict = {"Avg eval reward": avg_reward, "Avg eval Q": avg_Q}
          log_dict.update(losses)
          log_dict.update(grad_norms)
          wandb.log(log_dict)
        agent.train()  # Set online network back to training mode

        # If memory path provided, save it
        if args.memory is not None:
          save_memory(mem, args.memory, args.disable_bzip_memory)

      # Update target network
      if T % args.target_update == 0:
        agent.update_target_net()

      # Checkpoint the network
      if (args.checkpoint_interval != 0) and (T % args.checkpoint_interval == 0):
        agent.save(seed_dir, f'checkpoint_{T}.pth')

    state = next_state

env.close()
