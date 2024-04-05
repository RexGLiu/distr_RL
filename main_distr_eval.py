# -*- coding: utf-8 -*-
import argparse
import os
import glob

import atari_py
import numpy as np
import torch

from model import C51
from env import Env

'''
Script to evaluate distributions on frames at different stages of training.
'''


# Note that hyperparameters may originally be reported in ATARI game frames instead of agent steps
parser = argparse.ArgumentParser(description='Rainbow')
parser.add_argument('--game', type=str, default='breakout', choices=atari_py.list_games(), help='ATARI game')
parser.add_argument('--seed', type=int, required=True, help='Enter random seed')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--framepath', type=str, required=True, help='Location of frames to evaluate distributions on.')
parser.add_argument('--run_path', type=str, required=True, help='Location of saved data for the run.')
parser.add_argument('--architecture', type=str, default='canonical', choices=['canonical', 'data-efficient'], metavar='ARCH', help='Network architecture')
parser.add_argument('--hidden-size', type=int, default=512, metavar='SIZE', help='Network hidden size')
parser.add_argument('--noisy-std', type=float, default=0.1, metavar='σ', help='Initial standard deviation of noisy linear layers')
parser.add_argument('--atoms', type=int, default=51, metavar='C', help='Discretised size of value distribution')
parser.add_argument('--V-min', type=float, default=-10, metavar='V', help='Minimum of value distribution support')
parser.add_argument('--V-max', type=float, default=10, metavar='V', help='Maximum of value distribution support')
parser.add_argument('--render', action='store_true', help='Display screen (testing only)')
parser.add_argument('--enable-cudnn', action='store_true', help='Enable cuDNN (faster but nondeterministic)')
parser.add_argument('--savefile', type=str, required=True, help='Filename to save distributions.')

# Setup
args = parser.parse_args()

print(' ' * 26 + 'Options')
for k, v in vars(args).items():
  print(' ' * 26 + k + ': ' + str(v))

framepath = os.path.join('/users/rliu70/scratch/Rainbow', args.framepath)
run_path = os.path.join('/users/rliu70/scratch/Rainbow', args.run_path)

assert os.path.isfile(framepath), "Frame file does not exist."
assert os.path.exists(run_path), "Run path does not exist."

# initialise random seed and select device
seed = args.seed
np.random.seed(seed)
torch.manual_seed(np.random.randint(1, 10000))
if torch.cuda.is_available() and not args.disable_cuda:
  args.device = torch.device('cuda')
  torch.cuda.manual_seed(np.random.randint(1, 10000))
  torch.backends.cudnn.enabled = args.enable_cudnn
else:
  args.device = torch.device('cpu')

# Get action_space from environment
# first set dummy vars required by Env()
args.max_episode_length = 0
args.history_length = 4
env = Env(args)
env.eval()
action_space = env.action_space()
env.close()

# Get correctly sorted list of checkpoint filenames
seed_path = os.path.join(run_path, str(seed))
checkpoint_files = glob.glob(os.path.join(seed_path, "check*.pth"))
ckpt_intervals = [os.path.basename(ckpt) for ckpt in checkpoint_files]
ckpt_intervals = [os.path.splitext(ckpt)[0] for ckpt in ckpt_intervals]
ckpt_intervals = [int(ckpt.split('_')[1]) for ckpt in ckpt_intervals]  # convert intervals into int for sorting
ckpt_intervals.sort()
checkpoint_files = [f"checkpoint_{ckpt}.pth" for ckpt in ckpt_intervals]  # construct list of checkpoint files in correct order
n_checkpoints = len(checkpoint_files)

model = C51(args, action_space)

def load_checkpoint(checkpoint, model):
  if os.path.isfile(checkpoint):
    state_dict = torch.load(checkpoint,
                            map_location='cpu')  # Always load tensors onto CPU by default, will shift to GPU if necessary
    if 'conv1.weight' in state_dict.keys():
      for old_key, new_key in (
      ('conv1.weight', 'convs.0.weight'), ('conv1.bias', 'convs.0.bias'), ('conv2.weight', 'convs.2.weight'),
      ('conv2.bias', 'convs.2.bias'), ('conv3.weight', 'convs.4.weight'), ('conv3.bias', 'convs.4.bias')):
        state_dict[new_key] = state_dict[old_key]  # Re-map state dict for old pretrained models
        del state_dict[old_key]  # Delete old keys for strict load_state_dict
    model.load_state_dict(state_dict)
    print("Loading pretrained model: " + checkpoint)
  else:  # Raise error if incorrect model path provided
    raise FileNotFoundError(checkpoint)

# retrieve frames
with open(framepath, 'rb') as f:
  frame_memory = np.load(f)
  eval_scores = np.load(f)
n_scores = len(eval_scores)

n_frames = len(frame_memory)
distributions = np.empty((n_checkpoints, n_frames, action_space, args.atoms))

for ii, ckpt in enumerate(checkpoint_files):
  #load model
  ckpt_file = os.path.join(seed_path, ckpt)
  load_checkpoint(ckpt_file, model)
  model.eval()

  for jj, s in enumerate(frame_memory):
    with torch.no_grad():
      s = torch.Tensor(s)
      distributions[ii,jj] = model(s).detach().cpu().numpy()

savepath = os.path.join(run_path, f"{args.savefile}_{seed}.npy")
with open(savepath, 'wb') as f:
  np.save(f, distributions)
