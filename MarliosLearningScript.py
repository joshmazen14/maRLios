import torch
import torch.nn as nn
import random
import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros import SuperMarioBrosEnv
from tqdm import tqdm
import pickle 
import gym
import numpy as np
import collections 
import cv2
import matplotlib.pyplot as plt
import time
import datetime
import json
from toolkit.gym_env import *
from toolkit.action_utils_carlos import *
from toolkit.marlios_model_carlos import *
from toolkit.train_marlios_carlos import *
from toolkit.constants_carlos import *
from toolkit.train_test_samples import *
import toolkit.constants_carlos as action_constants
import toolkit.train_test_samples as train_test_samples
import argparse
import ast
import warnings

warnings.filterwarnings('ignore')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runs MaRLios agent on Super Mario bros. gym environment")

    parser.add_argument("--training-mode", type=ast.literal_eval, default=True, help="Training mode (default: True)")
    parser.add_argument("--pretrained", type=ast.literal_eval, default=False, help="Use pretrained model (default: False)")
    parser.add_argument("--lr", type=float, default=0.00025, help="Learning rate (default: 0.00025)")
    parser.add_argument("--lr-decay", type=float, default=0.999, help="Learning rate decay (default: 0.999)")
    parser.add_argument("--gamma", type=float, default=0.90, help="Discount factor (default: 0.90)")
    parser.add_argument("--exploration-decay", type=float, default=0.99, help="Exploration decay (default: 0.99)")
    parser.add_argument("--exploration-min", type=float, default=0.02, help="Exploration minimum (default: 0.02)")
    parser.add_argument("--exploration-max", type=float, default=1, help="Exploration maximum (default: 1.00)")
    parser.add_argument("--mario-env", type=str, default='SuperMarioBros-1-1-v0', help="Mario environment (default: 'SuperMarioBros-1-1-v0')")
    parser.add_argument("--actions", type=str, default='TRAIN_SET', help="Actions (default: 'TRAIN_SET')")
    parser.add_argument("--num-episodes", type=int, default=10, help="Number of episodes (default: 10)")
    parser.add_argument("--run-id", type=str, default=None, help="Run ID (default: epoch timestring)")
    parser.add_argument("--ep-stat", type=int, default=100, help="Number of episodes to store stats (default: 100)")
    parser.add_argument("--n-actions", type=int, default=None, help="Number of actions to to give to model (default: len(action_space))")
    parser.add_argument("--sample-actions", type=ast.literal_eval, default=False, help="Sample actions from action space (default: False)")
    parser.add_argument("--max-time-per-ep", type=float, default=200, help="Episode time limit to start training with  (default: 200)")
    parser.add_argument("--sample-step", type=ast.literal_eval, default=False, help="If true, subsample on every step, otherwise on every episode  (default: False)")

    args = parser.parse_args()
    print('test: ', args)

    try:
        action_space = getattr(train_test_samples, args.actions)
    except AttributeError as e:
        raise ValueError("Invalid actions argument.")
    
    if (not args.n_actions):
        n_actions = len(action_space) + 2 if args.sample_actions else len(action_space)
    else:
        n_actions = args.n_actions



    train(training_mode=args.training_mode,
         pretrained=args.pretrained,
         ep_per_stat=args.ep_stat,
         lr=args.lr,
         lr_decay=args.lr_decay,
         gamma=args.gamma,
         exploration_decay=args.exploration_decay,
         exploration_min=args.exploration_min,
         exploration_max=args.exploration_max,
         mario_env=args.mario_env,
         action_space=action_space,
         num_episodes=args.num_episodes,
         n_actions=n_actions, # +2 for no-op and sufficient action
         run_id=args.run_id,
         sample_actions=args.sample_actions,
         max_time_per_ep=args.max_time_per_ep,
         sample_step=args.sample_step
         )
    

