from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
# import torchvision
import torch.nn as nn
import random
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros import SuperMarioBrosEnv
from tqdm import tqdm
import numpy as np
import pickle 
import numpy as np
import collections 
import cv2
import matplotlib.pyplot as plt
import toolkit.action_utils 

class DQNSolver(nn.Module):

    def __init__(self, input_shape):
        super(DQNSolver, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, kernel_size=8, stride=4),
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=6, stride=4),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.LeakyReLU()
        )
        conv_out_size = self._get_conv_out(input_shape)
        # takes the output of the convolutions and gets vector to size 32
        self.conv_to_32 = nn.Sequential(
            nn.Linear(conv_out_size, 32),
            nn.LeakyReLU()
        )

        action_size = 10

        self.action_fc = nn.Sequential(
            nn.Linear(action_size, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 32)
        )
        
        # We take a vector of 5 being the initial action, and 5 being the second action for action size of 10
        self.fc = nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32), # added a new layer can play with the parameters
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        )
    
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x, sampled_actions):
        '''
        x - image being passed in as the state
        sampled_actions - np.array with n x 8 
        '''
        big_conv_out = self.conv(x).view(x.size()[0], -1)
        conv_out = self.conv_to_32(big_conv_out)

        batched_conv_out = conv_out.reshape(conv_out.shape[0], 1, conv_out.shape[-1]).repeat(1, sampled_actions.shape[-2], 1)

        latent_actions = self.action_fc(sampled_actions)

        batched_actions = torch.cat((batched_conv_out, latent_actions), dim=2)

        out =  torch.flatten(self.fc(batched_actions), start_dim=1)

        return out


# class DQNSolver(nn.Module):

#     def __init__(self, input_shape):
#         super(DQNSolver, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(input_shape[0], 64, kernel_size=8, stride=4),
#             nn.MaxPool2d(kernel_size=4, stride=2),
#             nn.LeakyReLU(),
#             nn.Conv2d(64, 64, kernel_size=6, stride=4),
#             nn.MaxPool2d(kernel_size=4, stride=2),
#             nn.LeakyReLU(),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1),
#             nn.LeakyReLU()
#         )

#         conv_out_size = self._get_conv_out(input_shape)
#         print("conv_out_size: ",conv_out_size)
#         action_size = 10

#         self.action_fc = nn.Sequential(
#             nn.Linear(action_size, 30),
#             nn.LeakyReLU(),
#             nn.Linear(30, action_size), # this doesn't neeeed to go to action size necessarily
#         )
        
#         # We take a vector of 5 being the initial action, and 5 being the second action for action size of 10
#         self.fc = nn.Sequential(
#             nn.Linear(conv_out_size + action_size, 512),
#             nn.ReLU(),
#             nn.Linear(512, 64), # added a new layer can play with the parameters
#             nn.ReLU(),
#             nn.Linear(64, 1)
#         )
    
#     def _get_conv_out(self, shape):
#         o = self.conv(torch.zeros(1, *shape))
#         return int(np.prod(o.size()))

#     def forward(self, x, sampled_actions):
#         '''
#         x - image being passed in as the state
#         sampled_actions - np.array with n x 8 
#         '''
#         conv_out = self.conv(x).view(x.size()[0], -1)
#         batched_conv_out = conv_out.reshape(conv_out.shape[0], 1, conv_out.shape[-1]).repeat(1, sampled_actions.shape[-2], 1)

#         latent_actions = self.action_fc(sampled_actions)

#         batched_actions = torch.cat((batched_conv_out, latent_actions), dim=2)

#         out =  torch.flatten(self.fc(batched_actions), start_dim=1)

#         return out
    

class DQNAgent:

    def __init__(self, action_space, max_memory_size, batch_size, gamma, lr, state_space,
                 dropout, exploration_max, exploration_min, exploration_decay, double_dq, pretrained, run_id='', n_actions = 32):

        # Define DQN Layers
        self.state_space = state_space

        self.action_space = action_space # this will be a set of actions ie: a subset of TWO_ACTIONS in constants.py
        self.n_actions = n_actions # initial number of actions to sample

        self.device ='cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        
        self.cur_action_space = torch.from_numpy(self.subsample_actions(self.n_actions)).to(torch.float32).to(self.device).unsqueeze(0) # make it include a batch dimension by defautl

        self.double_dq = double_dq
        self.pretrained = pretrained
        

        # this has been altered as we no longer need to pass the number of actions
        self.local_net = DQNSolver(self.state_space).to(self.device)
        self.target_net = DQNSolver(self.state_space).to(self.device)
        
        if self.pretrained:
            self.local_net.load_state_dict(torch.load(f"dq1-{run_id}.pt", map_location=torch.device(self.device)))
            self.target_net.load_state_dict(torch.load(f"dq2-{run_id}.pt", map_location=torch.device(self.device)))
        
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.local_net.parameters(), lr=lr)
        self.copy = 5000  # Copy the local model weights into the target network every 5000 steps
        self.step = 0
    
    

        # Create memory
        self.max_memory_size = max_memory_size
        if self.pretrained:
            # self.STATE_MEM = torch.load(f"STATE_MEM-{run_id}.pt")
            # self.ACTION_MEM = torch.load(f"ACTION_MEM-{run_id}.pt")
            # self.REWARD_MEM = torch.load(f"REWARD_MEM-{run_id}.pt")
            # self.STATE2_MEM = torch.load(f"STATE2_MEM-{run_id}.pt")
            # self.DONE_MEM = torch.load(f"DONE_MEM-{run_id}.pt")
            # self.SPACE_MEM = torch.load(f"SPACE_MEM-{run_id}.pt")
            with open(f"ending_position-{run_id}.pkl", 'rb') as f:
                self.ending_position = pickle.load(f)
            with open(f"num_in_queue-{run_id}.pkl", 'rb') as f:
                self.num_in_queue = pickle.load(f)
        else:
            # self.STATE_MEM = torch.zeros(max_memory_size, *self.state_space)
            # self.ACTION_MEM = torch.zeros(max_memory_size, 1) # this needs to be a matrix of the actual action taken
            # self.REWARD_MEM = torch.zeros(max_memory_size, 1)
            # self.STATE2_MEM = torch.zeros(max_memory_size, *self.state_space)
            # self.DONE_MEM = torch.zeros(max_memory_size, 1)
            # self.SPACE_MEM = torch.zeros(max_memory_size, self.n_actions, 10)
            self.ending_position = 0
            self.num_in_queue = 0

        self.STATE_MEM = torch.zeros(max_memory_size, *self.state_space)
        self.ACTION_MEM = torch.zeros(max_memory_size, 1) # this needs to be a matrix of the actual action taken
        self.REWARD_MEM = torch.zeros(max_memory_size, 1)
        self.STATE2_MEM = torch.zeros(max_memory_size, *self.state_space)
        self.DONE_MEM = torch.zeros(max_memory_size, 1)
        self.SPACE_MEM = torch.zeros(max_memory_size, self.n_actions, 10)
        
        self.memory_sample_size = batch_size
        
        # Learning parameters
        self.gamma = gamma
        self.l1 = nn.SmoothL1Loss().to(self.device) # Also known as Huber loss
        self.l2 = nn.MSELoss().to(self.device)
        self.exploration_max = exploration_max
        self.exploration_rate = exploration_max
        self.exploration_min = exploration_min
        self.exploration_decay = exploration_decay
        

    def subsample_actions(self, n_actions):
        '''
        Returns numpy array 
        '''
        return toolkit.action_utils.sample_actions(self.action_space, n_actions)


    def remember(self, state, action, reward, state2, done):
        self.STATE_MEM[self.ending_position] = state.float()
        self.ACTION_MEM[self.ending_position] = action.float()
        self.REWARD_MEM[self.ending_position] = reward.float()
        self.STATE2_MEM[self.ending_position] = state2.float()
        self.DONE_MEM[self.ending_position] = done.float()
        self.SPACE_MEM[self.ending_position] = self.cur_action_space
        self.ending_position = (self.ending_position + 1) % self.max_memory_size  # FIFO tensor
        self.num_in_queue = min(self.num_in_queue + 1, self.max_memory_size)
        
    def recall(self):
        # Randomly sample 'batch size' experiences
        idx = random.choices(range(self.num_in_queue), k=self.memory_sample_size)
        
        STATE = self.STATE_MEM[idx]
        ACTION = self.ACTION_MEM[idx]
        REWARD = self.REWARD_MEM[idx]
        STATE2 = self.STATE2_MEM[idx]
        DONE = self.DONE_MEM[idx]
        SPACE = self.SPACE_MEM[idx]
        
        return STATE, ACTION, REWARD, STATE2, DONE, SPACE

    def act(self, state):
        '''
        Returns the action vector
        '''
        # Epsilon-greedy action
        
        # increment step
        self.step += 1

        if random.random() < self.exploration_rate:  
            rand_ind = random.randrange(0, self.cur_action_space.shape[1])

            return torch.tensor(rand_ind).unsqueeze(0)
        
            # Local net is used for the policy

            # Updated for generalization:
        results = self.local_net(state.to(self.device), self.cur_action_space).cpu()
        return torch.argmax(results, dim=1)
        # action = torch.tensor(self.cur_action_space[act_index])

    def copy_model(self):
        # Copy local net weights into target net
        
        self.target_net.load_state_dict(self.local_net.state_dict())

    def decay_exploration(self):
        self.exploration_rate *= self.exploration_decay
        
        # Makes sure that exploration rate is always at least 'exploration min'
        self.exploration_rate = max(self.exploration_rate, self.exploration_min)
    def decay_lr(self, lr_decay):
        self.lr *= lr_decay
        self.lr = max(self.lr, 0.000001)
        for g in self.optimizer.param_groups:
            g['lr'] = self.lr

    def experience_replay(self, debug=False):
        
        if self.step % self.copy == 0:
            self.copy_model()

        if self.memory_sample_size > self.num_in_queue:
            return

        STATE, ACTION, REWARD, STATE2, DONE, SPACE = self.recall()
        STATE = STATE.to(self.device)
        ACTION = ACTION.to(self.device)
        REWARD = REWARD.to(self.device)
        STATE2 = STATE2.to(self.device)
        SPACE = SPACE.to(self.device)
        DONE = DONE.to(self.device)
        
        self.optimizer.zero_grad()
        # Double Q-Learning target is Q*(S, A) <- r + γ max_a Q_target(S', a)
    
        target = REWARD + torch.mul((self.gamma * 
                                    self.target_net(STATE2, SPACE).max(1).values.unsqueeze(1)), 
                                    1 - DONE)

        current = self.local_net(STATE, SPACE).gather(1, ACTION.long()) # Local net approximation of Q-value
    
    
        loss = self.l1(current, target) # maybe we can play with some L2 loss 
        loss.backward() # Compute gradients
        self.optimizer.step() # Backpropagate error

        # self.cur_action_space = torch.from_numpy(self.subsample_actions(self.n_actions)).to(torch.float32).to(self.device)
        # I am disabling this here for my testing, but also think we should add it to the run loop for testing til we are sure it works, idk
        # decay lr

        if debug:
            return target, current, loss