from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import random
import numpy as np
import pickle 
import numpy as np
import toolkit.action_utils_carlos as action_utils

class DQNSolver(nn.Module):

    def __init__(self, input_shape):
        super(DQNSolver, self).__init__()
        self.action_size = 10
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.LeakyReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)

        # takes the output of the convolutions and gets vector to size 32
        self.conv_to_fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 64),
            nn.LeakyReLU()
        )

        # We take a vector of 5 being the initial action, and 5 being the second action for action size of 10
        self.actions_fc = nn.Sequential(
            # nn.Linear(self.action_size, 100),
            nn.Linear(self.action_size, 12),
            nn.LeakyReLU(),
            # nn.ReLU()
        )
        self.fc = nn.Sequential(
            # nn.Linear(conv_out_size + 100, 512),
            nn.Linear(64 + 12, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            # nn.Linear(128, 32), # added a new layer can play with the parameters
            # nn.BatchNorm1d(32),
            # nn.Linear(512, 64), # added a new layer can play with the parameters
            # nn.BatchNorm1d(64),
            # nn.LeakyReLU(),
            # nn.Linear(64, 1)
            nn.Linear(16, 1)
        )

        # Apply weight initialization
        self.apply(self.init_weights)
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x, sampled_actions):
        '''
        x - image being passed in as the state
        sampled_actions - np.array with n x 8 
        '''
        big_conv_out = self.conv(x).view(x.size()[0], -1)
        conv_out = self.conv_to_fc(big_conv_out)
        batched_conv_out = conv_out.reshape(conv_out.shape[0], 1, conv_out.shape[-1]).repeat(1, sampled_actions.shape[-2], 1)

        latent_actions = self.actions_fc(sampled_actions)
        
        batched_state_actions = torch.cat((batched_conv_out, latent_actions), dim=2)
        # out =  torch.flatten(self.fc(batched_state_actions), start_dim=1)

        # Reshape input to 2D tensor before passing through fc layers
        reshaped_input = batched_state_actions.view(-1, batched_state_actions.shape[-1])

        fc_output = self.fc(reshaped_input)

        # Reshape output back to 3D tensor
        out = fc_output.view(batched_state_actions.shape[0], batched_state_actions.shape[1], -1)
        out =  torch.flatten(out, start_dim=1)

        return out
