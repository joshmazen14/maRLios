from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.init as init
import random
import numpy as np
import pickle 
import numpy as np
import toolkit.action_utils 

torch.autograd.set_detect_anomaly(True)
class DQNSolver(nn.Module):

    def __init__(self, input_shape, n_actions = 64):
        super(DQNSolver, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, kernel_size=8, stride=4),
            nn.AvgPool2d(kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=6, stride=4),
            nn.LeakyReLU()
        )
        conv_out_size = self._get_conv_out(input_shape)
        
        # Xavier initialization for the convolution layer weights
        for layer in self.conv:
            if isinstance(layer, nn.Conv2d):
                init.xavier_uniform_(layer.weight)

        self.lstm = nn.LSTM(input_size=conv_out_size, hidden_size=64, batch_first=True)
        
        # takes the output of the convolutions and gets vector to size 32
        self.lstm_to_32 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        for layer in self.lstm_to_32:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
       
        action_size = 10
        self.action_fc = nn.Sequential(
            nn.Linear(action_size, 32),
            nn.ReLU(),
        )

         # Xavier initialization for the fully connected layer weights
        for layer in self.action_fc:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
        
        # We take a vector of 5 being the initial action, and 5 being the second action for action size of 10
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(n_actions), # using batch size of 64, for now hard coded
            nn.ReLU(),
            nn.Linear(32, 10), # added a new layer can play with the parameters
            nn.ReLU(),
            nn.Linear(10, 1)
        )

        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)

    
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x, sampled_actions, prev_hidden_state = None):
        '''
        x - image being passed in as the state
        sampled_actions - np.array with n x 8 
        prev_hidden_state - tuple of format(hidden, cell) for the hidden states
        '''
        if prev_hidden_state is None:
            # initialize empty hidden state of 0's
            h_0 = torch.zeros(1, 1, 64).to(x.device)
            c_0 = torch.zeros(1, 1, 64).to(x.device)
        else:
            h_0, c_0 = prev_hidden_state 

        big_conv_out = self.conv(x).view(x.size()[0], -1) # has shape of (1, 1024) => (batch, output size)

        # print("LSTM SHAPES: ")
        # print("input: ", big_conv_out.unsqueeze(1).shape)
        # print("hidden/cell: ", h_0.shape)
        
        # pass to lstm layer, and store output and hidden state
        lstm_out, (h_n, c_n) = self.lstm(big_conv_out.unsqueeze(1), (h_0, c_0))
        lstm_out = lstm_out.squeeze(1) # remove the sequence length dimension
        lstm_out = self.lstm_to_32(lstm_out)

        batched_lstm_out = lstm_out.reshape(lstm_out.shape[0], 1, lstm_out.shape[-1]).repeat(1, sampled_actions.shape[-2], 1)

        latent_actions = self.action_fc(sampled_actions)

        batched_actions = torch.cat((batched_lstm_out, latent_actions), dim=2)

        out =  torch.flatten(self.fc(batched_actions), start_dim=1)

        return out, (h_n, c_n)

    

class DQNAgent:

    def __init__(self, action_space, max_memory_size, batch_size, gamma, lr, state_space,
                 dropout, exploration_max, exploration_min, exploration_decay, double_dq, pretrained, run_id='', n_actions = 64, device=None, init_max_time=500):

        # Define DQN Layers
        self.state_space = state_space

        self.action_space = action_space # this will be a set of actions ie: a subset of TWO_ACTIONS in constants.py
        self.n_actions = n_actions # initial number of actions to sample
        if device == None:
            self.device ='cpu'
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
        else:
            self.device = device
        

        self.subsample_actions()

        self.double_dq = double_dq
        self.pretrained = pretrained
        

        # this has been altered as we no longer need to pass the number of actions
        self.local_net = DQNSolver(self.state_space, n_actions=n_actions).to(self.device)
        self.target_net = DQNSolver(self.state_space, n_actions=n_actions).to(self.device)
        
        if self.pretrained:
            self.local_net.load_state_dict(torch.load(f"dq1-{run_id}.pt", map_location=torch.device(self.device)))
            self.target_net.load_state_dict(torch.load(f"dq2-{run_id}.pt", map_location=torch.device(self.device)))
        
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.local_net.parameters(), lr=lr)
        self.copy = 5000  # Copy the local model weights into the target network every 5000 steps
        self.step = 0
        self.max_time_per_ep = init_max_time
    
        # Create memory
        self.max_memory_size = max_memory_size
        if self.pretrained:
            with open(f"ending_position-{run_id}.pkl", 'rb') as f:
                self.ending_position = pickle.load(f)
            with open(f"num_in_queue-{run_id}.pkl", 'rb') as f:
                self.num_in_queue = pickle.load(f)
        else:
            self.ending_position = 0
            self.num_in_queue = 0

        self.STATE_MEM = torch.zeros(max_memory_size, *self.state_space)
        self.ACTION_MEM = torch.zeros(max_memory_size, 1) # this needs to be a matrix of the actual action taken
        self.REWARD_MEM = torch.zeros(max_memory_size, 1)
        self.STATE2_MEM = torch.zeros(max_memory_size, *self.state_space)
        self.DONE_MEM = torch.zeros(max_memory_size, 1)
        self.SPACE_MEM = torch.zeros(max_memory_size, self.n_actions, 10)

        # for the lstm layers, i think these need to be on the same device
        self.HIDDEN_MEM = torch.zeros(max_memory_size, 1, 64)
        self.CELL_MEM = torch.zeros(max_memory_size, 1, 64)
        
        self.memory_sample_size = batch_size
        
        # Learning parameters
        self.gamma = gamma
        self.l1 = nn.SmoothL1Loss().to(self.device) # Also known as Huber loss
        self.l2 = nn.MSELoss().to(self.device)
        self.exploration_max = exploration_max
        self.exploration_rate = exploration_max
        self.exploration_min = exploration_min
        self.exploration_decay = exploration_decay
        

    def subsample_actions(self):
        '''
        Changes curaction space to be a random sample of what it was
        '''

        self.cur_action_space = torch.from_numpy(toolkit.action_utils.sample_actions(self.action_space, self.n_actions)).to(torch.float32).to(self.device).unsqueeze(0)
    


    def remember(self, state, action, reward, state2, done, hidden_state):
        self.STATE_MEM[self.ending_position] = state.float()
        self.ACTION_MEM[self.ending_position] = action.float()
        self.REWARD_MEM[self.ending_position] = reward.float()
        self.STATE2_MEM[self.ending_position] = state2.float()
        self.DONE_MEM[self.ending_position] = done.float()
        self.SPACE_MEM[self.ending_position] = self.cur_action_space
        self.HIDDEN_MEM[self.ending_position] = hidden_state[0].squeeze(1) # hidden state is (1, 1, 64)
        self.CELL_MEM[self.ending_position] = hidden_state[1].squeeze(1)

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

        HIDDEN = self.HIDDEN_MEM[idx]
        CELL = self.CELL_MEM[idx]
        
        return STATE, ACTION, REWARD, STATE2, DONE, SPACE, HIDDEN.transpose(0, 1).detach(), CELL.transpose(0, 1).detach()

    def act(self, state, prev_hidden_state):
        '''
        Returns the action index and hidden state
        '''
        # Epsilon-greedy action
        
        # increment step
        self.step += 1
        results, hidden = self.local_net(state.to(self.device), self.cur_action_space, prev_hidden_state)
        ind = torch.argmax(results, dim=1) # index of the 'best' action

        if random.random() < self.exploration_rate:  
            rand_ind = random.randrange(0, self.cur_action_space.shape[1])
            ind = torch.tensor(rand_ind).unsqueeze(0) # wiht some probability, choose a random index
        
        return ind.cpu(), hidden

    def copy_model(self):
        # Copy local net weights into target net
        
        self.target_net.load_state_dict(self.local_net.state_dict())

    def decay_exploration(self):
        self.exploration_rate *= self.exploration_decay
        
        # Makes sure that exploration rate is always at least 'exploration min'
        self.exploration_rate = max(self.exploration_rate, self.exploration_min)

    def decay_lr(self, lr_decay):
        self.lr *= lr_decay
        self.lr = max(self.lr, 0.000000001)
        for g in self.optimizer.param_groups:
            g['lr'] = self.lr

    def experience_replay(self, debug=False):
        
        if self.step % self.copy == 0:
            self.copy_model()

        if self.memory_sample_size > self.num_in_queue:
            return None

        STATE, ACTION, REWARD, STATE2, DONE, SPACE, HIDDEN, CELL = self.recall()
        STATE = STATE.to(self.device)
        ACTION = ACTION.to(self.device)
        REWARD = REWARD.to(self.device)
        STATE2 = STATE2.to(self.device)
        SPACE = SPACE.to(self.device)
        DONE = DONE.to(self.device)
        HIDDEN = HIDDEN.to(self.device)
        CELL = CELL.to(self.device)

        self.optimizer.zero_grad()
        # Double Q-Learning target is Q*(S, A) <- r + γ max_a Q_target(S', a)

        current, (HIDDEN2, CELL2) = self.local_net(STATE, SPACE, (HIDDEN, CELL))
        current = current.gather(1, ACTION.long()) # Local net approximation of Q-value

        # print("Got current")
    
        target, _ = self.target_net(STATE2, SPACE, (HIDDEN2, CELL2))
        target = REWARD + torch.mul((self.gamma * target.max(1).values.unsqueeze(1)), 1 - DONE)
        # print("Got target")

        loss = self.l1(current, target) # maybe we can play with some L2 loss 

        loss.backward(retain_graph=True) # Compute gradients
        self.optimizer.step() # Backpropagate error
        # print("stepped")
        if debug:
            return loss.float()