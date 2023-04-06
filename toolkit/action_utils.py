import numpy as np
import itertools
from constants import BUTTONS, ACTION_SPACE, BUTTON_MAP


def sample_actions(action_set, n_actions, hold_out_actions):
    '''
    action_set - the actions available to the agent
    n_actions - batch size
    '''
    sampled_actions = np.zeros((n_actions, 10))
    sampled_idx = np.random.randint(0, len(action_set), size=n_actions)
    
    # sampled_idx_2 = np.random.randint(0, len(action_set), size=n_actions)

            


class ActionSet:

    def __init__(self, n_actions, hold_out_actions):
        
        self.cur_action_set = np.zeros(shape=[n_actions, 10])



def vec_to_action(vec):
    '''
    vector is a combination of two actions 
    sample vector[left, right, down, a, b, |split between actions| left, right, down, a, b]
    '''
    split_vec = np.split(vec, 2)

    act1_vec = split_vec[0]
    act2_vec = split_vec[1]

     # Use boolean indexing to get the indices of elements in the vector that are equal to 1
    act1_indices = np.where(act1_vec == 1)[0]
    act2_indices = np.where(act2_vec == 1)[0]

    # Use the indices to get the corresponding button names
    act1 = [BUTTONS[i] for i in act1_indices]
    act2 = [BUTTONS[i] for i in act2_indices]

    # If the action lists are empty, add 'NOOP'
    if not act1:
        act1 = ['NOOP']
    if not act2:
        act2 = ['NOOP']

    return act1, act2

def action_to_vec(actions):
    act1, act2 = actions
    vec1 = np.zeros(shape=[1, 5])
    vec2 = np.zeros(shape=[1, 5])

    for i, j in zip(act1, act2):
        ind1 = BUTTON_MAP[i]
        ind2 = BUTTON_MAP[i]
        vec1[ind1] = 1
        vec2[ind2] = 1
    
    return np.concatenate(vec1, vec2)
        