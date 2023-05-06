import numpy as np
import random
from toolkit.constants import BUTTONS, ACTION_SPACE, BUTTON_MAP
from toolkit.train_test_samples import *


'''
PRE: action set has been filtered to not include the holdout actions
Returns: an array of action vector representations to feed into our fc network
'''
def sample_actions(action_set, n_actions, add_sufficient=False, training_stage = "train"):
    '''
    action_set - the actions available to the agent
    n_actions - batch size
    add_sufficient - argument to specify whether we want to add a sufficient action or not

    We will in actuality sample n_actions - 1 from the general action space, and sample the remaining action from the sufficient action space. This
    May introduce a single duplicate value with some small probability, but we think that will not pose much of an issue. This is done to ensure
    that the sampled actions always contain an action that can be used to complete the level. 
    '''
    # get the appropriate sufficient set to sample from
    sufficient_jump = None
    sufficient_right = None
    if training_stage == "train":
        sufficient_jump = TRAIN_SET_SUFFICIENT_JUMP_SET
        sufficient_right = TRAIN_SET_SUFFICIENT_RIGHT_SET
    elif training_stage == "test": 
        sufficient_jump = TEST_SET_SUFFICIENT_JUMP_SET
        sufficient_right = TEST_SET_SUFFICIENT_RIGHT_SET
    else:
        sufficient_jump = VALIDATION_SET_SUFFICIENT_JUMP_SET
        sufficient_right = VALIDATION_SET_SUFFICIENT_RIGHT_SET

    action_vectors = np.zeros((n_actions, 10))

    dif = 0
    if add_sufficient:
        # dif = 2
        dif = 3 # this will add noop by default

    sampled_idx = random.sample(range(len(action_set)), n_actions-dif)
    cur_actions = [action_set[i] for i in sampled_idx]

    for i, actions in enumerate(cur_actions):
        vec = action_to_vec(actions)
        action_vectors[i] = vec

    if add_sufficient:
        suff_jump_idx = np.random.randint(0, len(sufficient_jump))
        suff_right_idx = np.random.randint(0, len(sufficient_right))
        action_vectors[n_actions - dif] = action_to_vec(sufficient_jump[suff_jump_idx])
        action_vectors[n_actions - dif+1] = action_to_vec(sufficient_right[suff_right_idx])
        #adding noop as well

    return action_vectors

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

    return tuple(act1), tuple(act2)

def action_to_vec(actions):
    act1, act2 = actions
    vec1 = np.zeros(shape=[5])
    vec2 = np.zeros(shape=[5])

    for i in act1:
        if i != 'NOOP':
            ind1 = BUTTON_MAP[i]
            vec1[ind1] = 1

    for i in act2:
        if i != 'NOOP':
            ind2 = BUTTON_MAP[i]
            vec2[ind2] = 1
    
    return np.concatenate([vec1, vec2], axis=None)



# We probably won't use this, because we made a design choice in sample_actions
def suffient_action_space(action_space):
    '''
    This function will analyze the actions in the action space proposed, and will determine if it 
    is sufficient to finish the game

    Returns Boolean. 
    '''
    
    right_present = False
    a_present = False
    
    for two_actions in action_space:
        if right_present and a_present:
            break

        act1, act2 = two_actions

        if ("right" in act1 and "left" not in act2) or ("right" in act2 and "left" not in act1):
            right_present = True

        if "A" in act1 or "A" in act2:
            a_present = True

    return right_present and a_present



# this is never used outside of my jupyter notebook
def is_sufficient_action(two_actions):

    right_present = False
    a_present = False

    act1, act2 = two_actions

    if ("right" in act1 and "left" not in act2) or ("right" in act2 and "left" not in act1):
        right_present = True

    if "A" in act1 or "A" in act2:
        a_present = True

    return right_present and a_present