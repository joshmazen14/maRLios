from toolkit.train_test_samples import *
from toolkit.action_utils import *
from toolkit.constants import *
import torch 

def get_stats_run(agent, env):
    agent.subsample_actions()
    
    state = env.reset() # take the final dimension of shape 
    state = torch.Tensor([state])
    total_reward = 0
    info = None
    # prev_hidden_state = torch.zeros(1, 1, agent.hidden_shape).to(agent.device)

    while True:
        
        two_actions_index = agent.act(state)
        two_actions_vector = agent.cur_action_space[0, two_actions_index[0]]
        two_actions = vec_to_action(two_actions_vector.cpu()) # tuple of actions
        
        reward = 0

        terminal = False
        for action in two_actions: 
            if not terminal:
                # compute index into ACTION_SPACE of our action
                step_action = ACTION_TO_INDEX[action]

                state_next, cur_reward, terminal, info = env.step(step_action)
                total_reward += cur_reward
                reward += cur_reward
                
        state_next = torch.Tensor([state_next])
        reward = torch.tensor([reward]).unsqueeze(0)        
        terminal = torch.tensor([int(terminal)]).unsqueeze(0)
        
        # lstm new
        # prev_hidden_state = hidden
        # change up action space
        state = state_next
        if terminal:
            break

    stats_gathered = {
        "total_reward" : total_reward,
        "flag_get" : info['flag_get'],
        "x_pos" : info['x_pos'],
        "pipe2" : info['x_pos'] >= 600,
        "pipe3" : info['x_pos'] >= 750,
        "pipe4" : info['x_pos'] >= 900
    }

    return stats_gathered
