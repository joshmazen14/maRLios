import datetime
import torch
import torch.nn as nn
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros import SuperMarioBrosEnv
from tqdm import tqdm
import pickle 
from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
import gym
import numpy as np
import matplotlib.pyplot as plt

from toolkit.gym_env import *
from toolkit.model import *

def save_model(agent):
    with open("ending_position.pkl", "wb") as f:
            pickle.dump(agent.ending_position, f)
    with open("num_in_queue.pkl", "wb") as f:
        pickle.dump(agent.num_in_queue, f)
    with open("total_rewards.pkl", "wb") as f:
        pickle.dump(total_rewards, f)
    if agent.double_dq:
        torch.save(agent.local_net.state_dict(), "dq1.pt")
        torch.save(agent.target_net.state_dict(), "dq2.pt")
    else:
        torch.save(agent.dqn.state_dict(), "dq.pt")  
    torch.save(agent.STATE_MEM,  "STATE_MEM.pt")
    torch.save(agent.ACTION_MEM, "ACTION_MEM.pt")
    torch.save(agent.REWARD_MEM, "REWARD_MEM.pt")
    torch.save(agent.STATE2_MEM, "STATE2_MEM.pt")
    torch.save(agent.DONE_MEM,   "DONE_MEM.pt")


def vectorize_action(action, action_space):
    # Given a scalar action, return a one-hot encoded action
    
    return [0 for _ in range(action)] + [1] + [0 for _ in range(action + 1, action_space)]

def show_state(env, ep=0, info=""):
    plt.figure(3)
    plt.clf()
    plt.imshow(env.render(mode='rgb_array'))
    plt.title("Episode: %d %s" % (ep, info))
    plt.axis('off')

    display.clear_output(wait=True)
    display.display(plt.gcf())

def make_env(env):
    env = MaxAndSkipEnv(env)
    env = ProcessFrame84(env)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, 4)
    env = ScaledFloatFrame(env)
    return JoypadSpace(env, SIMPLE_MOVEMENT) # change here for dif movement

def run(training_mode, pretrained):
   
    fh = open('progress.txt', 'a')
    env = gym.make('SuperMarioBros-1-1-v0')
    #env = gym_super_mario_bros.make('SuperMarioBros-v0')
    
    #env = make_env(env)  # Wraps the environment so that frames are grayscale 
    #env = SuperMarioBrosEnv()
    env = make_env(env)
    observation_space = env.observation_space.shape
    action_space = env.action_space.n

    #todo: add agent params as a setting/create different agents in diff functions to run 

    agent = DQNAgent(state_space=observation_space,
                     action_space=action_space,
                     max_memory_size=30000,
                     batch_size=32,
                     gamma=0.90,
                     lr=0.00025,
                     dropout=0.,
                     exploration_max=1.0,
                     exploration_min=0.02,
                     exploration_decay=0.99,
                     double_dq=True,
                     pretrained=pretrained)
    
    
    num_episodes = 10
    env.reset()
    total_rewards = []
    
    for ep_num in tqdm(range(num_episodes), file=fh):
        state = env.reset()
        state = torch.Tensor([state])
        total_reward = 0
        steps = 0
        while True:
            if not training_mode:
                show_state(env, ep_num)
            action = agent.act(state)
            steps += 1
            
            state_next, reward, terminal, info = env.step(int(action[0]))
            total_reward += reward
            state_next = torch.Tensor([state_next])
            reward = torch.tensor([reward]).unsqueeze(0)
            
            terminal = torch.tensor([int(terminal)]).unsqueeze(0)
            
            if training_mode:
                agent.remember(state, action, reward, state_next, terminal)
                agent.experience_replay()
            
            state = state_next
            if terminal:
                break
        
        if ep_num % 500 == 0 and training_mode:
            save_model(agent)

        total_rewards.append(total_reward)
        with open('total_reward.txt', 'a') as f:
            f.write("Total reward after episode {} is {}\n".format(ep_num + 1, total_rewards[-1]))
            if (ep_num%100 == 0):
                f.write("==================\n")
                f.write("{} current time at episode {}\n".format(datetime.datetime.now(), ep_num+1))
                f.write("==================\n")
            #print("Total reward after episode {} is {}".format(ep_num + 1, total_rewards[-1]))
            num_episodes += 1 
    
    env.close()
    fh.close()
    
    # if num_episodes > 500:
    #     plt.title("Episodes trained vs. Average Rewards (per 500 eps)")
    #     plt.plot([0 for _ in range(500)] + 
    #              np.convolve(total_rewards, np.ones((500,))/500, mode="valid").tolist())
    #     plt.show()

run(training_mode=True, pretrained=False)
