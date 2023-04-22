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
from toolkit.action_utils import *
from toolkit.marlios_model import *
from toolkit.constants import *
import wandb

def make_env(env, actions=ACTION_SPACE):
    env = MaxAndSkipEnv(env, skip=2) # I am testing out fewer fram repetitions for our two actions modelling
    env = ProcessFrame84(env)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, 4)
    env = ScaledFloatFrame(env)
    return JoypadSpace(env, actions)

def generate_epoch_time_id():
    epoch_time = int(time.time())
    return str(epoch_time)

def save_checkpoint(agent, total_rewards, terminal_info, run_id):
    with open(f"ending_position-{run_id}.pkl", "wb") as f:
        pickle.dump(agent.ending_position, f)
    with open(f"num_in_queue-{run_id}.pkl", "wb") as f:
        pickle.dump(agent.num_in_queue, f)
    with open(f"total_rewards-{run_id}.pkl", "wb") as f:
        pickle.dump(total_rewards, f)
    with open(f"terminal_info-{run_id}.pkl", "wb") as f:
        pickle.dump(terminal_info, f)
    if agent.double_dq:
        torch.save(agent.local_net.state_dict(), f"dq1-{run_id}.pt")
        torch.save(agent.target_net.state_dict(), f"dq2-{run_id}.pt")
    else:
        torch.save(agent.dqn.state_dict(), f"dq-{run_id}.pt")  

def load_item(from_file):
     with open(from_file, 'rb') as f:
        item = pickle.load(f)
        return item

def plot_rewards(ep_per_stat = 100, total_rewards = [], from_file = None):
    if from_file != None:
        total_rewards = load_item(total_rewards)
       
    avg_rewards = [np.mean(total_rewards[i:i+ep_per_stat]) for i in range(0, len(total_rewards), ep_per_stat)]
    std_rewards = [np.std(total_rewards[i:i+ep_per_stat]) for i in range(0, len(total_rewards), ep_per_stat)]

    fig, ax = plt.subplots()
    ax.plot(avg_rewards, label='Average Rewards')
    ax.fill_between(range(len(avg_rewards)), np.subtract(avg_rewards, std_rewards), np.add(avg_rewards, std_rewards), alpha=0.2, label='Reward StdDev')

    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    xtick_labels = [str(i*ep_per_stat) for i in range(len(avg_rewards))]
    plt.xticks(range(1, len(avg_rewards)+1), xtick_labels)
    plt.xticks(rotation=45)
    ax.legend(loc='lower right')
    plt.show()



# run function implements the wandb logging
def train(training_mode=True, pretrained=False, lr=0.0001, gamma=0.90, exploration_decay=0.995, exploration_min=0.02, ep_per_stat = 100, exploration_max = 1, 
        lr_decay = 0.99, mario_env='SuperMarioBros-1-1-v0', action_space=TWO_ACTIONS_SET, num_episodes=1000, run_id=None, n_actions=20, debug = True, name=None, max_time_per_ep = 500, device=None):
    

    run_id = run_id or generate_epoch_time_id()
    # from looking at the model, time starts at 400
    time_total = 400 #seconds
    time_taken = 0 #seconds

    wandb.init(
        # set the wandb project where this run will be logged
        project="my-awesome-project",
    
        # track hyperparameters and run metadata
        config={
        "name": name or run_id,
        "run_id": run_id,
        "lr": lr,
        "lr_decay": lr_decay,
        "exploration_decay": exploration_decay,
        "n_actions": n_actions,
        "gamma": gamma,
        "episodes": num_episodes,
        }
    )

    

    # fh = open(f'progress-{run_id}.txt', 'a') # suppressing this for local runs
    env = gym.make(mario_env)
    env = make_env(env, ACTION_SPACE)

    #todo: add agent params as a setting/create different agents in diff functions to run 
    exploration_max = min(1, max(exploration_max, exploration_min))

    agent = DQNAgent(
                     state_space=env.observation_space.shape,
                     action_space=action_space,
                     max_memory_size=30000,
                     batch_size=64,
                     gamma=gamma,
                     lr=lr,
                     dropout=None,
                     exploration_max=exploration_max,
                     exploration_min=exploration_min,
                     exploration_decay=exploration_decay,
                     double_dq=True,
                     pretrained=pretrained,
                     run_id=run_id,
                     n_actions=n_actions,
                     device=device
                     )
    

    # see if anyone can get this to work, i think it doesn't work on mps
    if device != 'mps':
        wandb.watch(agent.local_net, log_freq=100, log='all')

    # num_episodes = 10
    env.reset()
    total_rewards = []
    total_info = []
    avg_losses = []

    losses = []
    if pretrained:
        total_rewards = load_item(from_file='total_rewards-{}.pkl'.format(run_id))
        total_losses = load_item(from_file='total_losses-{}.pkl'.format(run_id))
        total_info = load_item(from_file='total_info-{}.pkl'.format(run_id))
    
    offset = len(total_rewards)   
    for iteration in tqdm(range(num_episodes)):
        ep_num = offset + iteration

        state = env.reset() 
        state = torch.Tensor([state])
        total_reward = 0
        steps = 0

        action_freq = {}
        while True:
            two_actions_index = agent.act(state)
            two_actions_vector = agent.cur_action_space[0, two_actions_index[0]]
            two_actions = vec_to_action(two_actions_vector.cpu()) # tuple of actions

            # debugging info
            key = " | ".join([",".join(i) for i in two_actions])
            if key in action_freq:
                action_freq[key] += 1
            else:
                action_freq[key] = 1
            
            steps += 1
            reward = 0
            info = None
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
        
            agent.remember(state, two_actions_index, reward, state_next, terminal)
            loss = agent.experience_replay(debug=debug)

            if loss != None:
                avg_loss_replay = torch.mean(loss).cpu().data.numpy().item(0)
                wandb.log({"average replay loss": avg_loss_replay})
                losses.append(avg_loss_replay)
            
            state = state_next
            if terminal or time_taken >= max_time_per_ep:
                break

        total_info.append(info)
        total_rewards.append(total_reward)
        if len(losses):
            avg_losses.append(np.mean(losses))
        if len(avg_losses):
            wandb.log({"average episode loss": avg_losses[-1]})

        avg_total_reward=0
        std_dev_total_reward=0
        if len(total_rewards)>ep_per_stat:
            avg_total_reward = np.average(total_rewards[-ep_per_stat:])
            std_dev_total_reward = np.std(total_rewards[-ep_per_stat:])

        losses = []

        time_taken = time_total - info["time"]

        wandb.log({"total reward" : total_reward, 
                   "current lr": agent.lr,
                   "current exploration": agent.exploration_rate,
                   "flag acquired": info['flag_get'],
                #    "game score": info['score'],
                   "time": time_taken,
                #    "time remaining on clock": info["time"],
                   "episode" : ep_num,
                   "avg total reward" : avg_total_reward,
                   "std dev total reward": std_dev_total_reward 
                   })
        


        agent.decay_lr(lr_decay)
        agent.decay_exploration()

        if training_mode and (ep_num % ep_per_stat) == 0 and ep_num != 0:
            save_checkpoint(agent, total_rewards, total_info, run_id)
        
        with open(f'actions_chosen-{run_id}.txt', 'a') as f:
            f.write("Action Frequencies for Episode {}, Exploration = {:4f}, Tot Reward = {}\n".format(ep_num + 1, agent.exploration_rate, total_reward))
            f.write(json.dumps(action_freq) + "\n\n")
        
        # if debug:
        #     with open(f'loss-{run_id}.txt', 'a') as f:
        #         f.write("loss: {}".format(loss))
    
    if training_mode:
        save_checkpoint(agent, total_rewards, total_info, run_id)
    
    env.close()
    # fh.close()
    
    if num_episodes > ep_per_stat:
        plot_rewards(ep_per_stat=ep_per_stat, total_rewards=total_rewards)

    # wandb.finish()

