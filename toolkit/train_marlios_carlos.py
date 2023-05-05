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
from toolkit.constants_carlos import *
from toolkit.train_test_samples import *
import wandb
import time
import warnings
import plotly.graph_objs as go

warnings.filterwarnings('ignore')

CONSECUTIVE_ACTIONS = 2

def show_state(env, ep=0, info=""):
    plt.figure(3)
    plt.clf()
    plt.imshow(env.render(mode='rgb_array'))
    plt.title("Episode: %d %s" % (ep, info))
    plt.axis('off')
    display(plt.gcf(), clear=True)

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

def save_checkpoint(agent, total_rewards, terminal_info, avg_loss, run_id):
    with open(f"ending_position-{run_id}.pkl", "wb") as f:
        pickle.dump(agent.ending_position, f)
    with open(f"num_in_queue-{run_id}.pkl", "wb") as f:
        pickle.dump(agent.num_in_queue, f)
    with open(f"total_rewards-{run_id}.pkl", "wb") as f:
        pickle.dump(total_rewards, f)
    with open(f"terminal_info-{run_id}.pkl", "wb") as f:
        pickle.dump(terminal_info, f)
    with open(f"avg_loss-{run_id}.pkl", "wb") as f:
        pickle.dump(avg_loss, f)
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


def plot_loss(ep_per_stat=100, avg_loss=[], from_file=None):
    if from_file != None:
        avg_loss = load_item(from_file)

    # avg_loss = [np.mean(avg_loss[i:i+ep_per_stat]) for i in range(0, len(avg_loss), ep_per_stat)]
    # std_loss = [np.std(avg_loss[i:i+ep_per_stat]) for i in range(0, len(avg_loss), ep_per_stat)]

    fig, ax = plt.subplots()
    # ax.plot(avg_loss, label='Average loss')
    ax.plot(avg_loss, label='Loss')
    # ax.fill_between(range(len(avg_loss)), np.subtract(avg_loss, std_loss), np.add(avg_loss, std_loss), alpha=0.2, label='Reward StdDev')

    ax.set_xlabel('Episode')
    ax.set_ylabel('Loss')
    xtick_labels = [str(i*ep_per_stat)
                    for i in range(len(avg_loss) // ep_per_stat)]
    plt.xticks(range(0, len(avg_loss), ep_per_stat), xtick_labels)
    ax.legend(loc='lower right')
    plt.show()


# run function implements the wandb logging
def train(
        training_mode=True, pretrained=False, lr=0.0001, gamma=0.90, exploration_decay=0.995,
        exploration_min=0.02, ep_per_stat=100, exploration_max=1, sample_actions=True,
        mario_env='SuperMarioBros-1-1-v0', action_space=TRAIN_SET, num_episodes=1000,
        run_id=None, n_actions=20, debug = True, name=None, max_time_per_ep = 500,
        device=None, sample_step=False, lr_min=0.00001
    ):
    

    run_id = run_id or generate_epoch_time_id()
    # from looking at the model, time starts at 400
    time_total = 400 #seconds
    time_taken = 0 #seconds
    lr_decay = (lr_min / lr) ** (2 / num_episodes)

    # fh = open(f'progress-{run_id}.txt', 'a') # suppressing this for local runs
    env = gym.make(mario_env)
    env = make_env(env, ACTION_SPACE)

    #todo: add agent params as a setting/create different agents in diff functions to run 
    exploration_max = min(1, max(exploration_max, exploration_min))

    # Convert the actions to tuples of tuples
    action_set_tuples = [tuple(map(tuple, action)) for action in action_space]

    # Initialize a dictionary to store the cumulative action count
    cumulative_action_count = {}
    episode_action_count = {}

    # Count the occurrences of each unique action
    for action in action_set_tuples:
        cumulative_action_count[action] = 0
        episode_action_count[action] = 0

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
                     device=device,
                     init_max_time=max_time_per_ep,
                     sample_actions=sample_actions,
                     lr_min=lr_min,
                     lr_decay=lr_decay
                     )

    wandb.init(
        # set the wandb project where this run will be logged
        project="my-awesome-project",
    
        # track hyperparameters and run metadata
        config={
            "name": name or run_id,
            "run_id": run_id,
            "model_architecture": str(agent.local_net),
            "lr": lr,
            "lr_decay": agent.lr_decay,
            "min_lr": agent.min_lr,
            "exploration_decay": exploration_decay,
            "n_actions": n_actions,
            "gamma": gamma,
            "episodes": num_episodes,
            "ep_per_stat": ep_per_stat
        }
    )    

    # see if anyone can get this to work, i think it doesn't work on mps
    if device != 'mps':
        wandb.watch(agent.local_net, log_freq=100, log='all')

    # num_episodes = 10
    env.reset()
    total_rewards = []
    total_info = []
    avg_losses = [0]
    avg_rewards = [0]
    avg_stdevs = [0]
    avg_completion = [0]
    total_rewards_val = []
    total_info_val = []
    avg_stdevs_val = [0]
    avg_rewards_val = [0]
    avg_completion_val = [0]
    ep_per_stat_val = ep_per_stat // 10

    losses = []
    if pretrained:
        total_rewards = load_item(from_file='total_rewards-{}.pkl'.format(run_id))
        avg_losses = load_item(from_file='avg_loss-{}.pkl'.format(run_id))
        # avg_losses = load_item(from_file='avg_losses-{}.pkl'.format(run_id))
        # total_info = load_item(from_file='total_info-{}.pkl'.format(run_id))
    
    offset = len(total_rewards)   
    for iteration in tqdm(range(num_episodes)):
        ep_num = offset + iteration

        state = env.reset() 
        state = torch.Tensor([state])
        total_reward = 0
        steps = 0

        action_freq = {}
        while True:
            
            # if steps%100 == 0 and steps>0:
            #     agent.decay_exploration()

            if sample_step:
                agent.subsample_actions() # subsample actions every step
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
                    if info["flag_get"] and terminal:
                        cur_reward += 500
                    total_reward += cur_reward
                    reward += cur_reward
                    
            state_next = torch.Tensor([state_next])
            reward = torch.tensor([reward]).unsqueeze(0)        
            terminal = torch.tensor([int(terminal)]).unsqueeze(0)
            time_taken = time_total - info["time"]
            
            # update action count
            cumulative_action_count[two_actions] += 1
            episode_action_count[two_actions] += 1
        
            agent.remember(state, two_actions_index, reward, state_next, terminal)
            loss = agent.experience_replay(debug=debug)

            if loss != None:
                # agent.decay_exploration()
                avg_loss_replay = torch.mean(loss).cpu().data.numpy().item(0)
                # wandb.log({"average replay loss": avg_loss_replay})
                losses.append(avg_loss_replay)
            
            state = state_next
            if terminal or time_taken >= agent.max_time_per_ep:
                break

        total_info.append(info)
        total_rewards.append(total_reward)
        # Gather loss stats
        if len(losses):
            avg_losses.append(np.mean(losses))
        # if len(avg_losses):
        #     wandb.log({"average episode loss": avg_losses[-1]})
        # gather average reward per eg:100 episodes stat
        # if len(total_rewards)%ep_per_stat == 0 and iteration > 0:
        avg_rewards.append(np.average(total_rewards[-ep_per_stat:]))
        avg_stdevs.append(np.std(total_rewards[-ep_per_stat:]))  
        avg_completion.append(np.average([i['flag_get'] for i in total_info[-ep_per_stat:]]))
         
       
        losses = []


        # plot the line charts:
        time_taken = time_total - info["time"]
        
        # if len(avg_rewards):
        #     ub = [i + j for i, j in zip(avg_rewards, avg_stdevs)]
        #     lb = [i - j for i, j in zip(avg_rewards, avg_stdevs)]
            # wandb.log({"my_custom_id" : wandb.plot.line_series(
            #             xs=[i for i in range(0, ep_num, ep_per_stat)], 
            #             ys=[avg_rewards, ub, lb],
            #             keys=["Avg Total Rewards", "upper std", "lower std"],
            #             title="Avg Rewards per {} Episodes".format(ep_per_stat),
            #             xname="episode ({}'s)".format(ep_per_stat))})
            
        # Create a stacked bar chart using Plotly
        data_episode_action_count = [go.Bar(x=[str(key) for key in episode_action_count.keys()], y=list(episode_action_count.values()), name="Actions")]
        data_cumul_act_dist = [go.Bar(x=[str(key) for key in cumulative_action_count.keys()], y=list(cumulative_action_count.values()), name="Actions")]

        layout_episode_action_count = go.Layout(
            title="Cumulative Action Distribution",
            xaxis=dict(title="Episode"),
            yaxis=dict(title="Count"),
            barmode="stack"
        )
        layout_cumul_act_dist = go.Layout(
            title="Cumulative Action Distribution",
            xaxis=dict(title="Episode"),
            yaxis=dict(title="Count"),
            barmode="stack"
        )

        fig_episode_action_count = go.Figure(data=data_episode_action_count, layout=layout_episode_action_count)
        fig_cumul_act_dist = go.Figure(data=data_cumul_act_dist, layout=layout_cumul_act_dist)

        wandb.log({"total reward" : total_reward, 
                   "current lr": agent.lr,
                   "current exploration": agent.exploration_rate,
                   "Avg Completion Rate": avg_completion[-1],
                   "time": time_taken,
                   "x_position": info['x_pos'],
                   "avg_loss": avg_losses[-1],
                   "max_time_per_ep": max_time_per_ep,
                   "avg_total_rewards": avg_rewards[-1],
                   "avg_std_dev": avg_stdevs[-1],
                   # Log action distribution at the end of the episode
                    "cumulative_action_distribution": fig_cumul_act_dist,
                    "episode_action_distribution": fig_episode_action_count,
                   })
        
        # Log cumulative action distribution at the end of the episode
        # wandb.log({
        #     "cumulative_action_distribution": wandb.plot.stacked_bar(
        #         chart_id="cumulative_action_distribution",
        #         keys=list(cumulative_action_count.keys()),
        #         values=list(cumulative_action_count.values()),
        #         title="Cumulative Action Distribution",
        #         xlabel="Episode",
        #         ylabel="Count"
        #     ),
            
        # })

        agent.decay_lr()
        agent.decay_exploration()
        if not sample_step:
            agent.subsample_actions() # subsample actions every episode

        # Run validation run every 10 episodes
        if ep_num % 10 == 0 and ep_num != 0:
            total_reward_val, info_val = validate_run(agent, env)
            total_rewards_val.append(total_reward_val)
            total_info_val.append(info_val)
            avg_rewards_val.append(np.average(total_rewards_val[-ep_per_stat_val:]))
            avg_stdevs_val.append(np.std(total_rewards_val[-ep_per_stat_val:]))  
            avg_completion_val.append(np.average([i['flag_get'] for i in total_info_val[-ep_per_stat_val:]]))

            wandb.log({
                "total_rewards_validation": total_rewards_val[-1],
                "avg_total_rewards_validation": avg_rewards_val[-1],
                "avg_std_dev_validation": avg_stdevs_val[-1],
                "Avg Completion Rate Validatiton": avg_completion_val[-1]
            })
        
        # update the max time per episode every 1000 episodes
        if ep_num % 500 == 0 and agent.max_time_per_ep < 450 and iteration>0:
            agent.max_time_per_ep += 50

        if training_mode and (ep_num % ep_per_stat) == 0 and ep_num != 0:
            save_checkpoint(agent, total_rewards, total_info, avg_losses, run_id)
        
        with open(f'actions_chosen-{run_id}.txt', 'a') as f:
            f.write("Action Frequencies for Episode {}, Exploration = {:4f}, Tot Reward = {}\n".format(ep_num + 1, agent.exploration_rate, total_reward))
            f.write(json.dumps(action_freq) + "\n\n")
    
    if training_mode:
        save_checkpoint(agent, total_rewards, total_info, avg_losses, run_id)
    
    env.close()
    # fh.close()
    
    if num_episodes > ep_per_stat:
        plot_rewards(ep_per_stat=ep_per_stat, total_rewards=total_rewards)

    wandb.finish()

def validate_run(agent, env):
    state = env.reset() 
    state = torch.Tensor([state])
    total_reward = 0
    agent.subsample_val_actions() # subsample actions every episode
    while True:
        # agent.subsample_val_actions() # subsample actions every step
        two_actions_index = agent.act_validate(state)
        two_actions_vector = agent.cur_val_action_space[0, two_actions_index[0]]
        two_actions = vec_to_action(two_actions_vector.cpu()) # tuple of actions
        
        reward = 0
        info = None
        terminal = False
        for action in two_actions: 
            if not terminal:
                # compute index into ACTION_SPACE of our action
                step_action = ACTION_TO_INDEX[action]

                state_next, cur_reward, terminal, info = env.step(step_action)
                if info["flag_get"] and terminal:
                    cur_reward += 500
                total_reward += cur_reward
                reward += cur_reward
                
        state_next = torch.Tensor([state_next])
        reward = torch.tensor([reward]).unsqueeze(0)        
        terminal = torch.tensor([int(terminal)]).unsqueeze(0)
        
        state = state_next
        if terminal:
            break

    return total_reward, info

def visualize(run_id, action_space, n_actions, lr=0.0001, exploration_min=0.02, ep_per_stat = 100, exploration_max = 0.1, mario_env='SuperMarioBros-1-1-v0',  num_episodes=1000, log_stats = False, randomness = True, sample_actions=True):
   
   
    fh = open(f'progress-{run_id}.txt', 'a')
    env = gym.make(mario_env)
    env = make_env(env, ACTION_SPACE)


    # observation_space = env.observation_space.shape # not using this anymore

    #todo: add agent params as a setting/create different agents in diff functions to run 
    if randomness:
        exploration_max = min(1, max(exploration_max, exploration_min))
    else:
        exploration_max = 0
        exploration_min = 0

    agent = DQNAgent(
                     state_space=env.observation_space.shape,
                     action_space=action_space,
                     max_memory_size=30000,
                     batch_size=64,
                     gamma=0.9,
                     lr=lr,
                     dropout=None,
                     exploration_max=exploration_max,
                     exploration_min=exploration_min,
                     exploration_decay=0.9995,
                     double_dq=True,
                     pretrained=True,
                     run_id=run_id,
                     n_actions=n_actions,
                     sample_actions=sample_actions,
                     mode=action_utils.TEST)
    
    
    # num_episodes = 10
    env.reset()
    total_rewards = []
    total_info = []
 
    for ep_num in tqdm(range(num_episodes)):
      
        state = env.reset() # take the final dimension of shape 
        state = torch.Tensor([state])# converts (1, 84, 84) to (1, 1, 84, 84)
        total_reward = 0
        steps = 0

        action_freq = {}
        while True:
            # agent.subsample_actions()
            show_state(env, ep_num)

            two_actions_index = agent.act(state)
            two_actions_vector = agent.cur_action_space[0, two_actions_index[0]]
            two_actions = vec_to_action(two_actions_vector.cpu()) # tuple of actions
            
            print(two_actions)

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
            
            
            state = state_next
            if terminal:
                break

        total_info.append(info)
        total_rewards.append(total_reward)
        agent.subsample_actions()

        if log_stats:
            with open(f'visualized_rewards-{run_id}.txt', 'a') as f:
                f.write("Total reward after episode {} is {}\n".format(ep_num + 1, total_rewards[-1]))
                if (ep_num%100 == 0):
                    f.write("==================\n")
                    f.write("{} current time at episode {}\n".format(datetime.datetime.now(), ep_num+1))
                    f.write("==================\n")
                #print("Total reward after episode {} is {}".format(ep_num + 1, total_rewards[-1]))
                num_episodes += 1
            
            with open(f'visualized_actions_chosen-{run_id}.txt', 'a') as f:
                f.write("Action Frequencies for Episode {}, Exploration = {:4f}\n".format(ep_num + 1, agent.exploration_rate))
                f.write(json.dumps(action_freq) + "\n\n")
    
    env.close()
    fh.close()
    
    if num_episodes > ep_per_stat:
        plot_rewards(ep_per_stat=ep_per_stat, total_rewards=total_rewards)