import sys
import logging
import argparse
import configparser
import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from DGN import DGN
from buffer import ReplayBuffer
from config import *
import gym
import numpy as np
import time
from copy import deepcopy
import git

from crowd_sim.envs.utils.state import *
from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.robot import Robot

parser = argparse.ArgumentParser('Parse configuration file')
parser.add_argument('--env_config', type=str, default='configs/env.config')
parser.add_argument('--model_dir', type=str, default='data/output')
# parser.add_argument('--train_config', type=str, default='configs/train.config')
parser.add_argument('--output_dir', type=str, default='data/output')
# parser.add_argument('--phase', type=str, default='test')
parser.add_argument('--gpu', default=True, action='store_true')
parser.add_argument('--debug', default=False, action='store_true')
args = parser.parse_args()

# configure logging and device
logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s',
                    datefmt="%Y-%m-%d %H:%M:%S")
device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
logging.info('Using device: %s', device)
USE_CUDA = torch.cuda.is_available()

# configure environment
env_config = configparser.RawConfigParser()
env_config.read(args.env_config)
env = gym.make('collision_avoidance-v1')
env.configure(env_config)
robot = Robot(env_config, 'robots')
env.set_robot(robot)

agent_num = env.robot_num
observation_space = env.observation_space
n_action = env.action_num

buff = ReplayBuffer(capacity)
model = DGN(agent_num, observation_space, hidden_dim, n_action)
model_tar = DGN(agent_num, observation_space, hidden_dim, n_action)
model = model.cuda()
model_tar = model_tar.cuda()
optimizer = optim.Adam(model.parameters(), lr=lr)

Obs = np.ones((batch_size, agent_num, observation_space))
Next_Obs = np.ones((batch_size, agent_num, observation_space))
matrix = np.ones((batch_size, agent_num, agent_num))
next_matrix = np.ones((batch_size, agent_num, agent_num))

# rl model path
rl_model_dir = 'graph_rl_model.pth'

f = open('r.txt', 'w')
reward_total = []
loss_total = []
conflict_total = []
collide_wall_total = []
success_total = []
attempt_num = 0
reach_goal_num = 0
start = time.time()
while i_episode < n_episode:

    if i_episode > start_episode:
        epsilon -= epsilon_decay
        if epsilon < 0.1:
            epsilon = 0.1
    reward_episode = []
    i_episode += 1
    steps = 0
    terminated = False
    obs, adj = env.reset()
    # print("obs", obs.shape)
    print("current episode {} ".format(i_episode))
    # print("reach_goal_num", reach_goal_num)
    while steps < max_step:
        if not terminated:
            # print(" {} episode {} step ".format(i_episode, steps))
            steps += 1
            action = []
            obs1 = np.expand_dims(obs, 0) # shape （1, 6, 9(observation_space)）
            adj1 = np.expand_dims(adj, 0)
            q = model(torch.Tensor(obs1).cuda(), torch.Tensor(adj1).cuda())[0]  # shape (6, 3)
            for i in range(agent_num):
                if np.random.rand() < epsilon:
                    a = np.random.randint(n_action)
                else:
                    a = q[i].argmax().item()
                action.append(a)

            next_obs, next_adj, reward, done_signals, info = env.step(action)

            buff.add(obs, action, reward, next_obs, adj, next_adj, done_signals)
            obs = next_obs
            adj = next_adj

            reward_episode.append(sum(reward) / 10000)
        else:
            print(" robot_terminated_times:", env.robot_times)
            print(" all agents done!")
            break

    reward_total.append(sum(reward_episode))
    # print(" current done agent:", env.done_agent_idx)
    if i_episode % 5 == 0:
        # reward_episode_ave = sum(reward_total) / len(reward_total)
        # print(" {} i_episode ave reward is {}".format(i_episode, reward_episode_ave))
        # f.write(str(reward_episode_ave) + '\n')
        # reward_episode_ave = 0
        print(" {} i_episode ave reward is {}".format(i_episode, sum(reward_episode)))
        f.write(str(sum(reward_episode)) + '\n')


    if i_episode % 10 == 0:
        print("conflict num :", env.conflict_num)
        print("exit boundary num：", env.collide_wall_num)
        print("success num：", env.success_num)
        conflict_total.append(env.conflict_num)
        collide_wall_total.append(env.collide_wall_num) # max
        success_total.append(env.success_num)
        env.conflict_num = 0
        env.collide_wall_num = 0
        env.success_num = 0

    if i_episode % 50 == 0:
        env.render(mode='traj')

    if i_episode < start_episode:
        continue

    for epoch in range(n_epoch):
        loss_sum = 0
        batch = buff.getBatch(batch_size)
        for j in range(batch_size):
            sample = batch[j]
            Obs[j] = sample[0]
            Next_Obs[j] = sample[3]
            matrix[j] = sample[4]
            next_matrix[j] = sample[5]

        q_values = model(torch.Tensor(Obs).cuda(), torch.Tensor(matrix).cuda()) # shape (128, 6, 3)
        target_q_values = model_tar(torch.Tensor(Next_Obs).cuda(), torch.Tensor(next_adj).cuda()).max(dim=2)[0] # shape  (128, 6)
        target_q_values = np.array(target_q_values.cpu().data) # shape  (128, 6)
        expected_q = np.array(q_values.cpu().data) # (batch_size, agent_num, action_num)

        for j in range(batch_size):
            sample = batch[j]
            for i in range(agent_num):
                # sample[1]: action selection list ; sample[2]: reward size-agent_num ; sample[6]: terminated
                if sample[6][i] == 0 or 2:
                    expected_q[j][i][sample[1][i]] = sample[2][i] + gamma * target_q_values[j][i]
                    # expected_q[j][i][sample[1][i]] = sample[2][i] + (1 - sample[6]) * GAMMA * target_q_values[j][i]
                else:
                    expected_q[j][i][sample[1][i]] = sample[2][i]

        loss = (q_values - torch.Tensor(expected_q).cuda()).pow(2).mean()
        loss_sum += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_ave = loss_sum / n_epoch
    loss_total.append(loss_ave)
    print(" {} i_episode {} epoch loss_ave is {}".format(i_episode, n_epoch, loss_ave))


    if i_episode % 5 == 0:
        model_tar.load_state_dict(model.state_dict())

    if i_episode != 0 and i_episode % checkpoint_interval == 0:
        # torch.save(model.state_dict(), rl_model_dir)
        print("torch save model for rl_weight")

end = time.time()
print("花费时间:", end - start)

# print(len(reward_total))
# print(len(loss_total))
import matplotlib.pyplot as plt

plt.figure('reward')
plt.plot(np.arange(n_episode + 1), reward_total)
# plt.ylim((-2, 1))
plt.title('reward')
plt.xlabel('i_episode')
plt.ylabel('reward')
plt.show()


fig, a = plt.subplots(2,2)
x = np.arange(0, n_episode + 1, 10)
a[0][0].plot(x, conflict_total, 'b')
a[0][0].set_title('conflict_num')
a[0][1].plot(x, collide_wall_total, 'y')
a[0][1].set_title('exit_boundary_num')
a[1][0].plot(x, success_total, 'r')
a[1][0].set_title('success_num')
# plt.legend()

plt.show()


# plt.figure('loss')
# plt.plot(np.arange(start_episode, start_episode + len(loss_total)), loss_total)
# plt.xlabel('i_episode')
# plt.ylabel('loss')
# plt.show()

# if not loss_total:
#     plt.figure('loss')
#     plt.plot(np.arange(start_episode, start_episode + len(loss_total)), loss_total)
#     plt.xlabel('i_episode')
#     plt.ylabel('loss')
#     plt.show()




