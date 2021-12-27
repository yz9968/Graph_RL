from collections import namedtuple
import os, time
import numpy as np
import matplotlib.pyplot as plt
import gym
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logging.info('Using device: %s', device)
USE_CUDA = torch.cuda.is_available()

seed = 1
log_interval = 10
torch.manual_seed(seed)
Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state'])


class Actor(nn.Module):
    def __init__(self, args, agent_id):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(args.obs_shape[agent_id], 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.action_head = nn.Linear(128, args.action_shape[agent_id])

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        action_prob = F.softmax(self.action_head(x), dim=1)
        return action_prob


class Critic(nn.Module):
    def __init__(self, args, agent_id):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(args.obs_shape[agent_id], 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.state_value = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        value = self.state_value(x)
        return value


class PPO:
    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_update_time = 10

    def __init__(self, args, agent_id):
        super(PPO, self).__init__()
        self.args = args
        self.agent_id = agent_id
        self.buffer = []
        self.training_step = 0
        self.gamma = self.args.gamma
        self.batch_size = self.args.batch_size
        self.buffer_capacity = self.args.buffer_size
        # create actor-critic network
        self.actor_network = Actor(args, agent_id).to(device)
        self.critic_network = Critic(args, agent_id).to(device)

        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr=1e-3)
        self.critic_network_optimizer = optim.Adam(self.critic_network.parameters(), lr=3e-3)
        # create the dict for store the model
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        # path to save the model
        self.model_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        self.model_path = self.model_path + '/' + 'agent_%d' % agent_id
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        self.actor_model_name = '/30_actor_params.pkl'
        self.critic_model_name = '/30_critic_params.pkl'
        # 加载模型
        if self.training_step % 500 == 0 and self.training_step > 0:
            if os.path.exists(self.model_path + self.actor_model_name):
                self.actor_network.load_state_dict(torch.load(self.model_path + self.actor_model_name))
                self.critic_network.load_state_dict(torch.load(self.model_path + self.critic_model_name))
                print('Agent {} successfully loaded actor_network: {}'.format(self.agent_id,
                                                                              self.model_path + self.actor_model_name))
                print('Agent {} successfully loaded critic_networkwork: {}'.format(self.agent_id,
                                                                               self.model_path + self.critic_model_name))

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        with torch.no_grad():
            action_prob = self.actor_network(state)
        c = Categorical(action_prob)
        action = c.sample()
        return action.item(), action_prob[:, action.item()].item()

    def get_value(self, state):
        state = torch.from_numpy(state).to(device)
        with torch.no_grad():
            value = self.critic_network(state)
        return value.item()

    def save_param(self):
        model_path = os.path.join(self.args.save_dir, self.args.scenario_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, 'agent_%d' % self.agent_id)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.actor_network.state_dict(), model_path + self.actor_model_name)
        torch.save(self.critic_network.state_dict(),  model_path + self.critic_model_name)

    def store_transition(self, transition):
        self.buffer.append(transition)

    def update(self):
        state = torch.tensor([t.state for t in self.buffer], dtype=torch.float)
        action = torch.tensor([t.action for t in self.buffer], dtype=torch.long).view(-1, 1)
        reward = [t.reward for t in self.buffer]
        # update: don't need next_state
        # reward = torch.tensor([t.reward for t in self.buffer], dtype=torch.float).view(-1, 1)
        # next_state = torch.tensor([t.next_state for t in self.buffer], dtype=torch.float)
        old_action_log_prob = torch.tensor([t.a_log_prob for t in self.buffer], dtype=torch.float).view(-1, 1)
        R = 0
        Gt = []
        for r in reward[::-1]:
            R = r + self.gamma * R
            Gt.insert(0, R)
        Gt = torch.tensor(Gt, dtype=torch.float)
        # print("The agent is updateing....")
        for i in range(self.ppo_update_time):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), self.batch_size, False):
                # with torch.no_grad():
                Gt_index = Gt[index].view(-1, 1)
                V = self.critic_network(state[index].to(device))
                delta = Gt_index.to(device) - V
                advantage = delta.detach()
                # epoch iteration, PPO core!!!
                action_prob = self.actor_network(state[index].to(device)).gather(1, action[index].to(device))  # new policy

                ratio = (action_prob / old_action_log_prob[index].to(device))
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage

                # update actor network
                action_loss = -torch.min(surr1, surr2).mean()  # MAX->MIN desent
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_network.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # update critic network
                value_loss = F.mse_loss(Gt_index.to(device), V)
                self.critic_network_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_network.parameters(), self.max_grad_norm)
                self.critic_network_optimizer.step()
                self.training_step += 1

        if self.training_step > 0 and self.training_step % self.args.save_rate == 0:
            self.save_param()

        del self.buffer[:]  # clear experience

