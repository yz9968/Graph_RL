import torch
import os
import numpy as np
import logging
from collections import namedtuple
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state'])

class embedding(nn.Module):
    def __init__(self, args):
        super(embedding, self).__init__()
        intruder_num = args.n_agents - 1
        self.lstm = nn.LSTM(intruder_num * 9, 32)
        self.fc1 = nn.Linear(32 + 9, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 32)

    def forward(self, obs, obs_in):
        """
        :param obs: （9， ）
        :param obs_in: （(n-1)*9 ，1）
        :return:
        """
        obs_in = obs_in.view(-1, 1, len(obs_in))
        x, _ = self.lstm(obs_in)
        x = x.squeeze(dim=1)
        obs = obs.view(-1, len(obs))
        x = torch.cat([x, obs], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = x.view(x.size()[0], -1) # torch size [1, 32]
        return x

class Runner_PPO_LSTM:
    def __init__(self, args, env):
        self.args = args
        self.epsilon = args.epsilon
        self.max_step = args.max_episode_len
        self.env = env
        self.agents = self.env.agents
        self.agent_num = self.env.agent_num
        self.lstm = embedding(args)
        self.save_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def get_state(self, obs):
        """
        :param obs: numpy [agent_n, 9]
        :return: numpy [agent_n, 32]
        """
        state = []
        for i, agent in enumerate(self.agents):
            obs_own = obs[i] # (9, )
            obs_intruder = np.delete(obs, i, axis=0).reshape(-1, 1)
            encoder = self.lstm(torch.Tensor(obs_own), torch.Tensor(obs_intruder))
            ob = encoder.detach().numpy()[0] # (32, )
            state.append(ob)
        return np.array(state)

    def run(self):
        returns = []
        reward_total = []
        conflict_total = []
        collide_wall_total = []
        nmac_total = []
        success_total = []
        start = time.time()
        for episode in range(self.args.num_episodes):
            reward_episode = []
            steps = 0
            self.epsilon = max(0.05, self.epsilon - 0.0004)
            s = self.env.reset()
            s = self.get_state(s)
            print("current_episode {}".format(episode))
            while steps < self.max_step:
                if not self.env.simulation_done:
                    actions = []
                    action_probs = []
                    for i, agent in enumerate(self.agents):
                        action, action_prob = agent.policy.select_action(s[i])
                        actions.append(action)
                        action_probs.append(action_prob)

                    s_next, r, done, info = self.env.step(actions)
                    s_next = self.get_state(s_next)
                    for i, agent in enumerate(self.agents):
                        trans = Transition(s[i], actions[i], action_probs[i], r[i], s_next[i])
                        agent.policy.store_transition(trans)
                    s = s_next

                    reward_episode.append(sum(r) / 1000)
                else:
                    # print("robot_terminated_times:", self.env.agent_times)
                    if self.env.simulation_done:
                        print("all agent done!")
                        for i, agent in enumerate(self.agents):
                            if len(agent.policy.buffer) >= self.args.batch_size:
                                agent.policy.update()
                    break

            reward_total.append(sum(reward_episode))

            if episode > 0 and episode % self.args.evaluate_rate == 0:
                rew, info = self.evaluate()
                if episode % (5 * self.args.evaluate_rate) == 0:
                    self.env.render(mode='traj')
                returns.append(rew)
                conflict_total.append(info[0])
                collide_wall_total.append(info[1])
                success_total.append(info[2])
                nmac_total.append(info[3])
            self.env.conflict_num_episode = 0
            self.env.nmac_num_episode = 0

        end = time.time()
        print("花费时间", end - start)
        plt.figure()
        plt.plot(range(1, len(returns)), returns[1:])
        plt.xlabel('evaluate num')
        plt.ylabel('average returns')
        plt.savefig(self.save_path + '/15_train_return.png', format='png')
        np.save(self.save_path + '/15_train_returns', returns)

        fig, a = plt.subplots(2, 2)
        x = range(len(conflict_total))
        a[0][0].plot(x, conflict_total, 'b')
        a[0][0].set_title('conflict_num')
        a[0][1].plot(x, collide_wall_total, 'y')
        a[0][1].set_title('exit_boundary_num')
        a[1][0].plot(x, success_total, 'r')
        a[1][0].set_title('success_num')
        a[1][1].plot(x, nmac_total)
        a[1][1].set_title('nmac_num')
        plt.savefig(self.save_path + '/15_train_metric.png', format='png')
        np.save(self.save_path + '/15_train_conflict', conflict_total)

        plt.show()

    def evaluate(self):
        print("now is evaluate!")
        self.env.collision_num = 0
        self.env.exit_boundary_num = 0
        self.env.success_num = 0
        self.env.nmac_num = 0
        returns = []
        deviation = []
        for episode in range(self.args.evaluate_episodes):
            # reset the environment
            s = self.env.reset()
            s = self.get_state(s)
            rewards = 0
            for time_step in range(self.args.evaluate_episode_len):
                # self.env.render()
                if not self.env.simulation_done:
                    actions = []
                    for agent_id, agent in enumerate(self.agents):
                        action, action_prob = agent.policy.select_action(s[agent_id])
                        actions.append(action)
                    s_next, r, done, info = self.env.step(actions)
                    s_next = self.get_state(s_next)
                    rewards += sum(r)
                    s = s_next
                else:
                    dev = self.env.route_deviation_rate()
                    deviation.append(np.mean(dev))
                    break
            rewards = rewards / 10000
            returns.append(rewards)
            print('Returns is', rewards)
        print("conflict num :", self.env.collision_num)
        print("nmac num：", self.env.nmac_num)
        print("exit boundary num：", self.env.exit_boundary_num)
        print("success num：", self.env.success_num)
        print("路径平均偏差率：", np.mean(deviation))

        return sum(returns) / self.args.evaluate_episodes, (self.env.collision_num, self.env.exit_boundary_num, self.env.success_num, self.env.nmac_num)

    def evaluate_model(self):
        """
        对现有最新模型进行评估
        :return:
        """
        print("now evaluate the ppo model")
        conflict_total = []
        collide_wall_total = []
        success_total = []
        nmac_total = []
        self.env.collision_num = 0
        self.env.nmac_num = 0
        self.env.exit_boundary_num = 0
        self.env.success_num = 0
        returns = []
        deviation = []
        eval_episode = 100
        for episode in range(eval_episode):
            # reset the environment
            s = self.env.reset()
            rewards = 0
            s = self.get_state(s)
            for time_step in range(self.args.evaluate_episode_len):
                # self.env.render()
                if not self.env.simulation_done:
                    actions = []
                    for agent_id, agent in enumerate(self.agents):
                        action, action_prob = agent.policy.select_action(s[agent_id])
                        actions.append(action)
                    s_next, r, done, info = self.env.step(actions)
                    s_next = self.get_state(s_next)
                    rewards += sum(r)
                    s = s_next
                else:
                    dev = self.env.route_deviation_rate()
                    deviation.append(np.mean(dev))
                    break

            if episode > 0 and episode % 50 == 0:
                self.env.render(mode='traj')

            # plt.figure()
            # plt.title('collision_value——time')
            # x = range(len(self.env.collision_value))
            # plt.plot(x, self.env.collision_value)
            # plt.xlabel('timestep')
            # plt.ylabel('collision_value')
            # plt.savefig(self.save_path + '/collision_value/30_agent/' + str(episode) + 'collision_value.png', format='png')
            # np.save(self.save_path + '/collision_value/30_agent/' + str(episode) + 'collision_value.npy', self.env.collision_value)
            # plt.close()

            rewards = rewards / 1000
            returns.append(rewards)
            print('Returns is', rewards)
            print("conflict num :", self.env.collision_num)
            print("nmac num：", self.env.nmac_num)
            print("exit boundary num：", self.env.exit_boundary_num)
            print("success num：", self.env.success_num)
            conflict_total.append(self.env.collision_num)
            nmac_total.append(self.env.nmac_num)
            collide_wall_total.append(self.env.exit_boundary_num)
            success_total.append(self.env.success_num)
            self.env.collision_num = 0
            self.env.nmac_num = 0
            self.env.exit_boundary_num = 0
            self.env.success_num = 0

        plt.figure()
        plt.plot(range(1, len(returns)), returns[1:])
        plt.xlabel('evaluate num')
        plt.ylabel('average returns')
        # plt.savefig(self.save_path + '/15_eval_return.png', format='png')

        fig, a = plt.subplots(2, 2)
        x = range(len(conflict_total))
        ave_conflict = np.mean(conflict_total)
        ave_nmac = np.mean(nmac_total)
        ave_success = np.mean(success_total)
        ave_exit = np.mean(collide_wall_total)
        zero_conflict = sum(np.array(conflict_total) == 0)
        print("平均冲突数", ave_conflict)
        print("平均NMAC数", ave_nmac)
        print("平均成功率", ave_success / self.agent_num)
        print("平均出界率", ave_exit / self.agent_num)
        print("0冲突占比：", zero_conflict / len(conflict_total))
        print("平均偏差率", np.mean(deviation))
        a[0][0].plot(x, conflict_total, 'b')
        a[0][0].set_title('conflict_num')
        a[0][1].plot(x, collide_wall_total, 'y')
        a[0][1].set_title('exit_boundary_num')
        a[1][0].plot(x, success_total, 'r')
        a[1][0].set_title('success_num')
        a[1][1].plot(x, nmac_total)
        a[1][1].set_title('nmac_num')
        # plt.savefig(self.save_path + '/15_eval_metric.png', format='png')

        plt.show()