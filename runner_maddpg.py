from common.replay_buffer import Buffer
import torch
import os
import numpy as np
import logging
import time
import matplotlib.pyplot as plt

class Runner_maddpg:
    def __init__(self, args, env):
        self.args = args
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
        # logging.info('Using device: %s', self.device)
        # USE_CUDA = torch.cuda.is_available()
        self.noise = args.noise_rate
        self.epsilon = args.epsilon
        self.max_step = args.max_episode_len
        self.env = env
        self.agents = self.env.agents
        self.agent_num = self.env.agent_num
        self.buffer = Buffer(args)
        self.save_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def run(self):
        returns = []
        reward_total = []
        conflict_total = []
        collide_wall_total = []
        success_total = []
        start = time.time()
        for episode in range(self.args.num_episodes):
            reward_episode = []
            steps = 0
            self.epsilon = max(0.05, self.epsilon - 0.00001)

            s = self.env.reset()
            print("current_episode {}".format(episode))
            while steps < self.max_step:
                self.noise = max(0.05, self.noise - 0.0000005)
                if not self.env.simulation_done:
                    actions = []
                    u = []
                    with torch.no_grad():
                        for i, agent in enumerate(self.agents):
                            action = agent.select_action(s[i], self.noise, self.epsilon)
                            u.append(action)
                            actions.append(action)

                    s_next, r, done, info = self.env.step(actions)

                    self.buffer.store_episode(s, u, r, s_next)
                    s = s_next
                    if self.buffer.current_size >= self.args.batch_size:
                        transitions = self.buffer.sample(self.args.batch_size)
                        for agent in self.agents:
                            other_agents = self.agents.copy()
                            other_agents.remove(agent)
                            agent.learn(transitions, other_agents)
                    reward_episode.append(sum(r) / 1000)

                else:
                    # print("robot_terminated_times:", self.env.agent_times)
                    if self.env.simulation_done:
                        print("all agent done!")
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
            self.env.conflict_num_episode = 0

        end = time.time()
        print("花费时间", end - start)
        plt.figure()
        plt.plot(range(1, len(returns)), returns[1:])
        plt.xlabel('evaluate num')
        plt.ylabel('average returns')
        plt.savefig(self.save_path + '/train_return.png', format='png')
        np.save(self.save_path + '/train_returns.pkl', returns)

        fig, a = plt.subplots(2, 2)
        x = range(len(conflict_total))
        a[0][0].plot(x, conflict_total, 'b')
        a[0][0].set_title('conflict_num')
        a[0][1].plot(x, collide_wall_total, 'y')
        a[0][1].set_title('exit_boundary_num')
        a[1][0].plot(x, success_total, 'r')
        a[1][0].set_title('success_num')
        plt.savefig(self.save_path + '/train_metric.png', format='png')
        np.save(self.save_path + '/train_returns.pkl', conflict_total)

        plt.show()

    def evaluate(self):
        print("now is evaluate!")
        self.env.collision_num = 0
        self.env.exit_boundary_num = 0
        self.env.success_num = 0
        returns = []
        for episode in range(self.args.evaluate_episodes):
            # reset the environment
            s = self.env.reset()
            rewards = 0
            for time_step in range(self.args.evaluate_episode_len):
                # self.env.render()
                if not self.env.simulation_done:
                    actions = []
                    with torch.no_grad():
                        for agent_id, agent in enumerate(self.agents):
                            action = agent.select_action(s[agent_id], 0, 0)
                            actions.append(action)
                    s_next, r, done, info = self.env.step(actions)
                    rewards += sum(r)
                    s = s_next
                else:
                    break
            rewards = rewards / 10000
            returns.append(rewards)
            print('Returns is', rewards)
        print("conflict num :", self.env.collision_num)
        print("exit boundary num：", self.env.exit_boundary_num)
        print("success num：", self.env.success_num)

        return sum(returns) / self.args.evaluate_episodes, (self.env.collision_num, self.env.exit_boundary_num, self.env.success_num)

    def evaluate_model(self):
        """
        对现有最新模型进行评估
        :return:
        """
        print("now evaluate the model")
        conflict_total = []
        collide_wall_total = []
        success_total = []
        self.env.collision_num = 0
        self.env.exit_boundary_num = 0
        self.env.success_num = 0
        returns = []
        eval_episode = 100
        for episode in range(eval_episode):
            # reset the environment
            s = self.env.reset()
            rewards = 0
            for time_step in range(self.args.evaluate_episode_len):
                # self.env.render()
                if not self.env.simulation_done:
                    actions = []
                    with torch.no_grad():
                        for agent_id, agent in enumerate(self.agents):
                            action = agent.select_action(s[agent_id], 0, 0)
                            actions.append(action)
                    s_next, r, done, info = self.env.step(actions)
                    rewards += sum(r)
                    s = s_next
                else:
                    break

            if episode > 0 and episode % 10 == 0:
                self.env.render(mode='traj')

            # plt.figure()
            # plt.title('collision_value——time')
            # x = range(len(self.env.collision_value))
            # plt.plot(x, self.env.collision_value)
            # plt.xlabel('timestep')
            # plt.ylabel('collision_value')
            # plt.savefig(self.save_path + '/collision_value/8_agent/' + str(episode) + 'collision_value.png', format='png')

            rewards = rewards / 1000
            returns.append(rewards)
            print('Returns is', rewards)
            print("conflict num :", self.env.collision_num)
            print("exit boundary num：", self.env.exit_boundary_num)
            print("success num：", self.env.success_num)
            conflict_total.append(self.env.collision_num)
            collide_wall_total.append(self.env.exit_boundary_num)
            success_total.append(self.env.success_num)
            self.env.collision_num = 0
            self.env.exit_boundary_num = 0
            self.env.success_num = 0

        plt.figure()
        plt.plot(range(1, len(returns)), returns[1:])
        plt.xlabel('evaluate num')
        plt.ylabel('average returns')
        plt.savefig(self.save_path + '/8_duifeval_return.png', format='png')

        fig, a = plt.subplots(2, 2)
        x = range(len(conflict_total))
        ave_conflict = np.mean(conflict_total)
        print("平均冲突", ave_conflict)
        a[0][0].plot(x, conflict_total, 'b')
        a[0][0].set_title('conflict_num')
        a[0][1].plot(x, collide_wall_total, 'y')
        a[0][1].set_title('exit_boundary_num')
        a[1][0].plot(x, success_total, 'r')
        a[1][0].set_title('success_num')
        plt.savefig(self.save_path + '/8_eval_metric.png', format='png')

        plt.show()