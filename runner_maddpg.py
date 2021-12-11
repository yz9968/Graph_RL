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
        for episode in range(self.args.num_episodes):
            reward_episode = []
            steps = 0
            self.epsilon = max(0.05, self.epsilon - 0.0001)

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
                returns.append(rew)
                conflict_total.append(info[0])
                collide_wall_total.append(info[1])
                success_total.append(info[2])

            if episode % 50 == 0:
                self.env.render(mode='traj')

        plt.figure()
        plt.plot(range(1, len(returns)), returns[1:])
        plt.xlabel('evaluate num')
        plt.ylabel('average returns')
        plt.savefig(self.save_path + '/return.png', format='png')
        np.save(self.save_path + '/returns.pkl', returns)

        fig, a = plt.subplots(2, 2)
        x = range(len(conflict_total))
        a[0][0].plot(x, conflict_total, 'b')
        a[0][0].set_title('conflict_num')
        a[0][1].plot(x, collide_wall_total, 'y')
        a[0][1].set_title('exit_boundary_num')
        a[1][0].plot(x, success_total, 'r')
        a[1][0].set_title('success_num')
        plt.savefig(self.save_path + '/metric.png', format='png')

        plt.show()

        # for time_step in range(self.args.time_steps):
        #     # reset the environment
        #     if time_step % self.max_step == 0:
        #         s = self.env.reset()
        #     u = []
        #     actions = []
        #     with torch.no_grad():
        #         for agent_id, agent in enumerate(self.agents):
        #             action = agent.select_action(s[agent_id], self.noise, self.epsilon)
        #             u.append(action)
        #             actions.append(action)
        #     for i in range(self.args.n_agents, self.args.n_players):
        #         actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
        #     s_next, r, done, info = self.env.step(actions)
        #
        #     self.buffer.store_episode(s[:self.args.n_agents], u, r[:self.args.n_agents], s_next[:self.args.n_agents])
        #     s = s_next
        #     if self.buffer.current_size >= self.args.batch_size:
        #         transitions = self.buffer.sample(self.args.batch_size)
        #         for agent in self.agents:
        #             other_agents = self.agents.copy()
        #             other_agents.remove(agent)
        #             agent.learn(transitions, other_agents)
        #     if time_step > 0 and time_step % self.args.evaluate_rate == 0:
        #         returns.append(self.evaluate())
        #         plt.figure()
        #         plt.plot(range(len(returns)), returns)
        #         plt.xlabel('episode * ' + str(self.args.evaluate_rate / self.episode_limit))
        #         plt.ylabel('average returns')
        #         plt.savefig(self.save_path + '/plt.png', format='png')
        #     self.epsilon = max(0.05, self.epsilon - 0.0000005)
        #     np.save(self.save_path + '/returns.pkl', returns)

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
            rewards = rewards / 1000
            returns.append(rewards)
            print('Returns is', rewards)
        print("conflict num :", self.env.collision_num)
        print("exit boundary num：", self.env.exit_boundary_num)
        print("success num：", self.env.success_num)

        return sum(returns) / self.args.evaluate_episodes, (self.env.collision_num, self.env.exit_boundary_num, self.env.success_num)
