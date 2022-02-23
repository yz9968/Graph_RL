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
        device = self.args.device
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
        nmac_total = []
        success_total = []
        start = time.time()
        for episode in range(self.args.num_episodes):
            reward_episode = []
            self.epsilon = max(0.05, self.epsilon - 0.00016)
            s = self.env.reset()
            print("current_episode {}".format(episode))
            for steps in range(self.max_step):
                if steps % 20 == 0:
                    print('episode: {}, step: {}'.format(episode,steps))
                self.noise = max(0.05, self.noise - 0.0000005)
                if not self.env.simulation_done:
                    # self.env.render(mode='traj')
                    self.env.m_render()
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
                # if episode % (5 * self.args.evaluate_rate) == 0:
                #     self.env.render(mode='traj')
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
        plt.savefig(self.save_path + '/{}_train_return.png'.format(self.args.n_agents), format='png')
        np.save(self.save_path + '/{}_train_returns.pkl'.format(self.args.n_agents), returns)

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
        plt.savefig(self.save_path + '/{}_train_metric.png'.format(self.args.n_agents), format='png')
        np.save(self.save_path + '/{}_train_returns.pkl'.format(self.args.n_agents), conflict_total)

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
            rewards = 0
            for time_step in range(self.args.evaluate_episode_len):
                # self.env.render()
                self.env.m_render()
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
                    dev = self.env.route_deviation_rate()
                    deviation.append(np.mean(dev))
                    break
            rewards = rewards / 10000
            returns.append(rewards)
            print('Returns is', rewards)
        print("conflict num :", self.env.collision_num)
        print("nmac num", self.env.nmac_num)
        print("exit boundary num：", self.env.exit_boundary_num)
        print("success num：", self.env.success_num)
        print("路径平均偏差率：", np.mean(deviation))

        return sum(returns) / self.args.evaluate_episodes, (self.env.collision_num, self.env.exit_boundary_num, self.env.success_num, self.env.nmac_num)

    def evaluate_model(self):
        """
        对现有最新模型进行评估
        :return:
        """
        print("now evaluate the model")
        conflict_total = []
        collide_wall_total = []
        success_total = []
        deviation = []
        nmac_total = []
        self.env.collision_num = 0
        self.env.exit_boundary_num = 0
        self.env.success_num = 0
        self.env.nmac_num = 0
        returns = []
        eval_episode = 100
        
        time_num = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))

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
                    dev = self.env.route_deviation_rate()
                    deviation.append(np.mean(dev))
                    break

            # if episode > 0 and episode % 50 == 0:
            #     self.env.render(mode='traj')

            # plt.figure()
            # plt.title('collision_value——time')
            # x = range(len(self.env.collision_value))
            # plt.plot(x, self.env.collision_value)
            # plt.xlabel('timestep')
            # plt.ylabel('collision_value')
            # plt.savefig(self.save_path + '/collision_value/30_agent/' + str(episode) + 'collision_value.png',
            #             format='png')
            # np.save(self.save_path + '/collision_value/30_agent/' + str(episode) + 'collision_value.npy',
            #         self.env.collision_value)
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
            self.env.exit_boundary_num = 0
            self.env.success_num = 0
            self.env.nmac_num = 0

        plt.figure()
        plt.plot(range(1, len(returns)), returns[1:])
        plt.xlabel('evaluate num')
        plt.ylabel('average returns')
        # plt.savefig(self.save_path + '/30_eval_return.png', format='png')

        fig, a = plt.subplots(2, 2)
        x = range(len(conflict_total))
        ave_conflict = np.mean(conflict_total)
        ave_nmac = np.mean(nmac_total)
        ave_success = np.mean(success_total)
        ave_exit = np.mean(collide_wall_total)
        zero_conflict = sum(np.array(conflict_total) == 0)

        np.save(
            self.save_path + '/{}_agent_evaluate_metrics_{}.npy'.format(self.agent_num, time_num),
            [ave_conflict, ave_nmac, ave_success/self.agent_num, ave_exit/self.agent_num,
             zero_conflict/len(conflict_total), np.mean(deviation)]
        )

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
        plt.savefig(self.save_path + '/{}_eval_metric_02211034.png'.format(self.args.n_agents), format='png')

        plt.show()