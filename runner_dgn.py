from tqdm import tqdm
from dgn.buffer import ReplayBuffer
import torch
import torch.optim as optim
from dgn.DGN import DGN
import os
import numpy as np
import logging
import time


class Runner_DGN:
    def __init__(self, args, env):
        self.args = args
        self.env = env
        self.epsilon = args.epsilon
        self.num_episode = args.num_episodes
        self.max_step = args.max_episode_len
        self.agents = self.env.agents
        self.agent_num = self.env.agent_num
        self.buffer = ReplayBuffer(args.buffer_size)
        self.n_action = 3
        self.hidden_dim = 128
        self.lr = 1e-2
        self.batch_size = args.batch_size
        self.train_epoch = 5
        self.gamma = args.gamma
        self.observation_space = self.env.observation_space
        self.model = DGN(self.agent_num, self.observation_space, self.hidden_dim, self.n_action)
        self.model_tar = DGN(self.agent_num, self.observation_space, self.hidden_dim, self.n_action)
        self.model = self.model.cuda()
        self.model_tar = self.model_tar.cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.save_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
        logging.info('Using device: %s', device)
        USE_CUDA = torch.cuda.is_available()

    def run(self):
        Obs = np.ones((self.batch_size, self.agent_num, self.observation_space))
        Next_Obs = np.ones((self.batch_size, self.agent_num, self.observation_space))
        matrix = np.ones((self.batch_size, self.agent_num, self.agent_num))
        next_matrix = np.ones((self.batch_size, self.agent_num, self.agent_num))

        reward_total = []
        conflict_total = []
        collide_wall_total = []
        success_total = []
        start_episode = 100
        start = time.time()
        episode = -1
        rl_model_dir = 'graph_rl_weight.pth'
        while episode < self.num_episode:
            if episode > start_episode:
                self.epsilon = max(0.05, self.epsilon - 0.0004)

            reward_episode = []
            episode += 1
            step = 0
            obs, adj = self.env.reset()
            print("current episode {}".format(episode))
            while step < self.max_step:
                if not self.env.simulation_done:
                    # print(" {} episode {} step ".format(i_episode, steps))
                    step += 1
                    action = []
                    obs1 = np.expand_dims(obs, 0)  # shape （1, 6, 9(observation_space)）
                    adj1 = np.expand_dims(adj, 0)
                    q = self.model(torch.Tensor(obs1).cuda(), torch.Tensor(adj1).cuda())[0]  # shape (100, 3)
                    # 待改
                    for i, agent in enumerate(self.agents):
                        if agent.done == 0:
                            a = -1
                        elif np.random.rand() < self.epsilon:
                            a = np.random.randint(self.n_action)
                        else:
                            a = q[i].argmax().item()
                        action.append(a)

                    next_obs, next_adj, reward, done_signals, info = self.env.step(action)

                    self.buffer.add(obs, action, reward, next_obs, adj, next_adj, info['simulation_done'])
                    obs = next_obs
                    adj = next_adj
                    reward_episode.append(sum(reward) / 1000)
                else:
                    # print(" agent_terminated_times:", self.env.agent_times)
                    if self.env.simulation_done:
                        print("all agents done!")
                    break

            reward_total.append(sum(reward_episode))
            # print(" current done agent:", env.done_agent_idx)
            if episode % 5 == 0:
                # reward_episode_ave = sum(reward_total) / len(reward_total)
                # print(" {} i_episode ave reward is {}".format(i_episode, reward_episode_ave))
                # f.write(str(reward_episode_ave) + '\n')
                # reward_episode_ave = 0
                print(" {} i_episode ave reward is {}".format(episode, sum(reward_episode)))

            if episode % 5 == 0:
                print("conflict num :", self.env.collision_num)
                print("exit boundary num：", self.env.exit_boundary_num)
                print("success num：", self.env.success_num)
                conflict_total.append(self.env.collision_num)
                collide_wall_total.append(self.env.exit_boundary_num)  # max
                success_total.append(self.env.success_num)
                self.env.collision_num = 0
                self.env.exit_boundary_num = 0
                self.env.success_num = 0

            if episode % 50 == 0:
                self.env.render(mode='traj')

            if episode < start_episode:
                continue

            for epoch in range(self.train_epoch):
                loss_sum = 0
                batch = self.buffer.getBatch(self.batch_size)
                for j in range(self.batch_size):
                    sample = batch[j]
                    Obs[j] = sample[0]
                    Next_Obs[j] = sample[3]
                    matrix[j] = sample[4]
                    next_matrix[j] = sample[5]

                q_values = self.model(torch.Tensor(Obs).cuda(), torch.Tensor(matrix).cuda())  # shape (128, 6, 3)
                target_q_values = self.model_tar(torch.Tensor(Next_Obs).cuda(), torch.Tensor(next_adj).cuda()).max(dim=2)[0]  # shape  (128, 6)
                target_q_values = np.array(target_q_values.cpu().data)  # shape  (128, 6)
                expected_q = np.array(q_values.cpu().data)  # (batch_size, agent_num, action_num)

                for j in range(self.batch_size):
                    sample = batch[j]
                    for i in range(self.agent_num):
                        # sample[1]: action selection list ; sample[2]: reward size-agent_num ; sample[6]: terminated
                        expected_q[j][i][sample[1][i]] = sample[2][i] + (1 - sample[6]) * self.gamma * target_q_values[j][i]
                        # if sample[6][i] != 1:
                        #     expected_q[j][i][sample[1][i]] = sample[2][i] + self.gamma * target_q_values[j][i]
                        # else:
                        #     expected_q[j][i][sample[1][i]] = sample[2][i]

                loss = (q_values - torch.Tensor(expected_q).cuda()).pow(2).mean()
                loss_sum += loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if episode % 5 == 0:
                self.model_tar.load_state_dict(self.model.state_dict())

            if episode != 0 and episode % 2000 == 0:
                torch.save(self.model.state_dict(), rl_model_dir)
                print("torch save model for rl_weight")

        end = time.time()
        print("花费时间:", end - start)

        import matplotlib.pyplot as plt

        plt.figure('reward')
        plt.plot(np.arange(self.num_episode + 1), reward_total)
        plt.title('reward')
        plt.xlabel('i_episode')
        plt.ylabel('reward')
        plt.show()

        fig, a = plt.subplots(2, 2)
        x = np.arange(0, self.num_episode + 1, 5)
        a[0][0].plot(x, conflict_total, 'b')
        a[0][0].set_title('conflict_num')
        a[0][1].plot(x, collide_wall_total, 'y')
        a[0][1].set_title('exit_boundary_num')
        a[1][0].plot(x, success_total, 'r')
        a[1][0].set_title('success_num')
        # plt.legend()

        plt.show()
        # for time_step in tqdm(range(self.args.time_steps)):
        #     # reset the environment
        #     if time_step % self.episode_limit == 0:
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
        #     self.noise = max(0.05, self.noise - 0.0000005)
        #     self.epsilon = max(0.05, self.epsilon - 0.0000005)
        #     np.save(self.save_path + '/returns.pkl', returns)

    def evaluate(self):
        returns = []
        for episode in range(self.args.evaluate_episodes):
            # reset the environment
            s = self.env.reset()
            rewards = 0
            for time_step in range(self.args.evaluate_episode_len):
                self.env.render()
                actions = []
                with torch.no_grad():
                    for agent_id, agent in enumerate(self.agents):
                        action = agent.select_action(s[agent_id], 0, 0)
                        actions.append(action)
                for i in range(self.args.n_agents, self.args.n_players):
                    actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
                s_next, r, done, info = self.env.step(actions)
                rewards += r[0]
                s = s_next
            returns.append(rewards)
            print('Returns is', rewards)
        return sum(returns) / self.args.evaluate_episodes
