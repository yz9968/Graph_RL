import numpy as np
import matplotlib.pyplot as plt

dir = r"C:\File\Traffic\强化学习\Graph_RL\model"

maddpg_dir = dir + "cr_maddpg\\collision_value\\30_agent\\" + "86collision_value.npy"
ppo_dir = dir + "cr_ppo\\collision_value\\30_agent\\" + "60collision_value.npy"
grl_dir = dir + "cr_grl\\30_agent\\collision_value\\" + "78collision_value.npy"

maddpg = np.load(maddpg_dir).tolist()
ppo = np.load(ppo_dir).tolist()
grl = np.load(grl_dir).tolist()
max_len = 200
for i in range(200 - len(maddpg)):
    maddpg.append(0)
for i in range(200 - len(ppo)):
    ppo.append(0)
for i in range(200 - len(grl)):
    grl.append(0)

x = range(len(maddpg))
plt.figure()
plt.subplot(311)
plt.plot(x, maddpg, 'blue')
plt.title('maddpg')
plt.xlabel('timestep')
plt.ylabel('collision_value')
plt.subplot(312)
plt.plot(x, ppo, 'green')
plt.title('ppo')
plt.xlabel('timestep')
plt.ylabel('collision_value')
plt.subplot(313)
plt.plot(x, grl, 'red')
plt.title('grl')
plt.xlabel('timestep')
plt.ylabel('collision_value')

plt.show()