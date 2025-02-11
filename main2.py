from runner_dgn import Runner_DGN
from runner_maddpg import Runner_maddpg
from runner_ppo import Runner_PPO
from common.arguments import get_args
from common.utils import make_env
import numpy as np
import torch


if __name__ == '__main__':
    # get the params
    args = get_args()
    args.scenario_name="cr_ppo"
    
    # args.num_episodes = 500 # default 5001
    # args.evaluate_rate = 5 # default 50
    # args.evaluate_episodes = 5 # default 10
    # args.save_rate = 5 # default 500
    
    # need to try: 8 15 20 30 50
    args.n_agents = 50 # default 30 
    args.render=False

    env, args = make_env(args)
    # runner = Runner_DGN(args, env)
    # runner = Runner_maddpg(args, env)
    runner = Runner_PPO(args, env)
    # if args.evaluate:
    #     runner.evaluate_model()
    # else:
    #     runner.run()
    runner.evaluate_model()

