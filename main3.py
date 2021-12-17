from runner_dgn import Runner_DGN
from runner_ppo import Runner_PPO
from runner_maddpg import Runner_maddpg
from common.arguments import get_args
from common.utils import make_env
import numpy as np
import torch


if __name__ == '__main__':
    # get the params
    args = get_args()
    env, args = make_env(args)
    runner = Runner_DGN(args, env)
    # runner = Runner_PPO(args, env)
    if args.evaluate:
        runner.evaluate_model()
    else:
        runner.run()
