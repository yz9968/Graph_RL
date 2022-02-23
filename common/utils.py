import numpy as np
import inspect
import functools
import torch
import logging

def store_args(method):
    """Stores provided method args as instance attributes.
    """
    argspec = inspect.getfullargspec(method)
    defaults = {}
    if argspec.defaults is not None:
        defaults = dict(
            zip(argspec.args[-len(argspec.defaults):], argspec.defaults))
    if argspec.kwonlydefaults is not None:
        defaults.update(argspec.kwonlydefaults)
    arg_names = argspec.args[1:]

    @functools.wraps(method)
    def wrapper(*positional_args, **keyword_args):
        self = positional_args[0]
        # Get default arg values
        args = defaults.copy()
        # Add provided arg values
        for name, value in zip(arg_names, positional_args[1:]):
            args[name] = value
        args.update(keyword_args)
        self.__dict__.update(args)
        return method(*positional_args, **keyword_args)

    return wrapper


def make_env(args):
    from multiagent_particle_envs.multiagent.environment import MultiAgentEnv, MultiAgentEnv_GRL, MultiAgentEnv_maddpg, MultiAgentEnv_ppo, MultiAgentEnv_ppo_cnn, MultiAgentEnv_ppo_lstm
    import multiagent_particle_envs.multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(args.scenario_name + ".py").Scenario()

    # create world
    world = scenario.make_world(args.n_agents)

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    logging.info('Using device: %s', device)
    USE_CUDA = torch.cuda.is_available()
    args.device = device
    # create multiagent environment
    if args.scenario_name == 'cr_grl':
        env = MultiAgentEnv_GRL(world, scenario.reset_world, scenario.reward)
    elif args.scenario_name == 'cr_maddpg':
        env = MultiAgentEnv_maddpg(world, scenario.reset_world, scenario.reward, args=args)
    elif args.scenario_name == 'cr_ppo':
        env = MultiAgentEnv_ppo(world, scenario.reset_world, scenario.reward, args=args)
    elif args.scenario_name == 'cr_ppo_cnn':
        env = MultiAgentEnv_ppo_cnn(world, scenario.reset_world, scenario.reward, args=args)
    else:
        env = MultiAgentEnv_ppo_lstm(world, scenario.reset_world, scenario.reward, args=args)
    # env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    # 以下部分添加到MultiAgentEnv中
    args.n_agents = env.agent_num
    args.obs_shape = [9 for _ in range(args.n_agents)]
    args.action_shape = [5 for _ in range(args.n_agents)]
    args.high_action = 1
    args.low_action = -1
    return env, args
