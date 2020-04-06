import torch
import gym


# Setup Envs
env =gym.make('CartPole-v0')
env.seed(123)


# Utils for Training
def select_actions(state):
    state = torch.from_numpy(state).unsqueeze(0)
    action_mean, _, action_std = policy_net