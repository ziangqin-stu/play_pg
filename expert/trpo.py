import numpy as np
import torch
from torch.optim import Adam
import gym
import time


class TRPOBuffer:
    """
    A buffer for storing trajectories experienced by a TRPO agent interacting
    with the environment.
    """

    def __init__(self, capacity):
        """
        Init memory, may need refactor under different envs
        :param capacity: max storage number of trajectories
        """
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self):
        pass

    def sample(self):
        pass



class TRPO:
    """
    Trust Region Policy Optimization
    """

    # Logger
    # Random Seed
    # Env Instantiation
    # Construct the module
    # Instantiate Experience Buffer
    # Loss
    # Optimizers
    # Modle Saving
    # Update Func
    # Main Loop
    pass

if __name__ == '__main__':
    print("developing")