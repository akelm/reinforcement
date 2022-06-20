import random

import gym
from gym import spaces

# dane z Hasselt, Hado. "Double Q-learning." Advances in neural information processing systems 23 (2010).

class Gridworld33Env(gym.Env):
    def __init__(self):
        self.height = 3
        self.width = 3
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Tuple((
                spaces.Discrete(self.height),
                spaces.Discrete(self.width)
                ))
        self.moves = {
                0: (-1, 0),  # up
                1: (0, 1),   # right
                2: (1, 0),   # down
                3: (0, -1),  # left
                }
        self.reward_range = (-12, 10)
        # begin in start state
        self.reset()

    def step(self, action):
        x, y = self.moves[action]
        self.S = self.S[0] + x, self.S[1] + y

        self.S = max(0, self.S[0]), max(0, self.S[1])
        self.S = (min(self.S[0], self.height - 1),
                  min(self.S[1], self.width - 1))

        if self.S == (2, 2):
            return self.S, 5, True, {}

        reward = random.choice( (10, -12) )
        return self.S, reward, False, {}

    def reset(self):
        self.S = (0, 0)
        return self.S
