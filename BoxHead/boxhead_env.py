import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import boxhead as bh
from boxhead import BoxHead, CharAction

# register module
register(
    id='boxhead',                                # call it whatever you want
    entry_point='boxhead_env:BoxHeadEnv', # module_name:class_name
)

class BoxHeadEnv(gym.Env):
    # metadata is a required attribute
    # render_modes in our environment is either None or 'human'.
    metadata = {"render_modes": ["human"], 'render_fps': 4}

    def __init__(self, render_mode=None):
        super(BoxHeadEnv, self).__init__()

        # Define action space (example: discrete actions)
        self.action_space = spaces.Discrete(5)  # 0: up, 1: down, 2: left, 3: right, 4: shoot

        # Define observation space (example: player position and monster positions)
        self.observation_space = spaces.Box(low=0, high=1, shape=(6,), dtype=np.float32)  # Adjust shape as needed

        self.game = bh.BoxHead()

    def reset(self):
        self.game.reset()
        return self._get_observation()
    
    def step(self, action):
        # Map action to game action
        if action == 0:
            self.game.perform_action(bh.CharAction.NORTH)
        elif action == 1:
            self.game.perform_action(bh.CharAction.SOUTH)
        elif action == 2:
            self.game.perform_action(bh.CharAction.EAST)
        elif action == 3:
            self.game.perform_action(bh.CharAction.WEST)
        elif action == 4:
            self.game.perform_action(bh.CharAction.SHOOT)

        self.game.step()

        observation = self._get_observation()
        reward = self._get_reward()
        done = self._is_done()
        info = {}

        return observation, reward, done, info
    
    def _get_observation(self):
        # Extract relevant game state information
        player_x, player_y = self.game.player.get_position()
        monster_x, monster_y = self.game.monster.get_position()
        return np.array([player_x, player_y, monster_x, monster_y, self.game.points, self.game.game_over])
    
    def _get_reward(self):
        # Define your reward function (e.g., reward for hitting the monster, penalty for getting hit)
        if self.game.points > 0:
            return 1
        else:
            return -1

    def _is_done(self):
        return self.game.game_over

    def render(self, mode='human'):
        self.game.render()

    def close(self):
        pass



