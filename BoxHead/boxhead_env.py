import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env
import pygame

import random
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

import boxhead as bh
from boxhead import BoxHead, CharAction

# register module
register(
    id='boxhead',                            
    entry_point='boxhead_env:BoxHeadEnv',
)

class BoxHeadEnv(gym.Env):
    # metadata is a required attribute
    # render_modes in our environment is either None or 'human'.
    metadata = {"render_modes": ["human"], 'render_fps': 60}

    def __init__(self, render_mode=None):
        super(BoxHeadEnv, self).__init__()

        self.render_mode = render_mode
        self.game = bh.BoxHead(fps=self.metadata['render_fps'])

        # Define action space (6 actions)
         # 0: up, 1: down, 2: left, 3: right, 4: shoot
        self.action_space = spaces.Discrete(len(CharAction))

        # Define observation space (example: player position and monster positions)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0]),
            high=np.array([BoxHead.WIDTH, BoxHead.HEIGHT, # player position
                           BoxHead.WIDTH, BoxHead.HEIGHT]), # monster position
            shape=(4,),
            dtype=np.int32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset(seed=seed)
        
        obs = self._get_observation()
        info = {}

        # Render environment
        if(self.render_mode=='human'):
            self.render()

        return obs, info
    
    def step(self, action):
        char_action = self.game.perform_action(bh.CharAction(action))

        # Get observation
        observation = self._get_observation()
        reward, done = self._get_reward(char_action)
        info = {}

        if(self.render_mode=='human'):
            self.render()

        return observation, reward, done, False, info
    
    def _get_observation(self):
        # Extract relevant game state information
        player_x, player_y = self.game.player.get_position()
        monster_x, monster_y = self.game.monster.get_position()
        return np.array([player_x, player_y, monster_x, monster_y])
    
    def _get_reward(self, action):
        # Define your reward function (e.g., reward for hitting the monster, penalty for getting hit)
        reward = 0
        done = False

        x, y = self.game.player.get_position()

        # define the reward
        if action:
            if self.game.points > 0: # bullet hits monster
                reward = 1
                done = True
            elif self.game.game_over: # monster hits player
                reward = -1
                done = True
            elif x < 0 + 32 or x > self.game.WIDTH - 32: # player somehow go out of bounds despite supposed collision
                reward = -1
                done = True
            elif y < 0 + 8 or y > self.game.HEIGHT - 8: # player somehow go out of bounds despite supposed collision
                reward = -1
                done = True
            else:
                reward = 0
                done = False
        else:
            reward = 0
            done = False
        
        return reward, done

    def _is_done(self):
        return self.game.game_over

    def render(self, mode='human'):
        self.game.render()  # call the render method of BoxHead

    def close(self):
        # self.pygame.quit()
        pass

if __name__=="__main__":
    env = gym.make('boxhead', render_mode='human')

    # Use this to check our custom environment
    print("Check environment begin")
    # check_env(env.unwrapped)
    print("Check environment end")

    # Reset environment
    obs = env.reset()[0]

    # Take some random actions
    while(True):
        random_action = env.action_space.sample()
        print(random_action)
        obs, reward, done, _, _ = env.step(random_action)
        print(f'Observation: {obs}, Reward: {reward}, Done: {done}')

        if(done):
            obs = env.reset()[0]
