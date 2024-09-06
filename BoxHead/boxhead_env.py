import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env
import pygame

import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

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

        self.render_mode = render_mode
        self.game = bh.BoxHead(fps=self.metadata['render_fps'])

        # Define action space (6 actions)
         # 0: up, 1: down, 2: left, 3: right, 4: shoot
        self.action_space = spaces.Discrete(len(CharAction))

        # Define observation space (example: player position and monster positions)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0]),
            high=np.array([BoxHead.WIDTH, BoxHead.HEIGHT, # player position
                           BoxHead.WIDTH, BoxHead.HEIGHT, # monster position
                           100]), # points
            shape=(5,),
            dtype=np.float32)

    def reset(self):
        self.game.reset()
        return self._get_observation()
    
    def step(self, action):
        char_action = CharAction(action)
        self.game.perform_action(char_action)

        # Update game state
        self.game.step()

        # Get observation
        observation = self._get_observation()
        reward = self._get_reward()
        done = self._is_done()
        info = {}

        return observation, reward, done, False, info
    
    def _get_observation(self):
        # Extract relevant game state information
        player_x, player_y = self.game.player.get_position()
        monster_x, monster_y = self.game.monster.get_position()
        return np.array([player_x, player_y, monster_x, monster_y, self.game.points])
    
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
        self.pygame.quit()

if __name__=="__main__":
    env = gym.make('boxhead', render_mode='human')

    # Use this to check our custom environment
    print("Check environment begin")
    # check_env(env.unwrapped)
    print("Check environment end")

    # Reset environment
    obs = env.reset()[0]

    # Take some random actions
    clock = pygame.time.Clock()
    while(True):
        rand_action = env.action_space.sample()
        obs, reward, terminated, _, _ = env.step(rand_action)

        if(terminated):
            obs = env.reset()[0]

        clock.tick(60)  # Limit the game loop to 60 FPS