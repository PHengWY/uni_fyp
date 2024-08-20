import random
import math
import time
from enum import Enum
import pygame
import sys
import os
from os import path
from os import listdir
from os.path import isfile, join
from collections import namedtuple

class CharAction(Enum):
    # might not use 1, 3, 5, 7

    ## MOVEMENT
    NORTH = 0  # Go up
    NORTHEAST = 1  # Go diagonally 45 degree away from north towards the right
    EAST = 2  # Go right
    SOUTHEAST = 3  # Go diagonally 45 degree downwards towards the right
    SOUTH = 4  # Go down
    SOUTHWEST = 5  # Go diagonally 45 degree downwards towards the left
    WEST = 6  # Go left
    NORTHWEST = 7  # Go diagonally 45 degree upwards towards the left

    ## OTHERS
    SHOOT = 8

class BoxHead:

    BG_COLOR = (0, 150, 200, 150)
    WIDTH, HEIGHT = 1400, 900
    FPS = 60
    PLAYER_SPEED = 2
    MONSTER_SPEED = 1

    def __init__(self):
        self.reset()
        self.last_action = ''
        self._pygame_initialise()

    ## helper functions for different parts of the program

    # function to load the tile image
    def get_background(self, file):
        self.bg_image = pygame.image.load(file)
        _, _, self.bg_width, self.bg_height = self.bg_image.get_rect()
        self.bg_tiles = []

        for i in range(self.WIDTH // self.bg_width + 1):
            for j in range(self.HEIGHT // self.bg_height + 1):
                self.tile_pos = (i * self.bg_width, j * self.bg_height)
                self.bg_tiles.append(self.tile_pos)

        return self.bg_tiles, self.bg_image

    def _pygame_initialise(self):
        pygame.display.set_caption('BoxHead')
        pygame.init()
        pygame.display.init()
        self.clock = pygame.time.Clock()
        self.window = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self.font = pygame.font.Font(None, 36)

        # initialise background
        self.bg_tiles, self.bg_image = self.get_background('images/Terrain/chosen_terrain.png')


    def reset(self):
        pass

    def perform_action(self, char_action:CharAction) -> bool:
        self.last_action = char_action

        # Move Robot to the next cell
        if char_action == CharAction.LEFT:
            if self.robot_pos[1] > 0:
                self.robot_pos[1] -= 1
        elif char_action == CharAction.RIGHT:
            if self.robot_pos[1] < self.grid_cols - 1:
                self.robot_pos[1] += 1
        elif char_action == CharAction.UP:
            if self.robot_pos[0] > 0:
                self.robot_pos[0] -= 1
        elif char_action == CharAction.DOWN:
            if self.robot_pos[0] < self.grid_rows - 1:
                self.robot_pos[0] += 1

    def render(self):

        self._emergency_event()

        # Render pygame background & environment
        # self.window.fill(self.BG_COLOR)

        # draw background tiles
        for tile in self.bg_tiles:
            self.window.blit(self.bg_image, tile)


        # Constantly updates the pygame environment per frame
        pygame.display.update()

        # Limit FPS
        self.clock.tick(self.FPS)

    def _emergency_event(self):
        # Event to quit the game when necessary
        get_event = pygame.event.get()
        for event in get_event:
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                # User hit escape key
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()

if __name__ == "__main__":
    boxHead = BoxHead()
    
    while True:
        boxHead.render()