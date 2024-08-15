import random
from enum import Enum
import pygame
import sys
from os import path

class CharAction(Enum):
    ## MOVEMENT
    NORTH = 0 # Go up
    NORTHEAST = 1 # Go diagonally 45 degree away from north towards the right
    EAST = 2 # Go right
    SOUTHEAST = 3 # Go diagonally 45 degree downwards towards the right
    SOUTH = 4 # Go down
    SOUTHWEST = 5 # Go diagonally 45 degree downwards towards the left
    WEST = 6 # Go left
    NORTHWEST = 7 # Go diagonally 45 degree upwards towards the left

    ## OTHERS
    SHOOT = 8

class BoxHead:

    def __init__(self, fps=60):
        self.reset()

        self.fps = fps
        self.last_action=''
        self._pygame_initialise()

    def _pygame_initialise(self):
        
        pygame.display.set_caption('BoxHead')

        pygame.init()
        pygame.display.init()
        self.clock = pygame.time.Clock()

    def reset(self):
        pass

    def perform_action(self, char_action:CharAction) -> bool:

        self.last_action = char_action

        # Move Robot to the next cell
        if char_action == CharAction.LEFT:
            if self.robot_pos[1]>0:
                self.robot_pos[1]-=1
        elif char_action == CharAction.RIGHT:
            if self.robot_pos[1]<self.grid_cols-1:
                self.robot_pos[1]+=1
        elif char_action == CharAction.UP:
            if self.robot_pos[0]>0:
                self.robot_pos[0]-=1
        elif char_action == CharAction.DOWN:
            if self.robot_pos[0]<self.grid_rows-1:
                self.robot_pos[0]+=1
