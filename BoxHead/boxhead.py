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

    ## MOVEMENT
    NORTH = 0  # Go up
    EAST = 1  # Go right
    SOUTH = 2  # Go down
    WEST = 3  # Go left

    ## OTHERS
    SHOOT = 4

class Object(pygame.sprite.Sprite):
    def __init__(self, x, y, width, height, name=None):
        super().__init__()
        self.rect = pygame.Rect(x, y, width, height)
        self.image = pygame.Surface((width, height), pygame.SRCALPHA, 32)
        self.width = width
        self.height = height
        self.name = name

    def load_object(dir1, dir2, file, width, height, scaling=1):
        path = join(dir1, dir2, file)
        image = pygame.image.load(path).convert_alpha()
        surface = pygame.Surface((width, height), pygame.SRCALPHA, 32)
        rect = pygame.Rect(0, 0, width, height)
        surface.blit(image, (0, 0), rect)
        return pygame.transform.scale_by(surface, scaling)

    def draw(self, game_window):
        game_window.blit(self.image, (self.rect.x, self.rect.y))

def load_object(dir1, dir2, file, width, height, scaling=1):
    path = join(dir1, dir2, file)
    image = pygame.image.load(path).convert_alpha()
    surface = pygame.Surface((width, height), pygame.SRCALPHA, 32)
    rect = pygame.Rect(0, 0, width, height)
    surface.blit(image, (0, 0), rect)
    return pygame.transform.scale_by(surface, scaling)

class Wall(Object):
    def __init__(self, x, y, width, height, rotation=0):
        super().__init__(x, y, width, height)
        self.rotation = rotation
        wall = load_object('images', 'Terrain', 'chosen_wall.png', width, height)
        self.original_image = wall.copy()  # store original image
        self.image.blit(wall, (0, 0))

        # rotate image
        self.image = pygame.transform.rotate(self.original_image, self.rotation)
        self.mask = pygame.mask.from_surface(self.image)

# player class
class Player(pygame.sprite.Sprite):
    ANIMATION_SPEED = 10

    def __init__(self, x, y, width, height):
        super().__init__()
        self.rect = pygame.Rect(x, y, width, height)
        self.x_vel = 0
        self.y_vel = 0
        self.mask = None
        self.direction = 'up'
        self.fire_direction = 'up'
        # self.gun = Gun.PISTOL
        # self.current_ammo = self.gun.value.ammo_capacity
        self.animation_count = 0
        self.health = 90
        self.damage = 15
        self.last_fired = 0
        self.speed = 3
        self.dir_x = 0
        self.last_x = 0
        self.dir_y = 0
        self.last_y = 0
        self.fire_gun = 0

        self.SPRITES = Player._load_player_sprite_sheets(30, 34, True)

    def _load_player_sprite_sheets(width, height, direction=False):
        path = 'images/Player'
        images = [a for a in listdir(path) if isfile(join(path, a))]

        all_sprites = {}

        for image in images:
            sprite_sheet = pygame.image.load(join(path, image)).convert_alpha()
            sprites = []
            for b in range(sprite_sheet.get_width() // width):
                surface = pygame.Surface((width, height), pygame.SRCALPHA, 32)
                rect = pygame.Rect(b * width, 0, width, height)
                surface.blit(sprite_sheet, (0, 0), rect)
                sprites.append(surface)

            if direction:
                all_sprites[image.replace('MainGuySpriteSheet_', '').replace('.png', '')] = sprites
            else:
                all_sprites[image.replace('MainGuySpriteSheet_', '').replace('.png', '')] = sprites

        return all_sprites

    def move(self, dx, dy):
        self.rect.x += dx
        self.rect.y += dy

    def auto_move(self):
        self.last_x = self.rect.x
        self.last_y = self.rect.y
        self.rect.x += self.dir_x
        self.rect.y += self.dir_y
        self.dir_x = 0
        self.dir_y = 0
        self.update_sprite()

    def move_left(self, vel):
        self.x_vel = -vel
        if self.direction != 'left':
            self.direction = 'left'
            self.animation_count = 0

    def move_right(self, vel):
        self.x_vel = vel
        if self.direction != 'right':
            self.direction = 'right'
            self.animation_count = 0

    def move_up(self, vel):
        self.y_vel = -vel
        if self.direction != 'up':
            self.direction = 'up'
            self.animation_count = 0

    def move_down(self, vel):
        self.y_vel = vel
        if self.direction != 'down':
            self.direction = 'down'
            self.animation_count = 0

    def action(self, action):
        if action == 1:  # left
            self.dir_x = -1
        elif action == 2:  # right
            self.dir_x = 1
        elif action == 3:  # up
            self.dir_y = -1
        elif action == 4:  # down
            self.dir_y = 1
        elif action == 5:  # fire gun
            self.fire_gun = 1
        else:
            self.dir_x = 0
            self.dir_y = 0  # stationary
            self.fire_gun = 0

    def loop(self, fps):
        self.move(self.x_vel, self.y_vel)
        self.update_sprite()

    def update_sprite(self):
        sprites = self.SPRITES[self.direction]
        sprite_index = (self.animation_count // self.ANIMATION_SPEED) % len(sprites)
        self.sprite = sprites[sprite_index]
        self.animation_count += 1
        self.update_mask()

    def update_mask(self):
        self.rect = self.sprite.get_rect(topleft=(self.rect.x, self.rect.y))
        self.mask = pygame.mask.from_surface(self.sprite)

    def draw(self, game_window):
        game_window.blit(self.sprite, (self.rect.x, self.rect.y))

    def get_position(self):
        return self.rect.x, self.rect.y
    
# monster class
class Monster(pygame.sprite.Sprite):

    def __init__(self, x, y, width, height):
        super().__init__()
        self.rect = pygame.Rect(x, y, width, height)
        self.x_vel = 0
        self.y_vel = 0
        self.mask = None
        self.monster_sprite = load_object('images', 'Monster', 'monster_a.png', 16, 16, 2)
        self.sprite = self.monster_sprite
        self.direction = 'down'
        self.speed = 1

    def move(self, dx, dy):
        self.rect.x += dx
        self.rect.y += dy

    def loop(self, fps):
        self.move(self.x_vel, self.y_vel)
        self.update_sprite()

    def update_sprite(self):
        self.sprite = self.monster_sprite
        self.update_mask()

    def update_mask(self):
        self.rect = self.sprite.get_rect(topleft=(self.rect.x, self.rect.y))
        self.mask = pygame.mask.from_surface(self.sprite)

    def draw(self, game_window):
        game_window.blit(self.sprite, (self.rect.x, self.rect.y))

    def get_position(self):
        return self.rect.x, self.rect.y
    
class Bullet(pygame.sprite.Sprite):
    def __init__(self, x, y, width, height, direction):
        super().__init__()
        self.rect = pygame.Rect(x, y, width, height)
        self.image = pygame.Surface((width, height))
        self.image.fill((255, 255, 0))  # Yellow color
        self.speed = 15
        self.direction = direction

    def update(self):
        if self.direction == 'up':
            self.rect.y -= self.speed
        elif self.direction == 'down':
            self.rect.y += self.speed
        elif self.direction == 'left':
            self.rect.x -= self.speed
        elif self.direction == 'right':
            self.rect.x += self.speed

    def draw(self, game_window):
        game_window.blit(self.image, self.rect.topleft)

    def has_collided_with_monster(self, monster):
        return pygame.sprite.collide_rect(self, monster)

class BoxHead:

    BG_COLOR = (0, 150, 200, 150)
    # WIDTH, HEIGHT = 1400, 900
    WIDTH, HEIGHT = 1280, 720
    PLAYER_SPEED = 5 # 2 for pc
    MONSTER_SPEED = 3 # 1 for pc

    def __init__(self, fps):
        self.reset()
        self.fps = fps
        self.last_action = ''
        self._pygame_initialise()
        self.player = Player(self.WIDTH / 2 - 25, self.HEIGHT / 2 - 25, 50, 50)
        self.monster = Monster(self.WIDTH / 2 - 25, self.HEIGHT / 2 - 225, 16, 16)

        self.bullets = []  # List to store bullets
        self.last_bullet_fired = 0  # Time when the last bullet was fired

        # Initialize the game over flag
        self.game_over = False

        self.points = 0

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
    
    # function to print the points gained
    def points_text(self, game_window):
        points_text = self.font.render(f"Points: {self.points}", True, (0, 0, 0, 150))
        game_window.blit(points_text, (self.WIDTH - points_text.get_width() - 70, 15))

    # function for horizontal collision
    def collide_horizontal(self, avatar, objects, dx, offset=0):
        avatar.move(dx + offset, 0)
        avatar.update_mask()
        self.collided_object = None
        for obj in objects:
            if hasattr(obj, 'mask') and obj.mask is not None:
                if pygame.sprite.collide_mask(avatar, obj):
                    self.collided_object = obj
                    break
        avatar.move(-dx - offset, 0)
        avatar.update_mask()
        return self.collided_object

    # function for vertical collision
    def collide_vertical(self, avatar, objects, dy, offset=0):
        avatar.move(0, dy + offset)
        avatar.update_mask()
        self.collided_object = None
        for obj in objects:
            if hasattr(obj, 'mask') and obj.mask is not None:
                if pygame.sprite.collide_mask(avatar, obj):
                    self.collided_object = obj
                    break
        avatar.move(0, -dy - offset)
        avatar.update_mask()
        return self.collided_object

    # initialise the pygame environment
    def _pygame_initialise(self):
        pygame.display.set_caption('BoxHead')
        pygame.init()
        pygame.display.init()
        self.clock = pygame.time.Clock()
        self.window = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self.font = pygame.font.Font('Fonts/arial.ttf', 20)

        # initialise background
        self.bg_tiles, self.bg_image = self.get_background('images/Terrain/chosen_terrain.png')

    def reset(self):
        pass

    # player action
    def perform_action(self, char_action: CharAction) -> bool:
        if char_action == CharAction.NORTH:
            self.player.move_up(self.PLAYER_SPEED)
        elif char_action == CharAction.EAST:
            self.player.move_right(self.PLAYER_SPEED)
        elif char_action == CharAction.SOUTH:
            self.player.move_down(self.PLAYER_SPEED)
        elif char_action == CharAction.WEST:
            self.player.move_left(self.PLAYER_SPEED)
        elif char_action == CharAction.SHOOT:
            self.shoot_bullet()
            print("Player is shooting!")
        return True

    def shoot_bullet(self):
        current_time = pygame.time.get_ticks()
        if current_time - self.last_bullet_fired >= 500:  # 200 milliseconds = 0.2 seconds
            bullet_x = self.player.rect.x + self.player.rect.width // 2 - 5
            bullet_y = self.player.rect.y + self.player.rect.height // 2 - 5
            bullet = Bullet(bullet_x, bullet_y, 10, 5, self.player.direction)
            self.bullets.append(bullet)
            self.last_bullet_fired = current_time

    def step(self):
        pass

    def render(self):

        # draw background tiles
        self._draw_background(self.bg_image, self.bg_tiles)

        # draw wall tiles
        self.wall_width, self.wall_height = 64, 16
        self.range_width = (self.WIDTH // self.wall_width + 1) // 2
        self.range_height = (self.HEIGHT // self.wall_height + 1) // 3
        self.all_walls = self._draw_walls(self.window, self.WIDTH, self.HEIGHT, self.wall_width, self.wall_height, self.range_width, self.range_height)

        # render and draw the player
        self.player.loop(self.FPS)
        self.player.draw(self.window)

        # render and draw the monster
        self.monster.draw(self.window)
        
        # if player and monster collide
        if pygame.sprite.collide_mask(self.player, self.monster):
            self.game_over = True

        # Update and draw bullets
        for bullet in self.bullets:
            bullet.update()
            bullet.draw(self.window)

            if bullet.has_collided_with_monster(self.monster):
                self.points += 1
                # Remove the current monster and spawn a new one at a random position
                self.monster = self.spawn_monster()
                self.bullets.remove(bullet)

        self.points_text(self.window)

        self._emergency_event()

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

        self._player_movement(self.all_walls, self.PLAYER_SPEED)

        self._monster_movement(self.all_walls, self.MONSTER_SPEED)
        
    # function for player movement
    def _player_movement(self, objects, player_speed):

        self.player.x_vel = 0
        self.player.y_vel = 0

        self.collide_left = self.collide_horizontal(self.player, objects, -player_speed)
        self.collide_right = self.collide_horizontal(self.player, objects, player_speed)
        self.collide_up = self.collide_vertical(self.player, objects, -player_speed)
        self.collide_down = self.collide_vertical(self.player, objects, player_speed)

        key_pressed = pygame.key.get_pressed()

        if key_pressed[pygame.K_w] and not self.collide_up:
            self.perform_action(CharAction.NORTH)
        elif key_pressed[pygame.K_d] and not self.collide_right:
            self.perform_action(CharAction.EAST)
        elif key_pressed[pygame.K_s] and not self.collide_down:
            self.perform_action(CharAction.SOUTH)
        elif key_pressed[pygame.K_a] and not self.collide_left:
            self.perform_action(CharAction.WEST)
        elif key_pressed[pygame.K_SPACE]:
            self.perform_action(CharAction.SHOOT)

    # function for monster movement (heading towards the player)
    def _monster_movement(self, objects, monster_speed):

        self.monster.x_vel = 0
        self.monster.y_vel = 0

        # Calculate the direction vector towards the player
        player_x_pos, player_y_pos = self.player.get_position()
        monster_x_pos, monster_y_pos = self.monster.get_position()

        if monster_y_pos < 50:
            self.monster.y_vel = monster_speed
        else:
            direction_x = player_x_pos - monster_x_pos
            direction_y = player_y_pos - monster_y_pos

            # Normalize the direction vector
            distance = math.sqrt(direction_x ** 2 + direction_y ** 2)
            if distance != 0:
                direction_x /= distance
                direction_y /= distance

            # Move the monster towards the player
            self.monster.x_vel = direction_x * monster_speed
            self.monster.y_vel = direction_y * monster_speed

            # Check for collisions and adjust movement accordingly
            collide_left = self.collide_horizontal(self.monster, objects, -monster_speed)
            collide_right = self.collide_horizontal(self.monster, objects, monster_speed)
            collide_up = self.collide_vertical(self.monster, objects, -monster_speed)
            collide_down = self.collide_vertical(self.monster, objects, monster_speed)

            if collide_left is not None:
                self.monster.x_vel = 0
            if collide_right is not None:
                self.monster.x_vel = 0
            if collide_up is not None:
                self.monster.y_vel = 0
            if collide_down is not None:
                self.monster.y_vel = 0

        # Move the monster based on the adjusted velocities
        self.monster.move(self.monster.x_vel, self.monster.y_vel)
        self.monster.update_sprite()


    def step(self):
        pass

    # helper function for drawing on the pygame window
    def _draw_background(self, bg_img, bg_tiles):
        for tile in bg_tiles:
            self.window.blit(bg_img, tile)

    # drawing walls around the map
    def _draw_walls(self, window, win_width, win_height, wall_width, wall_height, range_width, range_height):
        wall = [Wall(i * wall_width, 0,
                wall_width, wall_height) for i in range(0, range_width - 3)]

        wall2 = [Wall(win_width - i * wall_width, 0,
                 wall_width, wall_height) for i in range(range_width - 3, 0, -1)]
        
        wall3 = [Wall(i * wall_width, win_height - wall_height,
                 wall_width, wall_height) for i in range(0, range_width - 3)]
        
        wall4 = [Wall(win_width - i * wall_width, win_height - wall_height,
                 wall_width, wall_height) for i in range(range_width - 3, 0, -1)]
        
        wall5 = [Wall(0, i * wall_height, wall_width, wall_height, 90) for i in range(0, range_height - 3)]

        wall6 = [Wall(0, win_height - i * wall_height, wall_width, wall_height, 90) for i in range(range_height - 2, 0, -1)]

        wall7 = [Wall(win_width - wall_height, i * wall_height,
                      wall_width, wall_height, -90) for i in range(0, range_height - 3)]
        
        wall8 = [Wall(win_width - wall_height, win_height - i * wall_height,
                      wall_width, wall_height, -90) for i in range(range_height - 3, 0, -1)]
        
        all_walls = wall + wall2 + wall3 + wall4 + wall5 + wall6 + wall7 + wall8

        for w in all_walls:
            w.draw(window)
        
        return all_walls
    
    def spawn_monster(self):
        # Spawn a monster at a random location and add it to the monster group.
        area_select = random.choice([i for i in range(1, 5)])
        if area_select == 1:
            monster = Monster(random.randint(75, self.WIDTH - 75), 100, 16, 16)
            return monster
        elif area_select == 2:
            monster = Monster(75, random.randint(75, self.HEIGHT - 100), 16, 16)
            return monster
        elif area_select == 3:
            monster = Monster(self.WIDTH - 75, random.randint(75, self.HEIGHT - 100), 16, 16)
            return monster
        elif area_select == 4:
            monster = Monster(random.randint(75, self.WIDTH - 75), self.HEIGHT - 100, 16, 16)
            return monster

if __name__ == "__main__":
    boxHead = BoxHead(60)
    
    while True:
        boxHead.step()
        boxHead.render()