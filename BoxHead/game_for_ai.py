import os
import random
import math
import numpy as np
import pygame
import time
from os import listdir
from os.path import isfile, join
from enum import Enum
from collections import namedtuple

# import gymnasium as gym
# from gymnasium import spaces

pygame.init()
font = pygame.font.Font('Fonts/arial.ttf', 20)


def load_player_sprite_sheets(width, height, direction=False):
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


def load_object(dir1, dir2, file, width, height, scaling=1):
    path = join(dir1, dir2, file)
    image = pygame.image.load(path).convert_alpha()
    surface = pygame.Surface((width, height), pygame.SRCALPHA, 32)
    rect = pygame.Rect(0, 0, width, height)
    surface.blit(image, (0, 0), rect)
    return pygame.transform.scale_by(surface, scaling)


class Object(pygame.sprite.Sprite):
    def __init__(self, x, y, width, height, name=None):
        super().__init__()
        self.rect = pygame.Rect(x, y, width, height)
        self.image = pygame.Surface((width, height), pygame.SRCALPHA, 32)
        self.width = width
        self.height = height
        self.name = name

    def draw(self, game_window):
        game_window.blit(self.image, (self.rect.x, self.rect.y))


class Barrel(Object):
    def __init__(self, x, y, width, height):
        super().__init__(x, y, width, height)
        barrel = load_object('images', 'Barrel', 'chosen_barrels.png', width, height, 1)
        self.image.blit(barrel, (0, 0))
        self.mask = pygame.mask.from_surface(self.image)


class Block(Object):
    def __init__(self, x, y, size):
        super().__init__(x, y, size, size)
        block = load_object('images', 'Blocks', 'smooth_blocks.png', size, size)
        self.image.blit(block, (0, 0))
        self.mask = pygame.mask.from_surface(self.image)


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


def get_background(file):
    image = pygame.image.load(file)
    _, _, width, height = image.get_rect()
    tiles = []

    for i in range(Env.WIDTH // width + 1):
        for j in range(Env.HEIGHT // height + 1):
            tile_pos = (i * width, j * height)
            tiles.append(tile_pos)

    return tiles, image


def draw_health_bar(game_window, player):
    # draws a health bar on the screen
    max_health_width = 180  # 200 pixels wide when full
    current_health_width = (player.health / 90) * max_health_width
    health_bar_rect = pygame.Rect(10, 10, current_health_width, 20)
    pygame.draw.rect(game_window, (255, 0, 0), health_bar_rect)  # Red health bar


def gun_choice(player):
    key_pressed = pygame.key.get_pressed()

    # weapon selection
    if key_pressed[pygame.K_0]:  # pistol
        player.gun = Gun.PISTOL
        player.current_ammo = Gun.get_current_ammo(Gun.PISTOL)
    if key_pressed[pygame.K_1]:  # uzi
        player.gun = Gun.UZI
        player.current_ammo = Gun.get_current_ammo(Gun.UZI)


# functions that create portions of the game
GunProperties = namedtuple('GunProperties',
                           ['damage', 'fire_rate', 'ammo_capacity'])


class Gun(Enum):
    # name, damage, fire rate, ammo cap

    PISTOL = GunProperties(15, 0.4, 99999)
    UZI = GunProperties(20, 0.1, 100)

    def __init__(self, damage, fire_rate, ammo_capacity):
        self.damage = damage
        self.fire_rate = fire_rate  # Shots per second
        self.ammo_capacity = ammo_capacity
        self.current_ammo = self.ammo_capacity

    def get_current_ammo(self):
        return self.current_ammo


def bullet_to_monster(env, player, monster):
    # global ammo_counts

    if env.player.fire_gun == 1:
        current_time = pygame.time.get_ticks()
        if player.current_ammo <= 0:
            return

        # is gun ready to fire?
        gun_ready = current_time - player.last_fired >= 1000 * player.gun.value.fire_rate and 1 <= player.current_ammo <= 1000
        gun_ready_inf = current_time - player.last_fired >= 1000 * player.gun.value.fire_rate and player.current_ammo >= 1000

        if gun_ready:

            player.current_ammo -= 1
            player.last_fired = current_time

            player_up_x_min, player_up_x_max = player.rect.x + 15, player.rect.x + 50
            player_down_x_min, player_down_x_max = player.rect.x, player.rect.x + 15
            player_y_min, player_y_max = player.rect.y - 5, player.rect.y + 20

            if player.direction == 'left':
                if player_y_min <= monster.rect.y + 8 <= player_y_max and monster.rect.x < player.rect.x:
                    monster.health -= player.gun.value.damage
                    reward = env.REWARD
            if player.direction == 'right':
                if player_y_min <= monster.rect.y + 8 <= player_y_max and monster.rect.x > player.rect.x:
                    monster.health -= player.gun.value.damage
                    reward = env.REWARD
            if player.direction == 'up':
                if player_up_x_min <= monster.rect.x + 8 <= player_up_x_max and monster.rect.y < player.rect.y:
                    monster.health -= player.gun.value.damage
                    reward = env.REWARD
            if player.direction == 'down':
                if player_down_x_min <= monster.rect.x + 8 <= player_down_x_max and monster.rect.y > player.rect.y:
                    monster.health -= player.gun.value.damage
                    reward = env.REWARD

        elif gun_ready_inf:
            player.last_fired = current_time

            player_up_x_min, player_up_x_max = player.rect.x + 15, player.rect.x + 50
            player_down_x_min, player_down_x_max = player.rect.x, player.rect.x + 15
            player_y_min, player_y_max = player.rect.y - 5, player.rect.y + 20

            if player.direction == 'left':
                if player_y_min <= monster.rect.y + 8 <= player_y_max and monster.rect.x < player.rect.x:
                    monster.health -= player.gun.value.damage
            if player.direction == 'right':
                if player_y_min <= monster.rect.y + 8 <= player_y_max and monster.rect.x > player.rect.x:
                    monster.health -= player.gun.value.damage
            if player.direction == 'up':
                if player_up_x_min <= monster.rect.x + 8 <= player_up_x_max and monster.rect.y < player.rect.y:
                    monster.health -= player.gun.value.damage
            if player.direction == 'down':
                if player_down_x_min <= monster.rect.x + 8 <= player_down_x_max and monster.rect.y > player.rect.y:
                    monster.health -= player.gun.value.damage


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
        self.gun = Gun.PISTOL
        self.current_ammo = self.gun.value.ammo_capacity
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

        self.SPRITES = load_player_sprite_sheets(30, 34, True)

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
        self.health = 60
        self.damage = 15
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


def spawn_monster():

    # Spawn a monster at a random location and add it to the monster group.
    area_select = random.choice([i for i in range(1, 5)])
    if area_select == 1:
        monster = Monster(random.randint(75, Env.WIDTH - 75), 100, 16, 16)
        return monster
    elif area_select == 2:
        monster = Monster(75, random.randint(75, Env.HEIGHT - 100), 16, 16)
        return monster
    elif area_select == 3:
        monster = Monster(Env.WIDTH - 75, random.randint(75, Env.HEIGHT - 100), 16, 16)
        return monster
    elif area_select == 4:
        monster = Monster(random.randint(75, Env.WIDTH - 75), Env.HEIGHT - 100, 16, 16)
        return monster


def draw(game_window, background, image, player, barrels, blocks, wall, monster_group):
    for tile in background:
        game_window.blit(image, tile)

    for obj in barrels:
        obj.draw(game_window)

    for obj in blocks:
        obj.draw(game_window)

    for obj in wall:
        obj.draw(game_window)

    for monster in monster_group:
        monster.draw(game_window)

    # Draw the point table
    points_text = font.render(f"Points: {Env.points}", True, (0, 0, 0, 150))
    game_window.blit(points_text, (Env.WIDTH - points_text.get_width() - 70, 15))
    if player.current_ammo >= 1000:
        ammo_text = font.render(f"Ammo: Infinite", True, (0, 0, 0, 150))
        game_window.blit(ammo_text, (Env.WIDTH - points_text.get_width() - 70, 35))
    elif player.current_ammo < 1000:
        ammo_text = font.render(f"Ammo: {player.current_ammo}", True, (0, 0, 0, 150))
        game_window.blit(ammo_text, (Env.WIDTH - points_text.get_width() - 70, 35))

    draw_health_bar(game_window, player)
    player.draw(game_window)
    pygame.display.update()


def collide_vertical(player, objects, dy, offset=0):
    player.move(0, dy + offset)
    player.update_mask()
    collided_object = None
    for obj in objects:
        if hasattr(obj, 'mask') and obj.mask is not None:  # Check if obj has a mask
            if pygame.sprite.collide_mask(player, obj):
                collided_object = obj
                break
    player.move(0, -dy - offset)
    player.update_mask()
    return collided_object


def collide_horizontal(player, objects, dx, offset=0):
    player.move(dx + offset, 0)
    player.update_mask()
    collided_object = None
    for obj in objects:
        if hasattr(obj, 'mask') and obj.mask is not None:  # Check if obj has a mask
            if pygame.sprite.collide_mask(player, obj):
                collided_object = obj
                break
    player.move(-dx - offset, 0)
    player.update_mask()
    return collided_object


def check_collision(player, monsters):
    game_over = False
    # Check for collision between the player and any monsters.
    for monster in monsters:
        collided_object = None
        if hasattr(monster, 'mask') and monster.mask is not None:  # Check if obj has a mask
            if pygame.sprite.collide_mask(player, monster):
                game_over = True
                break
    return game_over


def player_movement(player, objects):
    key_pressed = pygame.key.get_pressed()

    player.x_vel = 0
    player.y_vel = 0
    collide_left = collide_horizontal(player, objects, -player.speed)
    collide_right = collide_horizontal(player, objects, player.speed)
    collide_up = collide_vertical(player, objects, -player.speed)
    collide_down = collide_vertical(player, objects, player.speed)

    # movement
    if key_pressed[pygame.K_LEFT] and not collide_left:  # x -= 1
        player.move_left(player.speed)
    if key_pressed[pygame.K_RIGHT] and not collide_right:  # x += 1
        player.move_right(player.speed)
    if key_pressed[pygame.K_UP] and not collide_up:  # y -= 1
        player.move_up(player.speed)
    if key_pressed[pygame.K_DOWN] and not collide_down:  # y += 1
        player.move_down(player.speed)


def monster_movement(monster, player, objects):

    # Calculate the current time
    current_time = pygame.time.get_ticks() // 100

    # Check if the monster's position has changed
    if monster.x_vel == 0 or monster.y_vel == 0:
        # If the monster has been stationary and print the statement only if it is not moving
        if current_time - Env.last_stationary_time >= 100:
            # print("Monster is stationary for too long. Game over!")
            Env.game_over = True
            return Env.game_over  # Indicate game over condition
    else:
        Env.last_stationary_time = current_time  # Update last stationary time if monster is moving
    # print(current_time - Env.last_stationary_time)
    # print(current_time)

    monster.x_vel = 0
    monster.y_vel = 0

    # Calculate the direction vector towards the player
    player_x_pos, player_y_pos = player.get_position()
    monster_x_pos, monster_y_pos = monster.get_position()

    if monster_y_pos < 50:
        monster.y_vel = monster.speed
    else:
        direction_x = player_x_pos - monster_x_pos
        direction_y = player_y_pos - monster_y_pos

        # Normalize the direction vector
        distance = math.sqrt(direction_x ** 2 + direction_y ** 2)
        if distance != 0:
            direction_x /= distance
            direction_y /= distance

        # Move the monster towards the player
        monster.x_vel = direction_x * Env.MONSTER_SPEED
        monster.y_vel = direction_y * Env.MONSTER_SPEED

        # Check for collisions and adjust movement accordingly
        collide_left = collide_horizontal(monster, objects, -Env.MONSTER_SPEED)
        collide_right = collide_horizontal(monster, objects, Env.MONSTER_SPEED)
        collide_up = collide_vertical(monster, objects, -Env.MONSTER_SPEED)
        collide_down = collide_vertical(monster, objects, Env.MONSTER_SPEED)

        if collide_left is not None:
            monster.x_vel = 0
        if collide_right is not None:
            monster.x_vel = 0
        if collide_up is not None:
            monster.y_vel = 0
        if collide_down is not None:
            monster.y_vel = 0

    # Move the monster based on the adjusted velocities
    monster.move(monster.x_vel, monster.y_vel)
    monster.update_sprite()

    return False


class Env:
    # class for Environment
    # set base values
    BG_COLOR = (255, 255, 255, 150)
    WIDTH, HEIGHT = 1000, 700
    FPS = 60
    PLAYER_SPEED = 3
    MONSTER_SPEED = 1

    # Environment specifics
    ENV_SHAPE = (WIDTH, HEIGHT, 1)
    ACTION_SPACE = [1, 2, 3, 4, 5]
    ACTION_SPACE_SIZE = len(ACTION_SPACE)
    ACTION_SHAPE = (ACTION_SPACE_SIZE,)
    PUNISHMENT = -1
    REWARD = 1
    SCORE = 0
    MAX_VAL = 2
    TARGET_POINTS_WIN = 5
    TARGET_POINTS_LOSS = 0
    # point system
    points = 0

    # time when monster hit player
    last_hit_time = 0
    # stationary period for monster
    last_stationary_time = 0

    def __init__(self):
        pygame.init()
        self.window = pygame.display.set_mode((Env.WIDTH, Env.HEIGHT))
        self.reset()

    def get_state(self, player, monster_group):
        state = self.get_state_as_array(player, monster_group)
        return state

    def get_state_as_array(self, player, monster_group):
        # Get the game field dimensions
        width, height, _ = self.ENV_SHAPE

        # Create a 2D numpy array with the appropriate values for empty spaces, monsters, and the player
        state_array = np.zeros((width, height))

        # Set the player position
        state_array[player.rect.x // 50, player.rect.y // 50] = 1

        # Set the monster positions
        for monster in monster_group:
            state_array[monster.rect.x // 50, monster.rect.y // 50] = 2

        # Normalize the 2D numpy array by dividing each cell by 2
        state_array = state_array / 2

        return state_array


    def reset(self):
        self.points = 0
        self.game_over = False
        self.game_won = False

        # render player
        self.player = Player(Env.WIDTH / 2 - 25, Env.HEIGHT / 2 - 25, 50, 50)

        # render monster
        self.monster_group = pygame.sprite.Group()
        self.monster_group.add(spawn_monster())

        return self.get_state(self.player, self.monster_group), self.player, self.monster_group

    def step(self, player, monster_group, action=None):
        # call every frame
        bg_tiles, bg_image = get_background('images/Terrain/chosen_terrain.png')
        barrel_width, barrel_height = 46, 60
        block_size = 32
        wall_width, wall_height = 64, 16

        # barrel + block position
        barrel_pos_x, barrel_pos_y = [325, 650], [350]
        block_pos_x, block_pos_y = [225, 750], [275, 450]

        # place barrels + blocks
        barrel_group = pygame.sprite.Group()
        barrels = [Barrel(barrel_pos_x[i], barrel_pos_y[j],
                          barrel_width, barrel_height) for i in range(2) for j in range(1)]
        barrel_group.add(barrels)
        block_group = pygame.sprite.Group()
        blocks = [Block(block_pos_x[1], block_pos_y[0], block_size),
                  Block(block_pos_x[1], block_pos_y[1], block_size)]
        block_group.add(blocks)

        # create walls
        range_end = (Env.WIDTH // wall_width + 1) // 2
        wall = [Wall(i * wall_width, 0,
                     wall_width, wall_height) for i in range(0, range_end - 3)]
        wall2 = [Wall(Env.WIDTH - i * wall_width, 0,
                      wall_width, wall_height) for i in range(range_end - 3, 0, -1)]
        wall3 = [Wall(i * wall_width, Env.HEIGHT - wall_height,
                      wall_width, wall_height) for i in range(0, Env.WIDTH // wall_width + 1)]
        wall4 = [Wall(0, i * wall_height,
                      wall_width, wall_height, 90) for i in range(0, Env.HEIGHT // wall_height + 1)]
        wall5 = [Wall(Env.WIDTH - wall_height, i * wall_height,
                      wall_width, wall_height, -90) for i in range(0, Env.HEIGHT // wall_height + 1)]

        all_walls = wall + wall2 + wall3 + wall4 + wall5

        if action is not None:
            player.action(action)
        player.auto_move()

        reward = 0

        gun_choice(player)

        some_collision_objects = list(barrel_group.sprites()) + all_walls
        some_collision_objects.append(player)
        all_collision_objects = list(barrel_group.sprites()) + all_walls + list(monster_group.sprites())
        player_movement(player, all_collision_objects)

        for monster in monster_group:
            monster_movement(monster, player, some_collision_objects)

            if check_collision(player, list(monster_group.sprites())):
                self.update_points(self.PUNISHMENT)
                reward = self.PUNISHMENT
                monster.kill()
                monster_group.add(spawn_monster())

            if self.player.fire_gun == 1:
                current_time = pygame.time.get_ticks()
                if player.current_ammo <= 0:
                    return

                # is gun ready to fire?
                gun_ready = current_time - player.last_fired >= 1000 * player.gun.value.fire_rate and 1 <= player.current_ammo <= 1000
                gun_ready_inf = current_time - player.last_fired >= 1000 * player.gun.value.fire_rate and player.current_ammo >= 1000

                if gun_ready:

                    player.current_ammo -= 1
                    player.last_fired = current_time

                    player_up_x_min, player_up_x_max = player.rect.x + 15, player.rect.x + 50
                    player_down_x_min, player_down_x_max = player.rect.x, player.rect.x + 15
                    player_y_min, player_y_max = player.rect.y - 5, player.rect.y + 20

                    if player.direction == 'left':
                        if player_y_min <= monster.rect.y + 8 <= player_y_max and monster.rect.x < player.rect.x:
                            monster.health -= player.gun.value.damage
                            self.update_points(self.REWARD)
                            reward = self.REWARD
                    if player.direction == 'right':
                        if player_y_min <= monster.rect.y + 8 <= player_y_max and monster.rect.x > player.rect.x:
                            monster.health -= player.gun.value.damage
                            self.update_points(self.REWARD)
                            reward = self.REWARD
                    if player.direction == 'up':
                        if player_up_x_min <= monster.rect.x + 8 <= player_up_x_max and monster.rect.y < player.rect.y:
                            monster.health -= player.gun.value.damage
                            self.update_points(self.REWARD)
                            reward = self.REWARD
                    if player.direction == 'down':
                        if player_down_x_min <= monster.rect.x + 8 <= player_down_x_max and monster.rect.y > player.rect.y:
                            monster.health -= player.gun.value.damage
                            self.update_points(self.REWARD)
                            reward = self.REWARD

                elif gun_ready_inf:
                    player.last_fired = current_time

                    player_up_x_min, player_up_x_max = player.rect.x + 15, player.rect.x + 50
                    player_down_x_min, player_down_x_max = player.rect.x, player.rect.x + 15
                    player_y_min, player_y_max = player.rect.y - 5, player.rect.y + 20

                    if player.direction == 'left':
                        if player_y_min <= monster.rect.y + 8 <= player_y_max and monster.rect.x < player.rect.x:
                            monster.health -= player.gun.value.damage
                            reward = self.REWARD
                    if player.direction == 'right':
                        if player_y_min <= monster.rect.y + 8 <= player_y_max and monster.rect.x > player.rect.x:
                            monster.health -= player.gun.value.damage
                            reward = self.REWARD
                    if player.direction == 'up':
                        if player_up_x_min <= monster.rect.x + 8 <= player_up_x_max and monster.rect.y < player.rect.y:
                            monster.health -= player.gun.value.damage
                            reward = self.REWARD
                    if player.direction == 'down':
                        if player_down_x_min <= monster.rect.x + 8 <= player_down_x_max and monster.rect.y > player.rect.y:
                            monster.health -= player.gun.value.damage
                            reward = self.REWARD

            if monster.health <= 0:
                monster.kill()
                Env.points += 1
                monster_group.add(spawn_monster())

            if self.points <= self.TARGET_POINTS_LOSS:
                self.game_over = True

            if self.points >= self.TARGET_POINTS_WIN:
                self.game_won = True

        return self.get_state(player, monster_group), reward, self.game_over or self.game_won, self.points, bg_tiles, bg_image, barrel_group, block_group, all_walls

    def update_points(self, delta):
        self.points += delta

    def render(self, window, bg_tiles, bg_image,
               player, barrel_group, block_group, all_walls, monster_group):

        # render player
        player.loop(Env.FPS)

        # draw everything in the game
        draw(window, bg_tiles, bg_image,
             player, barrel_group, block_group, all_walls, monster_group)


def main():
    pygame.init()
    clock = pygame.time.Clock()
    window = pygame.display.set_mode((1000, 700))
    env = Env()
    run = True

    obs, player, monster_group = env.reset()

    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        keys = pygame.key.get_pressed()

        if keys[pygame.K_LEFT]:
            env.player.dir_x = -1
        if keys[pygame.K_RIGHT]:
            env.player.dir_x = 1
        if keys[pygame.K_UP]:
            env.player.dir_y = -1
        if keys[pygame.K_DOWN]:
            env.player.dir_y = 1
        if keys[pygame.K_SPACE]:
            env.player.fire_gun = 1
        _, _, _, _, bg_tiles, bg_image, barrel_group, block_group, all_walls = env.step(player, monster_group)

        env.render(window, bg_tiles, bg_image,
                   player, barrel_group, block_group, all_walls, monster_group)

        pygame.display.flip()
        clock.tick(env.FPS)

    pygame.quit()

    return 0


if __name__ == "__main__":
    main()


