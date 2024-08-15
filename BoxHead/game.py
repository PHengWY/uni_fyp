import os
import random
import math
import pygame
import time
from os import listdir
from os.path import isfile, join
from enum import Enum
from collections import namedtuple

# import gymnasium as gym
# from gymnasium import spaces

pygame.display.set_caption('BoxHead')

# set base values
BG_COLOR = (255, 255, 255, 150)
WIDTH, HEIGHT = 1000, 700
FPS = 60
PLAYER_SPEED = 3
MONSTER_SPEED = 1

# point system
points = 0

# monster hit player
last_hit_time = 0
# stationary period for monster
last_stationary_time = time.time()

# initialise game
pygame.init()
# initialise clock
clock = pygame.time.Clock()
# initialise window
window = pygame.display.set_mode((WIDTH, HEIGHT))

font = pygame.font.Font(None, 36)


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


def draw_health_bar(game_window, player):
    """Draws a health bar on the screen."""
    max_health_width = 180  # 200 pixels wide when full
    current_health_width = (player.health / 90) * max_health_width
    health_bar_rect = pygame.Rect(10, 10, current_health_width, 20)
    pygame.draw.rect(game_window, (255, 0, 0), health_bar_rect)  # Red health bar


class Player(pygame.sprite.Sprite):
    SPRITES = load_player_sprite_sheets(30, 34, True)
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

    def move(self, dx, dy):
        self.rect.x += dx
        self.rect.y += dy

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


def gun_choice(player):
    key_pressed = pygame.key.get_pressed()

    # weapon selection
    if key_pressed[pygame.K_0]:  # pistol
        player.gun = Gun.PISTOL
        player.current_ammo = Gun.get_current_ammo(Gun.PISTOL)
    if key_pressed[pygame.K_1]:  # uzi
        player.gun = Gun.UZI
        player.current_ammo = Gun.get_current_ammo(Gun.UZI)


GunProperties = namedtuple('GunProperties', ['damage', 'fire_rate', 'ammo_capacity'])


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


def bullet_to_monster(player, monster):
    # global ammo_counts

    key_pressed = pygame.key.get_pressed()
    current_time = pygame.time.get_ticks()

    if player.current_ammo <= 0:
        return

    # is gun ready to fire?
    gun_ready = current_time - player.last_fired >= 1000 * player.gun.value.fire_rate and 1 <= player.current_ammo <= 1000
    gun_ready_inf = current_time - player.last_fired >= 1000 * player.gun.value.fire_rate and player.current_ammo >= 1000

    if key_pressed[pygame.K_SPACE] and gun_ready:

        player.current_ammo -= 1
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

    elif key_pressed[pygame.K_SPACE] and gun_ready_inf:
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


def spawn_monster():
    # Spawn a monster at a random location and add it to the monster group.
    area_select = random.choice([i for i in range(1, 5)])
    if area_select == 1:
        monster = Monster(random.randint(75, WIDTH - 75), 100, 16, 16)
        return monster
    elif area_select == 2:
        monster = Monster(75, random.randint(75, HEIGHT - 100), 16, 16)
        return monster
    elif area_select == 3:
        monster = Monster(WIDTH - 75, random.randint(75, HEIGHT - 100), 16, 16)
        return monster
    elif area_select == 4:
        monster = Monster(random.randint(75, WIDTH - 75), HEIGHT - 100, 16, 16)
        return monster


class Monster(pygame.sprite.Sprite):
    # Load monster sprite
    monster_sprite = load_object('images', 'Monster', 'monster_a.png', 16, 16, 2)

    def __init__(self, x, y, width, height):
        super().__init__()
        self.rect = pygame.Rect(x, y, width, height)
        self.x_vel = 0
        self.y_vel = 0
        self.mask = None
        self.sprite = self.monster_sprite
        self.direction = 'down'
        self.health = 60
        self.damage = 15

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

    for i in range(WIDTH // width + 1):
        for j in range(HEIGHT // height + 1):
            tile_pos = (i * width, j * height)
            tiles.append(tile_pos)

    return tiles, image


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
    points_text = font.render(f"Points: {points}", True, (0, 0, 0, 150))
    game_window.blit(points_text, (WIDTH - points_text.get_width() - 70, 15))
    if player.current_ammo >= 1000:
        ammo_text = font.render(f"Ammo: Infinite", True, (0, 0, 0, 150))
        game_window.blit(ammo_text, (WIDTH - points_text.get_width() - 70, 35))
    elif player.current_ammo < 1000:
        ammo_text = font.render(f"Ammo: {player.current_ammo}", True, (0, 0, 0, 150))
        game_window.blit(ammo_text, (WIDTH - points_text.get_width() - 70, 35))

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


def check_collision(player, monsters, current_time):
    global last_hit_time
    game_over = False
    # Check for collision between the player and any monsters.
    for monster in monsters:
        collided_object = None
        if hasattr(monster, 'mask') and monster.mask is not None:  # Check if obj has a mask
            if pygame.sprite.collide_mask(player, monster):
                if current_time - last_hit_time >= 1:
                    player.health -= 15  # Reduce player health by 15
                    last_hit_time = current_time
                    if player.health <= 0:
                        game_over = True
                        print("Player is dead!")  # Handle game over condition
                        return game_over


def player_movement(player, objects):
    key_pressed = pygame.key.get_pressed()

    player.x_vel = 0
    player.y_vel = 0
    collide_left = collide_horizontal(player, objects, -PLAYER_SPEED)
    collide_right = collide_horizontal(player, objects, PLAYER_SPEED)
    collide_up = collide_vertical(player, objects, -PLAYER_SPEED)
    collide_down = collide_vertical(player, objects, PLAYER_SPEED)

    # movement
    if key_pressed[pygame.K_LEFT] and not collide_left:  # x -= 1
        player.move_left(PLAYER_SPEED)
    if key_pressed[pygame.K_RIGHT] and not collide_right:  # x += 1
        player.move_right(PLAYER_SPEED)
    if key_pressed[pygame.K_UP] and not collide_up:  # y -= 1
        player.move_up(PLAYER_SPEED)
    if key_pressed[pygame.K_DOWN] and not collide_down:  # y += 1
        player.move_down(PLAYER_SPEED)


def monster_movement(monster, player, objects):
    global last_stationary_time

    # Calculate the current time
    current_time = time.time()

    # Check if the monster's position has changed
    if monster.x_vel != 0 or monster.y_vel != 0:
        last_stationary_time = current_time  # Update last stationary time if monster is moving
    else:
        # If the monster has been stationary for too long, end the game
        if current_time - last_stationary_time >= 8:
            print("Monster is stationary for too long. Game over!")
            return True  # Indicate game over condition

    # Update the last_stationary_time if the monster is moving

    monster.x_vel = 0
    monster.y_vel = 0

    # Calculate the direction vector towards the player
    player_x_pos, player_y_pos = player.get_position()
    monster_x_pos, monster_y_pos = monster.get_position()

    if monster_y_pos < 50:
        monster.y_vel = MONSTER_SPEED
    else:
        direction_x = player_x_pos - monster_x_pos
        direction_y = player_y_pos - monster_y_pos

        # Normalize the direction vector
        distance = math.sqrt(direction_x ** 2 + direction_y ** 2)
        if distance != 0:
            direction_x /= distance
            direction_y /= distance

        # Move the monster towards the player
        monster.x_vel = direction_x * MONSTER_SPEED
        monster.y_vel = direction_y * MONSTER_SPEED

        # Check for collisions and adjust movement accordingly
        collide_left = collide_horizontal(monster, objects, -MONSTER_SPEED)
        collide_right = collide_horizontal(monster, objects, MONSTER_SPEED)
        collide_up = collide_vertical(monster, objects, -MONSTER_SPEED)
        collide_down = collide_vertical(monster, objects, MONSTER_SPEED)

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


def main(game_window):
    # clock
    clock.tick(FPS)
    bg_tiles, bg_image = get_background('images/Terrain/chosen_terrain.png')

    barrel_width, barrel_height = 46, 60
    block_size = 32
    wall_width, wall_height = 64, 16

    # player details
    player = Player(window.get_width() / 2, window.get_height() / 2, 50, 50)

    barrel_pos_x, barrel_pos_y = [325, 650], [350]
    block_pos_x, block_pos_y = [225, 750], [275, 450]

    # create explosive barrels on screen
    barrel_group = pygame.sprite.Group()
    barrels = [Barrel(barrel_pos_x[i], barrel_pos_y[j],
                      barrel_width, barrel_height) for i in range(2) for j in range(1)]
    barrel_group.add(barrels)

    dist_barrels_horizontal = barrel_pos_x[1] - (barrel_pos_x[0] + barrel_width)
    blocks_mid_x = (dist_barrels_horizontal / 2) - (block_size / 2)

    # create blocks to refill ammunition
    block_group = pygame.sprite.Group()
    blocks = [Block(block_pos_x[0], block_pos_y[0], block_size),
              Block(barrel_pos_x[0] + blocks_mid_x + barrel_width, block_pos_y[0], block_size),
              Block(block_pos_x[1], block_pos_y[0], block_size),
              Block(block_pos_x[0], block_pos_y[1], block_size),
              Block(barrel_pos_x[0] + blocks_mid_x + barrel_width, block_pos_y[1], block_size),
              Block(block_pos_x[1], block_pos_y[1], block_size)]
    block_group.add(blocks)

    # create a wall around the game area, seperated into 5 segments
    # wall 1 = top left stretch, wall 2 = top right stretch
    # wall 3 = bottom stretch
    # wall 4 = left stretch, wall 5 = right stretch
    range_end = (WIDTH // wall_width + 1) // 2
    wall = [Wall(i * wall_width, 0,
                 wall_width, wall_height) for i in range(0, range_end - 3)]
    wall2 = [Wall(WIDTH - i * wall_width, 0,
                  wall_width, wall_height) for i in range(range_end - 3, 0, -1)]
    wall3 = [Wall(i * wall_width, HEIGHT - wall_height,
                  wall_width, wall_height) for i in range(0, WIDTH // wall_width + 1)]
    wall4 = [Wall(0, i * wall_height,
                  wall_width, wall_height, 90) for i in range(0, HEIGHT // wall_height + 1)]
    wall5 = [Wall(WIDTH - wall_height, i * wall_height,
                  wall_width, wall_height, -90) for i in range(0, HEIGHT // wall_height + 1)]

    # spawn monsters
    monster_group = pygame.sprite.Group()
    monster_group.add(spawn_monster())

    # set default boolean to run the game
    run = True

    while run:
        # clock
        clock.tick(FPS)

        # initialise game actions
        get_event = pygame.event.get()
        key_pressed = pygame.key.get_pressed()

        for event in get_event:

            gun_choice(player)

            if event.type == pygame.QUIT:
                # end loop when close window (close game)
                run = False
                break

        all_walls = wall + wall2 + wall3 + wall4 + wall5

        game_over2 = False


        # handle movement
        player.loop(FPS)
        all_collision_objects = barrels + all_walls + list(monster_group.sprites())
        check_collision(player, list(monster_group.sprites()), time.time())
        player_movement(player, all_collision_objects)

        # draw everything in the game
        draw(game_window, bg_tiles, bg_image,
             player, barrel_group, block_group, all_walls, monster_group)

        # present 2 scenarios where game over happens, 1 from player dying and another from monster bugging out
        game_over = check_collision(player, list(monster_group.sprites()), time.time())
        if game_over:
            break
        if game_over2:
            break

        # updates the display
        pygame.display.flip()

    print(f'Total points: {points}')

    pygame.quit()
    quit()


if __name__ == '__main__':
    main(window)
