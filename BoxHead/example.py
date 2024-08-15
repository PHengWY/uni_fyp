import pygame
import numpy as np
import tensorflow as tf


class Field:
    # class for compiling the array that the DQN will interpret
    def __init__(self, height=10, width=5):
        self.width = width
        self.height = height
        self.clear_field()

    def clear_field(self):
        self.body = np.zeros(shape=(self.height, self.width))

    def update_field(self, fruits, player):
        self.clear_field()

        # draw fruits
        for fruit in fruits:
            if not fruit.out_of_field:
                for y in range(fruit.y, min(fruit.y + fruit.height, self.height)):
                    for x in range(fruit.x, min(fruit.x + fruit.width, self.width - 1)):
                        self.body[y][x] = 1

        # draw player
        for i in range(player.width):
            self.body[player.y][player.x + i] = 2


class Fruit:
    # class for the fruit
    def __init__(self, height=1, width=1, x=None, y=0, speed=1, field=None):
        self.field = field
        self.height = height
        self.width = width
        self.x = self.generate_x() if x == None else x
        self.y = y
        self.speed = speed
        self.out_of_field = False
        self.is_caught = 0

    def generate_x(self):
        return np.random.randint(0, self.field.width - self.width)

    def set_out_of_field(self):
        self.out_of_field = True if (self.y > self.field.height - 1) else False

    def move(self):
        self.y += self.speed
        self.set_out_of_field()

    def set_is_caught(self, player):
        if self.y != player.y:
            self.is_caught = 0
        else:
            if self.x + self.width > player.x and (self.x < player.x + player.width):
                self.is_caught = 1
            else:
                self.is_caught = -1


class Player:
    # class for the player
    def __init__(self, height=1, width=1, field=None):
        self.field = field
        self.height = height
        self.width = width
        self.x = int(self.field.width / 2 - width / 2)
        self.last_x = self.x
        self.y = self.field.height - 1
        self.dir = 0
        self.colour = "blue"

    def move(self):
        self.last_x = self.x
        self.x += self.dir
        self.dir = 0
        self.constrain()

    def action(self, action):
        if action == 1:
            self.dir = -1
        elif action == 2:
            self.dir = 1
        else:
            self.dir = 0

    def constrain(self):
        if self.x < 0:
            self.x = self.field.width - self.width
        elif (self.x + self.width) > self.field.width:
            self.x = 0


class Environment:
    # class for the environment
    F_HEIGHT = 12
    F_WIDTH = 12
    PLAYER_WIDTH = 2
    FRUIT_WIDTH = 1

    ENVIRONMENT_SHAPE = (F_HEIGHT, F_WIDTH, 1)
    ACTION_SPACE = [0, 1, 2]
    ACTION_SPACE_SIZE = len(ACTION_SPACE)
    ACTION_SHAPE = (ACTION_SPACE_SIZE,)
    PUNISHMENT = -1
    REWARD = 1
    score = 0
    MAX_VAL = 2

    LOSS_SCORE = -5
    WIN_SCORE = 5

    DRAW_MUL = 30
    WINDOW_HEIGHT = F_HEIGHT * DRAW_MUL
    WINDOW_WIDTH = F_WIDTH * DRAW_MUL

    game_tick = 0
    FPS = 20
    MOVE_FRUIT_EVERY = 1
    MOVE_PLAYER_EVERY = 1
    MAX_FRUIT = 1
    INCREASE_MAX_FRUIT_EVERY = 100
    SPAWN_FRUIT_EVERY_MIN = 2
    SPAWN_FRUIT_EVERY_MAX = 12
    next_spawn_tick = 0

    FRUIT_COLOURS = {-1: "red", 0: "black", 1: "green"}

    def __init__(self):
        self.reset()

    def get_state(self):
        return self.field.body / self.MAX_VAL

    def reset(self):
        self.game_tick = 0
        self.game_over = False
        self.game_won = False
        self.field = Field(height=self.F_HEIGHT, width=self.F_WIDTH)
        self.player = Player(field=self.field, width=self.PLAYER_WIDTH)
        self.score = 0
        self.fruits = []
        self.spawn_fruit()
        self.field.update_field(self.fruits, self.player)

        return self.get_state()

    def spawn_fruit(self):
        if len(self.fruits) < self.MAX_FRUIT:
            self.fruits.append(Fruit(field=self.field, height=self.FRUIT_WIDTH, width=self.FRUIT_WIDTH))
            self.set_next_spawn_tick()

    def set_next_spawn_tick(self):
        self.next_spawn_tick = self.game_tick + np.random.randint(self.SPAWN_FRUIT_EVERY_MIN,
                                                                  self.SPAWN_FRUIT_EVERY_MAX)

    def step(self, action=None):
        # this runs every step of the game
        # the QDN can pass an action to the game, and in return gets next game state, reward, etc.

        self.game_tick += 1

        if self.game_tick % self.INCREASE_MAX_FRUIT_EVERY == 0:
            self.MAX_FRUIT += 1

        if self.game_tick >= self.next_spawn_tick or len(self.fruits) == 0:
            self.spawn_fruit()

        if action != None:
            self.player.action(action)
        self.player.move()

        reward = 0

        if self.game_tick % self.MOVE_FRUIT_EVERY == 0:
            in_field_fruits = []
            for fruit in self.fruits:
                fruit.move()
                fruit.set_is_caught(self.player)
                if fruit.is_caught == 1:
                    self.update_score(self.REWARD)
                    reward = self.REWARD
                elif fruit.is_caught == -1:
                    self.update_score(self.PUNISHMENT)
                    reward = self.PUNISHMENT
                if not fruit.out_of_field:
                    in_field_fruits.append(fruit)
            self.fruits = in_field_fruits

        self.field.update_field(fruits=self.fruits, player=self.player)

        if self.score <= self.LOSS_SCORE:
            self.game_over = True

        if self.score >= self.WIN_SCORE:
            self.game_won = True

        return self.get_state(), reward, self.game_over or self.game_won, self.score

    def update_score(self, delta):
        self.score += delta

    def render(self, screen, solo=True, x_offset=0, y_offset=0):
        # for rendering the game
        if solo:
            screen.fill("white")
            pygame.display.set_caption(f"Score: {self.score}")

        # draw player
        pygame.draw.rect(
            screen,
            self.player.colour,
            ((self.player.x * self.DRAW_MUL + x_offset, self.player.y * self.DRAW_MUL + y_offset),
             (self.player.width * self.DRAW_MUL, self.player.height * self.DRAW_MUL))
        )

        # draw fruit
        for fruit in self.fruits:
            pygame.draw.rect(
                screen,
                self.FRUIT_COLOURS[fruit.is_caught],
                ((fruit.x * self.DRAW_MUL + x_offset, fruit.y * self.DRAW_MUL + y_offset),
                 (fruit.width * self.DRAW_MUL, fruit.height * self.DRAW_MUL))
            )


def main():
    # if run as a script, the game is human playable at 15fps
    env = Environment()

    pygame.init()
    screen = pygame.display.set_mode((env.WINDOW_WIDTH, env.WINDOW_HEIGHT))
    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()

        if keys[pygame.K_LEFT]:
            env.player.dir = -1
        if keys[pygame.K_RIGHT]:
            env.player.dir = 1

        env.step()

        env.render(screen)

        pygame.display.flip()
        clock.tick(15)

    pygame.quit()

    return 0


if __name__ == "__main__":
    main()


class DQN:
    def __init__(self, state_shape, action_size, learning_rate_max=0.001, learning_rate_decay=0.995, gamma=0.75,
                 memory_size=2000, batch_size=32, exploration_max=1.0, exploration_min=0.01, exploration_decay=0.995):
        self.state_shape = state_shape
        self.state_tensor_shape = (-1,) + state_shape
        self.action_size = action_size
        self.learning_rate_max = learning_rate_max
        self.learning_rate = learning_rate_max
        self.learning_rate_decay = learning_rate_decay
        self.gamma = gamma
        self.memory_size = memory_size
        self.memory = PrioritizedReplayBuffer(capacity=2000)
        self.batch_size = batch_size
        self.exploration_rate = exploration_max
        self.exploration_max = exploration_max
        self.exploration_min = exploration_min
        self.exploration_decay = exploration_decay

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        # the actual neural network structure
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input(shape=self.state_shape))
        model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform',
                                         input_shape=self.state_shape))
        model.add(
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform'))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
        model.add(tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
        model.add(tf.keras.layers.Dropout(0.1))
        model.add(tf.keras.layers.Dense(self.action_size, activation='linear', name='action_values',
                                        kernel_initializer='he_uniform'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.push((state, action, reward, next_state, done))

    def act(self, state, epsilon=None):
        if epsilon == None:
            epsilon = self.exploration_rate
        if np.random.rand() < epsilon:
            return random.randrange(self.action_size)
        return np.argmax(self.target_model.predict(state, verbose=0)[0])

    def replay(self, episode=0):

        if self.memory.length() < self.batch_size:
            return None

        experiences, indices, weights = self.memory.sample(self.batch_size)
        unpacked_experiences = list(zip(*experiences))
        states, actions, rewards, next_states, dones = [list(arr) for arr in unpacked_experiences]

        # Convert to tensors
        states = tf.convert_to_tensor(states)
        states = tf.reshape(states, self.state_tensor_shape)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states)
        next_states = tf.reshape(next_states, self.state_tensor_shape)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        # Compute Q values and next Q values
        target_q_values = self.target_model.predict(next_states, verbose=0)
        q_values = self.model.predict(states, verbose=0)

        # Compute target values using the Bellman equation
        max_target_q_values = np.max(target_q_values, axis=1)
        targets = rewards + (1 - dones) * self.gamma * max_target_q_values

        # Compute TD errors
        batch_indices = np.arange(self.batch_size)
        q_values_current_action = q_values[batch_indices, actions]
        td_errors = targets - q_values_current_action
        self.memory.update_priorities(indices, np.abs(td_errors))

        # For learning: Adjust Q values of taken actions to match the computed targets
        q_values[batch_indices, actions] = targets

        loss = self.model.train_on_batch(states, q_values, sample_weight=weights)

        self.exploration_rate = self.exploration_max * self.exploration_decay ** episode
        self.exploration_rate = max(self.exploration_min, self.exploration_rate)
        self.learning_rate = self.learning_rate_max * self.learning_rate_decay ** episode
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, self.learning_rate)

        return loss

    def load(self, name):
        self.model = tf.keras.models.load_model(name)
        self.target_model = tf.keras.models.load_model(name)

    def save(self, name):
        self.model.save(name)


class PrioritizedReplayBuffer:
    def __init__(self, capacity, epsilon=1e-6, alpha=0.8, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.epsilon = epsilon
        self.alpha = alpha  # how much prioritisation is used
        self.beta = beta  # for importance sampling weights
        self.beta_increment = beta_increment
        self.priority_buffer = np.zeros(self.capacity)
        self.data = []
        self.position = 0

    def length(self):
        return len(self.data)

    def push(self, experience):
        max_priority = np.max(self.priority_buffer) if self.data else 1.0
        if len(self.data) < self.capacity:
            self.data.append(experience)
        else:
            self.data[self.position] = experience
        self.priority_buffer[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        priorities = self.priority_buffer[:len(self.data)]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.data), batch_size, p=probabilities)
        experiences = [self.data[i] for i in indices]

        total = len(self.data)
        weights = (total * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()

        self.beta = np.min([1., self.beta + self.beta_increment])

        return experiences, indices, weights

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            self.priority_buffer[idx] = error + self.epsilon


import numpy as np
from collections import deque

import game
import dqn

LEARNING_RATE = 1e-4
LEARNING_RATE_DECAY = 0.99
EXPLORATION_DECAY = 0.95
GAMMA = 0.975
UPDATE_TARGET_EVERY = 10

BATCH_SIZE = 128
EPISODES = 101

env = game.Environment()
agent = dqn.DQN(
    state_shape=env.ENVIRONMENT_SHAPE,
    action_size=env.ACTION_SPACE_SIZE,
    batch_size=BATCH_SIZE,
    learning_rate_max=LEARNING_RATE,
    learning_rate_decay=LEARNING_RATE_DECAY,
    exploration_decay=EXPLORATION_DECAY,
    gamma=GAMMA
)
agent.save(f'models/-1.h5')

state = env.reset()
state = np.expand_dims(state, axis=0)

most_recent_losses = deque(maxlen=BATCH_SIZE)

log = []

# fill up memory before training starts
while agent.memory.length() < BATCH_SIZE:
    action = agent.act(state)
    next_state, reward, done, score = env.step(action)
    next_state = np.expand_dims(next_state, axis=0)
    agent.remember(state, action, reward, next_state, done)
    state = next_state

for e in range(0, EPISODES):
    state = env.reset()
    state = np.expand_dims(state, axis=0)
    done = False
    step = 0
    ma_loss = None

    while not done:
        action = agent.act(state)
        next_state, reward, done, score = env.step(action)
        next_state = np.expand_dims(next_state, axis=0)
        agent.remember(state, action, reward, next_state, done)

        state = next_state
        step += 1

        loss = agent.replay(episode=e)
        most_recent_losses.append(loss)
        ma_loss = np.array(most_recent_losses).mean()

        if loss != None:
            print(f"Step: {step}. Score: {score}. -- Loss: {loss}", end="          \r")

        if done:
            print(f"Episode {e}/{EPISODES-1} completed with {step} steps. Score: {score:.0f}. LR: {agent.learning_rate:.6f}. EP: {agent.exploration_rate:.2f}. MA loss: {ma_loss:.6f}")
            break

    log.append([e, step, score, agent.learning_rate, agent.exploration_rate, ma_loss])

    agent.save(f'models/{e}.h5')

import numpy as np

import game
import dqn

model_path = "models\5.h5"

env = game.Environment()
agent = dqn.DQN(
    state_shape=env.ENVIRONMENT_SHAPE,
    action_size=env.ACTION_SPACE_SIZE
)
agent.load(model_path)

state = env.reset()
state = np.expand_dims(state, axis=0)

import pygame
pygame.init()
screen = pygame.display.set_mode((env.WINDOW_WIDTH, env.WINDOW_HEIGHT))
clock = pygame.time.Clock()
running = True
score = 0

while running:
    pygame.display.set_caption(f"Score: {score}")
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    action = agent.act(state, 0)
    state, reward, done, score = env.step(action)
    state = np.expand_dims(state, axis=0)

    env.render(screen)
    pygame.display.flip()
    clock.tick(30)

pygame.quit()

