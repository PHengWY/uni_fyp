import numpy as np
from collections import deque
import pygame

import game_for_ai
import DQN

LR = 1e-4
LR_DECAY = 0.99
EPS_DECAY = 0.95
GAMMA = 0.975
UPDATE_TARGET_EVERY = 10

BATCH_SIZE = 16
EPISODES = 101

env = game_for_ai.Env()
agent = DQN.DQN(
    input_shape=env.ENV_SHAPE,
    action_size=env.ACTION_SPACE_SIZE,
    batch_size=BATCH_SIZE,
    lr_max=LR,
    lr_decay=LR_DECAY,
    eps_decay=EPS_DECAY,
    gamma=GAMMA
)
agent.save_model(f'models/-1.h5')

state, _, _ = env.reset()
state = np.expand_dims(state, axis=0)

most_recent_losses = deque(maxlen=BATCH_SIZE)

log = []

# fill up memory before training starts
while agent.memory.__len__() < BATCH_SIZE:
    action = agent.predict(state)
    next_state, reward, done, points, _, _, _, _, _ = env.step(env.player,
                                                               env.monster_group,
                                                               action)
    next_state = np.expand_dims(next_state, axis=0)
    agent.transition(state, action, reward, next_state, done)
    state = next_state

for e in range(0, EPISODES):
    state, _, _ = env.reset()
    state = np.expand_dims(state, axis=0)
    done = False
    step = 0
    ma_loss = None

    while not done:
        action = agent.predict(state)
        next_state, reward, done, points, _, _, _, _, _ = env.step(env.player,
                                                                   env.monster_group,
                                                                   action)
        next_state = np.expand_dims(next_state, axis=0)
        agent.transition(state, action, reward, next_state, done)

        state = next_state
        step += 1

        loss = agent.replay_memory(episode=e)
        most_recent_losses.append(loss)
        ma_loss = np.array(most_recent_losses).mean()

        if loss is not None:
            print(f"Step: {step}. Points Attained: {points}. -- Loss: {loss}", end="          \r")

        if done:
            print(f"Episode {e}/{EPISODES-1} completed with {step} steps. Score: {points:.0f}. LR: {agent.lr:.6f}. EP: {agent.eps_start:.2f}. MA loss: {ma_loss:.6f}")
            break

    log.append([e, step, points, agent.lr, agent.eps_start, ma_loss])

    agent.save_model(f'models/{e}.h5')