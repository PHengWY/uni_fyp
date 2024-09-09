# from stable_baselines3 import DQN
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
# from tf_agents.agents.dqn.dqn_agent import DQNAgent
# from tf_agents.policies import random_policy
# from tf_agents.environments import utils
# from tf_agents.networks import Conv2DNetwork
# from tf_agents.replay_buffers import ReplayBuffer
# from tf_agents.trajectories import trajectory
# from tf_agents.utils import np_utils
from collections import deque

import numpy as np
# import matplotlib.pyplot as plt
import random


class DQN:
    def __init__(self, input_shape, action_size, lr_max=0.001,
                 lr_decay=0.995, gamma=0.99,
                 memory_size=2000, batch_size=128,
                 eps_max=1.0,
                 eps_min=0, eps_decay=0.999,
                 activation='relu', tau=0.5):
        # input shape
        self.input_shape = input_shape
        self.tensor_shape = (-1,) + input_shape
        self.action_size = action_size
        # learning rate
        self.lr_max = lr_max
        self.lr = self.lr_max
        self.lr_decay = lr_decay
        self.gamma = gamma
        # memory
        self.memory_size = memory_size
        self.memory = ReplayBuffer(self.memory_size)
        self.batch_size = batch_size
        # epsilon
        self.eps_start = eps_max
        self.eps_max = eps_max
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        # model activation tech
        self.activation = activation
        self.tau = tau

        # 1 model for training and 1 for baseline
        self.model = self.build_model()
        self.compiled_model = self.optimise_model(self.model)
        self.baseline_model = self.build_model()
        self.compiled_baseline_model = self.optimise_model(self.baseline_model)

    # define a convolutional neural network
    def build_model(self):
        model = models.Sequential()
        model.add(Conv2D(2**5, (3, 3), activation=self.activation, input_shape=self.input_shape))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(2**6, (3, 3), activation=self.activation))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(2**6, (3, 3), activation=self.activation))
        # model.add(MaxPooling2D((2, 2)))
        # model.add(Conv2D(2**7, (3, 3), activation=self.activation))
        model.add(Flatten())
        model.add(Dense(2**8, activation=self.activation))
        model.add(Dropout(0.1))
        model.add(Dense(2**7, activation=self.activation))
        model.add(Dropout(0.2))
        model.add(Dense(self.action_size, activation='softmax'))
        return model

    def optimise_model(self, model):
        model.compile(optimizer=Adam(learning_rate=self.lr), loss='mse', metrics=['accuracy'])
        return model

    # as the model gradually improves, update baseline model to scale with improvements
    def update_baseline_model(self):
        self.compiled_baseline_model.set_weights(self.model.get_weights())

    def transition(self, state, action, reward, next_state, done):
        self.memory.push((state, action, reward, next_state, done))

    def predict(self, state, eps=None):
        if eps is None:
            eps = self.eps_max
        if np.random.rand() <= eps:
            return random.randrange(self.action_size)
        return np.argmax(self.baseline_model.predict(state, verbose=1)[0])

    def replay_memory(self, episode=0):
        if len(self.memory) < self.batch_size:
            return None

        # Assuming self.memory is an instance of ReplayBuffer
        experiences, indices, weights = self.memory.sample(self.batch_size)

        # The rest of your code can remain largely unchanged
        unpacked_experiences = list(zip(*experiences))
        states, actions, rewards, next_states, dones = [list(arr) for arr in unpacked_experiences]

        # Convert state, action, reward, next_state to tensors
        states = tf.convert_to_tensor(states)
        states = tf.reshape(states, self.tensor_shape)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states)
        next_states = tf.reshape(next_states, self.tensor_shape)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        # Compute Q values and next Q values
        target_q_values = self.baseline_model.predict(next_states, verbose=1)
        q_values = self.model.predict(states, verbose=1)

        # Compute target values using the Bellman equation
        max_target_q_values = np.max(target_q_values, axis=1)
        targets = rewards + (1 - dones) * self.gamma * max_target_q_values

        # Compute TD errors
        batch_indices = np.arange(self.batch_size)
        q_values_current_action = q_values[batch_indices, actions]
        td_errors = targets - q_values_current_action

        q_values[batch_indices, actions] = targets
        loss = self.model.train_on_batch(states, q_values, sample_weight=weights)

        self.eps_start = self.eps_max * self.eps_decay ** episode
        self.eps_start = max(self.eps_min, self.eps_start)
        self.lr = self.lr_max * self.lr_decay ** episode
        self.model.optimizer.learning_rate.assign(self.lr)

        return loss

    def load_model(self, name):
        self.model = tf.keras.models.load_model(name)
        self.baseline_model = tf.keras.models.load_model(name)

    def save_model(self, name):
        self.model.save(f'{name}.keras')


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        # Randomly sample a batch of experiences from the buffer
        experiences = random.sample(self.buffer, batch_size)
        # For a simple deque-based replay buffer, indices and weights are not needed
        # However, if you want to mimic the output of the prioritized replay buffer
        # to make replacing it easier, you could return indices and uniform weights
        indices = list(range(len(experiences)))
        weights = [1.0] * batch_size  # Uniform weights since there's no prioritization
        return experiences, indices, weights

    def __len__(self):
        return len(self.buffer)

