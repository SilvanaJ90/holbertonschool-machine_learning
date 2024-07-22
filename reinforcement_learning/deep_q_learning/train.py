#!/usr/bin/env python3
""" Atari """

import tensorflow as tf
import keras
from keras import __version__ as keras_version
from tensorflow.keras import layers, models
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory
import numpy as np
import cv2
from rl.core import Processor
import keras
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from keras.layers import Dense, Flatten, Conv2D
import gym

def create_q_model(actions=4, window=4):
    # Define the input layer with shape (window, 84, 84)
    inputs = layers.Input(shape=(window, 84, 84))
    
    # Permute dimensions to (height, width, channels)
    x = layers.Permute((2, 3, 1))(inputs)
    
    # Apply convolutional layers
    x = layers.Conv2D(32, 8, strides=4, activation='relu', data_format="channels_last")(x)
    x = layers.Conv2D(64, 4, strides=2, activation='relu', data_format="channels_last")(x)
    x = layers.Conv2D(64, 3, strides=1, activation='relu', data_format="channels_last")(x)
    
    # Flatten the tensor to feed into fully connected layers
    x = layers.Flatten()(x)
    
    # Apply fully connected layers
    x = layers.Dense(512, activation='relu')(x)
    action = layers.Dense(actions, activation='linear')(x)
    
    # Create and return the model
    model = models.Model(inputs=inputs, outputs=action)
    
    return model

class AtariProcessor(Processor):
    def process_observation(self, observation):
        img = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (84, 84))
        img = img.reshape(84, 84, 1)
        return img

    def process_state_batch(self, batch):
  
        process_batch = batch.astype('float32') / 255.
        return process_batch

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)

env = gym.make('Breakout-v4')


# Reiniciar el entorno para obtener el primer estado
env.reset()
model = create_q_model()

policy = LinearAnnealedPolicy(
    EpsGreedyQPolicy(),
    attr='eps',
    value_max=1.,
    value_min=.1,
    value_test=.05,
    nb_steps=40000
)

memory = SequentialMemory(
    limit=1000000,
    window_length=4,  # Asegúrate de que la longitud de la ventana es correcta
)


# Create the DQN agent
agent = DQNAgent(
    model=model,
    nb_actions=4,
    policy=policy,
    memory=memory,
    processor=AtariProcessor(),
    gamma=.99,
    train_interval=4,
    delta_clip=1.,
)

# Compile the agent with optimizer and metrics
agent.compile(Adam(learning_rate=0.0001), metrics=['mae'])


# Train the agent
agent.fit(
    env,
    nb_steps=1000000,
    log_interval=1000,
    visualize=False,
    verbose=2
)

# Save the model weights
agent.save_weights('policy.h5', overwrite=True)

# Close the environment
env.close()

