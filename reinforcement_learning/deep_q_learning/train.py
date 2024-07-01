#!/usr/bin/env python3
""" Atari """

import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
import matplotlib.pyplot as plt

# Create the environment
env = gym.make('Breakout-v4')

# Reset the environment to get the initial state
env.reset()


def create_q_model(actions, window=4):
    """ Define the Q-network model"""
    model = keras.Sequential()
    model.add(
        keras.layers.Conv2D(32, (8, 8),
                            strides=(4, 4), activation='relu',
                            input_shape=(window, 84, 84, 1)))
    model.add(
        keras.layers.Conv2D(64, (2, 2),
                            strides=(2, 2), activation='relu'))
    model.add(
        keras.layers.Conv2D(64, (3, 3),
                            strides=(1, 1), activation='relu'))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dense(actions, activation='linear'))
    return model


# Instantiate the model
model = create_q_model(actions=env.action_space.n)

# Set up the experience replay memory
memory = SequentialMemory(limit=50000, window_length=4)

# Set up the policy
policy = EpsGreedyQPolicy()

# Create the DQN agent
agent = DQNAgent(
    model=model,
    nb_actions=env.action_space.n,
    memory=memory,
    nb_steps_warmup=10,
    target_model_update=1e-2,
    policy=policy)

# Compile the agent with optimizer and metrics
agent.compile(keras.optimizers.RMSprop(learning_rate=0.00025), metrics=['mae'])

# Train the agent
agent.fit(env, nb_steps=50000, visualize=False, verbose=1)

# Save the trained weights
agent.save_weights('policy.h5', overwrite=True)

# Close the environment
env.close()
