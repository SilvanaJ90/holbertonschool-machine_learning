#!/usr/bin/env python3
import numpy as np
from policy_gradient import policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False): # Added show_result parameter with default value
    """
    - env: initial environment
    - nb_episodes: number of episodes used for training
    - alpha: the learning rate
    - gamma: the discount factor
    - show_result: flag to indicate whether to render the environment
    Return: all values of the score
    (sum of all rewards during one episode loop)
    """
    scores = []
    weight = np.random.rand( # Fixed typo: np.ramdom -> np.random
        env.observation_space.shape[0],
        env.action_space.n
    )

    for episode in range(nb_episodes):
        state = env.reset()[None, :]

        grads = []
        rewards = []
        actions = []
        done = False

        while not done:
            if show_result is True and episode % 1000 == 0:
                env.render()

            action, grad = policy_gradient(state, weight)

            state, reward, done, _ = env.step(action)

            state = state[None, :]

            grads.append(grad)
            rewards.append(reward)
            actions.append(action)

        for i in range(len(grads)):
            reward = sum([R * gamma ** R for R in rewards[i:]])
            weight += alpha * grads[i] * reward
        
        scores.append(sum(rewards))

        print('Episode: {}  Score: {}'.format(episode, scores[episode]),
              end='\r', flush=False)
        
        return scores