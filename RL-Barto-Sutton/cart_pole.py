import gym
import numpy as np

env = gym.make('CartPole-v0')

for ep in range(20):
  oberservation = env.reset()
  for _ in range(1000):
    env.render()
    print(oberservation)
    action = env.action_space.sample()
    oberservation, reward, done, info = env.step(action) # take a random action
    if done: break

env.close()