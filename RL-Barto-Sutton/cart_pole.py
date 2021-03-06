import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers

reshape = lambda x: np.reshape(x, (1, x.shape[0]))

class DeepQAproximator(Model):
  '''Feed forward MLP for aproximating Q values'''
  def __init__(self, observation_shape, num_actions, learning_rate, **kwargs):
    super(DeepQAproximator, self).__init__(**kwargs)
    self.dense_relu1 = layers.Dense(32, input_shape=observation_shape, activation='relu')
    self.dense_relu2 = layers.Dense(32, activation='relu')
    self.dense_actions = layers.Dense(num_actions, activation='linear')
    self.compile(optimizer=optimizers.Adam(lr=learning_rate), loss='mse')

  def call(self, observation, training=False):
    return self.dense_actions(
      self.dense_relu2(self.dense_relu1(observation)))

class CartPoleControlAgent(object):
  def __init__(self,
              max_memory=1000,
              discount_factor=0.95,
              starting_epsilon=0.95,
              epsilon_decay_rate=0.995,
              learning_rate=1e-3):
    self.env = gym.make('CartPole-v1')
    self.max_memory = max_memory
    self.epsilon = starting_epsilon
    self.epsilon_decay_rate = epsilon_decay_rate
    self.discount_factor = discount_factor
    self.aproximator = DeepQAproximator(
      self.env.observation_space.shape,
      self.env.action_space.n,
      learning_rate)
    self.episode_cache = list()

  def _decide(self, observation):
    '''epsilon greedy behavior policy'''
    nA = self.env.action_space.n
    state_action_values  = self.aproximator.predict(reshape(observation))[0] # Q
    greedy_action = np.argmax(state_action_values)
    probs = np.ones(nA) * self.epsilon / nA
    probs[greedy_action] += (1.0 - self.epsilon)
    decision = np.random.choice(np.arange(nA), p=probs)
    return decision, state_action_values

  def play_episode(self):
    observation = self.env.reset()
    while True:
      self.env.render()
      action, Q = self._decide(observation)
      state_prime, reward, terminal, _ = self.env.step(action)
      reward = -reward if terminal else reward

      if len(self.episode_cache >= self.max_memory):
        episode_cache = episode_cache[len(self.episode_cache) / 2:]
      episode_cache.append((observation, action, reward, state_prime))

      greedy_action = np.max(self.aproximator.predict(reshape(state_prime))[0])
      Q[action] += reward + float(not terminal) * self.discount_factor * \
         greedy_action - Q[action]

      self.aproximator.fit(
        reshape(observation),
        reshape(Q),
        verbose=False)

      if terminal: break
      observation = state_prime

    self.epsilon = max(0.01, self.epsilon * self.epsilon_decay_rate)

agent = CartPoleControlAgent()
while True:
  agent.play_episode()
