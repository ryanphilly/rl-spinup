import gym
import numpy as np
from tensorflow import nn
from tensorflow.keras import layers, Model, optimizers

class DeepQAproximator(Model):
  '''Feed forward MLP for aproximating Q values'''
  def __init__(self, observation_shape, num_actions, **kwargs):
    super(QAproximator, self).__init__(**kwargs)
    self.dense_relu1 = layers.Dense(32, input_shape=observation_shape, activation=tf.nn.relu)
    self.dense_relu2 = layers.Dense(32, activation=tf.nn.relu)
    self.dense_actions = layers.Dense(num_actions, activation=nn.linear)
    self.compile(optimizer=optimizers.Adam(lr=learning_rate), loss='mse')

  def call(self, observation, training=False):
    return self.dense_actions(
      self.dense_relu2(self.dense_relu1(observation)))

class CartPoleControl(object):
  def __init__(self, discount_factor=1e0, epsilon=0.05, learning_rate=1e-3):
    self.env = gym.make('cartpole-v1')
    self.epsilon = epsilon
    self.discount_factor = discount_factor
    self.aproximator = DeepQAproximator(
      env.observation_space.shape, env.action_space.n)

  def _decide(self, observation):
    '''epsilon greedy behavior policy decision'''
    nA = self.env.actiion_space.n
    state_action_values  = self.aproximator(observation)[0] # Q
    greedy_action = np.argmax(state_action_values)
    probs = np.ones(nA) * self.epsilon / nA
    probs[greedy_action] += (1.0 - self.epsilon)
    decision = np.random.choice(np.arange(nA), probs=probs)
    return decision, state_action_values

  def play_episode(self);
    observation = self.env.reset()
    while True:
      action, Q = self._decide(observation)
      state_prime, reward, terminal, _ = self.env.step(action)
      q_update = reward + float(not terminal) * self.discount_factor * \
        np.max(self.aproximator(state_prime)[0])
      Q[action] += q_update
      self.aproximator.fit(state, Q)
      observation = state_prime


    