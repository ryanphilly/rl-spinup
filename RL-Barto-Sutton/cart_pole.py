import gym
import numpy as np
from tensorflow.keras import layers, Model

class DeepQAproximator(Model):
  def __init__(self, observation_shape, num_actions, discount_factor=1e0, epsilon=1e-1, **kwargs):
    super(QAproximator, self).__init__()
    self.dense_relu1 = layers.Dense(32, input_shape=observation_shape, activation=tf.nn.relu)
    self.dense_relu2 = layers.Dense(32, activation=tf.nn.relu)
    self.dense_action = layers.Dense(num_actions, activation=tf.nn.linear)
    self.compile(**kwargs)

  def call(self, observation, training=False):
    Y = self.dense_relu1(observation)
    Y = self.dense_relu2(Y)
    return self.dense_action(Y)

class CartPoleControl(object):
  def __init__(self, discount_factor=1e0, epsilon=1e-1):
    self.env = gym.make('cartpole-v1')
    pass