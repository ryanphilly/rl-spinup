import gym
import numpy as np
from tensorflow.keras import layers, Model

class DeepQAproximator(Model):
  def __init__(self, observation_shape, num_actions, discount_factor=1e0, epsilon=1e-1, *args, **kwargs):
    super(QAproximator, self).__init__()
    self.dense_relu1 = layers.Dense(32, input_shape=observation_shape, activation=tf.nn.relu)
    self.dense_relu2 = layers.Dense(32, activation=tf.nn.relu)
    self.dense_action = layers.Dense(num_actions, activation=tf.nn.linear)
    self.compile(**kwargs)

  def call(self, observation, training=False):
    pass