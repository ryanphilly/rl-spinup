import numpy as np
from tqdm import trange

class Bandit(object):
  def __init__(self,
              k_arms=10,
              exploration_epsilon=0.1,
              initial_value=0.0,
              real_reward=0.0,
              learning_rate=0.2,
              ucb_confidence=None,
              gradient=False,
              sample_average=False):
    self.k_arms = k_arms
    self.epsilon = exploration_epsilon
    self.learning_rate = learning_rate
    self.initial_value = initial_value
    self.real_reward = real_reward
    self.ucb_confidence = ucb_confidence
    self.sample_average = sample_average
    self.gradient = gradient
    self.is_clean = False
    self.reset()

  def reset(self):
    # check figure 2.1 for how these were initialized
    if not self.is_clean:
      self.action_space = np.arange(self.k_arms)
      self.value_estimations = np.zeros(self.k_arms) + self.initial_value
      self.real_value = np.random.randn(self.k_arms) + self.real_reward
      self.action_count = np.zeros(self.k_arms)
      self.time = 0
      self.average_reward = 0
      self.best_action = np.argmax(self.real_value)
      self.is_clean = True

  def act(self):
    self.is_clean = False
    # explore
    if np.random.rand() < self.epsilon:
      return np.random.choice(self.action_space)

    if self.ucb_confidence is not None:
      ucb_estimations = self.value_estimations + self.ucb_confidence * \
        np.sqrt(np.log(self.time + 1) / (self.action_count + 1e-5))
      best_action = np.max(ucb_estimations)
      return np.random.choice(
        np.where(ucb_estimations == best_action)[0])

    if self.gradient:
      exp_preference = np.exp(self.value_estimations)
      self.action_probability = exp_preference / np.sum(exp_preference)
      return np.random.choice(self.action_space, p=self.action_probability)

    # exploit
    best_action = np.max(self.value_estimations)
    return np.random.choice(
      np.where(self.value_estimations == best_action)[0])

  def step(self, action):
    self.time += 1
    reward = np.random.randn() + self.real_value[action]
    self.action_count[action] += 1
    # update running average
    self.average_reward += (reward - self.average_reward) / self.time
    if self.sample_average:
      # update running average for value estimation( on current action )
      self.value_estimations[action] += \
        (reward - self.value_estimations[action]) / self.action_count[action]
    elif self.gradient:
      # asumming the base line is always average reward
      one_hot = np.zeros(self.k_arms)
      one_hot[action] = 1
      self.action_probability += self.learning_rate * (reward - self.average_reward) * \
        (one_hot - self.action_probability)
    else:
      # temporal difference update
      self.value_estimations[action] += \
        self.learning_rate * (reward - self.value_estimations[action])
    
    return reward

def simulate(runs, time, bandits):
  rewards = np.zeros((len(bandits), runs, time))
  best_action_counts = np.zeros(rewards.shape)
  for i, bandit in enumerate(bandits):
    for r in trange(runs):
      bandit.reset()
      for t in range(time):
        action = bandit.act()
        reward = bandit.step(action)
        rewards[i, r, t] = reward
        if action == bandit.best_action:
          best_action_counts[i, r, t] = 1
  mean_best_action_counts = best_action_counts.mean(axis=1)
  mean_rewards = rewards.mean(axis=1)
  return mean_best_action_counts, mean_rewards

if __name__ == '__main__':
  _, _ = simulate(2000, 1000, [Bandit(ucb_confidence=2, sample_average=True)])








