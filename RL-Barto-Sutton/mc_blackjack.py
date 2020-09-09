from collections import defaultdict
import numpy as np

from blackjack import BlackjackEnv


def run_full_episode(env, policy):
  '''
  Plays a full game of blackjack
  and returns a cache full of
  (state, action, reward) tuples for each step
  '''
  episode_cache = list()
  state = env.reset()
  while True:
    action = policy(state)
    new_state, reward, done, _ = env.step(action)
    episode_cache.append((state, action, reward))
    if done: break
    state = new_state
  return episode_cache

def make_stochastic_policy(num_actions):
  '''Generates a stochastic policy'''
  def policy(observation):
    action_probs = np.ones(num_actions) / num_actions
    action_choice = np.random.choice(np.arange(num_actions), p=action_probs)
    return action_choice, action_probs
  return policy

def make_greedy_policy(state_action_values):
  '''Generates a greedy policy''' 
  def policy(observation):
    best_action = np.argmax(state_action_values[observation])
    action_probs = np.zeros(len(state_action_values[observation]))
    action_probs[best_action] = 1.0
    return best_action, action_probs
  return policy

def make_epsilon_greedy_policy(state_action_values, epsilon):
  '''Epsilon Greedy Policy Generator'''
  def policy(observation):
    num_actions = len(state_action_values[observation])
    action_probs = np.ones(num_actions, dtype=float) * epsilon / num_actions
    best_action = np.argmax(state_action_values[observation])
    action_probs[best_action] += (1.0 - epsilon)
    action_choice = np.random.choice(np.arange(len(action_probs)), p=action_probs)
    return action_choice, action_probs
  return policy

class OnPolicy(object):
  '''
  On Policy First-Visit MC Control
  for Epsilon Greedy Policies via Sample Averaging
  '''
  @staticmethod
  def mc_control_epsilon_greedy(env=BlackjackEnv(),
                                num_episodes=10000,
                                discount_factor=1.0,
                                epsilon=0.1):
    num_actions = env.action_space.n
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    state_action_values = defaultdict(lambda: np.zeros(num_actions))
    policy = make_epsilon_greedy_policy(state_action_values, epsilon)

    for _ in range(num_episodes):
      episode_cache = run_full_episode(env, policy)
      state_action_pairs = set((tuple(e[0]), e[1]) for e in episode_cache)
      for state, action in state_action_pairs:
        curr_pair = (state, action)
        first_visit = next(i for i, e in enumerate(episode_cache)
          if e[0] == state and e[1] == action)
        returns_sum[curr_pair] += sum([e[2] * (discount_factor**i)
          for i, e in enumerate(episode_cache[first_visit:])])
        returns_count[curr_pair] += 1
        state_action_values[state][action] += returns_sum[curr_pair] / returns_count[curr_pair]

    return state_action_values, policy

class OffPolicy(object):
  '''
  Off Policy MC Control via Importance Sampling
  '''
  pass








'''
Q, policy = OnPolicy.mc_control_epsilon_greedy(num_episodes=500000)
for state, actions in Q.items():
  print(state, 'HIT' if policy(state) == 1 else 'STICK')
'''