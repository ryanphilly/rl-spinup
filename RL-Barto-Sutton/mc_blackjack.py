from collections import defaultdict
import numpy as np
import gym


def run_full_episode(env, policy):
  '''
  Plays a full game of blackjack
  and returns a cache full of
  (state, action, reward) tuples for each step
  '''
  episode_cache = list()
  state = env.reset()
  while True:
    action = policy(state)[0]
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


def mc_control_epsilon_greedy(env=BlackjackEnv(),
                              num_episodes=10000,
                              discount_factor=1.0,
                              epsilon=0.1):
  '''
  On Policy First-Visit MC Control via Sample Averaging
  for an Epsilon Greedy Policy
  '''
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

def mc_control_weighted_importanace_sampling(env=BlackjackEnv(),
                                            num_episodes=100000,
                                            behavior_policy=None,
                                            discount_factor=1.0):
  '''Off Policy MC Control via Weighted Importance Sampling'''
  num_actions = env.action_space.n
  state_action_values = defaultdict(lambda: np.zeros(num_actions))
  target_policy = make_greedy_policy(state_action_values)
  behavior_policy = make_stochastic_policy(num_actions)
  if behavior_policy is not None:
    assert callable(behavior_policy), behavior_policy
    behavior_policy = behavior_policy
  cummulative_weight_sums = defaultdict(lambda: np.zeros(num_actions))

  for _ in range(num_episodes):
    episode_cache = run_full_episode(env, behavior_policy)
    returns, weights = 0.0, 1.0
    for ep in reversed(range(len(episode_cache))):
      state, action, reward = episode_cache[ep]
      returns = discount_factor * returns + reward
      cummulative_weight_sums[state][action] += weights
      state_action_values[state][action] += (weights / cummulative_weight_sums[state][action]) * \
        (returns - state_action_values[state][action])
      if action == target_policy(state)[0]:
        # I S Ratio: target(a|s) / behavior(a|s)
        # target greedy policy will always be 1
        weights *= (1.0 / behavior_policy(state)[1][action])

  return state_action_values, target_policy

Q, policy = OffPolicy.mc_control_weighted_importanace_sampling(num_episodes=500000)
for state, actions in Q.items():
  print(state, 'HIT' if policy(state)[0] == 1 else 'STICK')