import numpy as np
from collections.abc import Iterable

class GridWorldPolicy:

  ACTIONS = {
    'right': np.array([0, 1]),
    'left': np.array([0, -1]),
    'up': np.array([1, 0]),
    'down': np.array([-1, 0])
  }

  ACTION_PROBABILITY = 1 / len(ACTIONS)

  def __init__(self,
              shape=(4,4),
              terminal_states=[[0,0], [3,3]]):
    self.reset(shape=shape,
              terminal_states=terminal_states)

  def reset(self, shape=None, terminal_states=None):
    if shape is not None:
      assert isinstance(shape, Iterable), shape
      self.state_values = np.zeros(shape)
    else:
      self.state_values = np.zeros(self.state_values.shape)

    if terminal_states is not None:
      assert isinstance(terminal_states, Iterable), terminal_states
      self.terminal_states = terminal_states
      for terminal_state in self.terminal_states:
        assert isinstance(terminal_state, Iterable) and \
          self._in_bounds(terminal_state, self.state_values.shape), \
            f'Invalid terminal state {terminal_state}'
    
  @staticmethod
  def _in_bounds(state, shape):
    '''Returns true if the state exists in current world'''
    row, col = state
    row_shape, col_shape = shape
    return (row > -1 and row < row_shape) and \
      (col > -1 and col < col_shape)
  
  def _step(self, state, action):
    if state in self.terminal_states:
      return state, 0

    next_state = (np.array(state) + action).tolist()
    if not self._in_bounds(next_state, self.state_values.shape):
      next_state = state

    return next_state, -1

  def compute_optimal_policy(self, discount=1e0, max_delta_value=1e-4):
    '''
    Returns tuple of the optimal policy, and
    the epochs it took to achieve optimality
    '''
    epochs = 0
    while True:
      # loop until convergence
      old_state_vals = self.state_values.copy()
      for row in range(self.state_values.shape[0]):
        for col in range(self.state_values.shape[1]):
          self._update_state_value([row, col], discount)

      delta_value = abs(old_state_vals - self.state_values).max()
      if delta_value < max_delta_value: # converged
        break
      
      epochs += 1

    return self.state_values, epochs

  def _update_state_value(self, state, discount):
    '''
    Updates the value for the current state 
    using the bellman optimality equation
    '''
    value = 0
    row, col = state
    for _, action in self.ACTIONS.items():
      next_state, reward = self._step([row, col], action)
      next_row, next_col = next_state
      next_state_value = self.state_values[next_row, next_col]
      value += self.ACTION_PROBABILITY * \
        (reward + discount * next_state_value)
    self.state_values[row, col] = value


def get_path_starting_from(starting_state, optimal_policy):
  path = [starting_state]
  current_state = starting_state
  current_return = optimal_policy[starting_state[0], starting_state[1]]

  while current_return != 0:
    # loop until we hit the terminal state
    adjacent_values = {}
    for k, action in GridWorldPolicy.ACTIONS.items():
      next_state = (np.array(current_state) + action).tolist()
      if GridWorldPolicy._in_bounds(next_state, optimal_policy.shape):
        adjacent_values[f'{optimal_policy[next_state[0], next_state[1]]}'] = k

    # choose action that leads to highest return
    current_return = np.max([float(key) for key in adjacent_values.keys()])
    best_action = adjacent_values[f'{current_return}']
    current_state = (np.array(current_state) + np.array(GridWorldPolicy.ACTIONS[best_action])).tolist()
    path.append(current_state)
  
  return path

if __name__ == '__main__':
  c = GridWorldPolicy(shape=(15,15), terminal_states=[[0,0], [14,14]])
  policy , epochs = c.compute_optimal_policy()
  print(get_path_starting_from([7,7], policy))

      





