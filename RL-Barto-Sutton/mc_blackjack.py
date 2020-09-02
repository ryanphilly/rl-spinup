import numpy as np
from enum import Enum
from collections.abc import Iterable

class ACTIONS(Enum):
  HIT = 0
  STICK = 1
class REWARDS(Enum):
  WIN = 1
  LOSE = -1
  DRAW = 0

class Player(object):
  def __init__(self, initial_stick_states=None):
    self.policy = np.array([ACTIONS.HIT for _ in range(22)], dtype=ACTIONS)
    self.card_total = 0
    self.useable_ace = False
    self.bust = lambda: self.card_total > 21
    if initial_stick_states is None:
      self.policy[21] = self.policy[20] = ACTIONS.STICK
    else:
      self._update_policy({ ACTIONS.STICK: initial_stick_states })

  def _update_policy(self, action_state_map):
    assert isinstance(action_state_map, dict), action_state_map
    for action, states in action_state_map.items():
      assert action in ACTIONS, ACTIONS
      assert isinstance(states, Iterable), states

      for state in states:
        is_int = isinstance(state, int)
        is_iter = isinstance(state, Iterable)
        assert is_int or is_iter, state
        if is_iter:
          assert len(state) == 2, state
          assert state[0] >= 0 and state[1] <= 22, state
          for i in range(state[0], state[1]):
            self.policy[i] = action
        else:
          assert 21 >= state >= 0, state
          self.policy[state] = action

class BlackjackEpisode(object):
  draw_card = lambda: min(np.random.randint(1, 14), 10)
  card_value = lambda card: 11 if card == 1 else card
  def __init__(self, player, dealer):
    self.player = player
    self.dealer = dealer
    self.winner = None
    self.is_won = lambda: self.winner is not None
    self.is_dirty = False

  def _reset(self):
    pass
  
  def play(self):
    if self.is_dirty:
      self.reset()

    self._create_random_initial_state()

  def _create_random_initial_state(self):
    while self.player.card_total < 12:
      new_card_value = self.card_value(self.draw_card())
      self.player.card_total += new_card_value
      if self.player.card_total == 22:
        # two aces were drawn by player
        self.player.card_total -= 10
      else:
        self.player.useable_ace = (new_card_value == 11)

    dealer_first_card = self.card_value(self.draw_card())
    dealer_second_card = self.card_value(self.draw_card())
    self.dealer.useable_ace = 11 in [dealer_first_card, dealer_second_card]
    self.dealer.card_total += dealer_first_card + dealer_second_card
    if self.dealer.card_total == 22:
      # dealer drew two aces
      self.dealer.card_total -= 10
    



'''
player = Player(initial_stick_states=[20, 21])
dealer = Player(initial_stick_states=[(17, 22)])
print(player.policy)
print(dealer.policy)

c = BlackjackEpisode(player, dealer)
c.winner = c.player
c.winner = None
print(c.is_won())
'''