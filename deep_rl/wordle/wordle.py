import os
from typing import Optional, List

import gym
from gym import spaces
import numpy as np

import state # Changed from wordle.state to avoid import problems. Changed variables in this file too

WORDLE_N = 5
REWARD = 10
# from wordle.const import WORDLE_N, REWARD  # Commented out to avoid import problems

CUR_PATH = os.environ.get('PYTHONPATH', '.') # setting CUR_PATH to the value of the 'PYTHONPATH' environment variable if it exists, and if it doesn't exist, it sets CUR_PATH to a default value of '.' (the current directory).
import os
dirname = os.path.dirname(__file__) # dirname will contain the directory path of the current script or module
VALID_WORDS_PATH = f'{dirname}/../../data/wordle_words.txt' # constructs the VALID_WORDS_PATH by combining the directory path stored in the dirname variable with a relative path to a specific file.

# Takes an optinal argument (limit) which limits the number of words that will be loaded from the wordle_words.txt file (only the 1st "limit" words will be returned)
def _load_words(limit: Optional[int]=None) -> List[str]:
    with open(VALID_WORDS_PATH, 'r') as f:
        lines = [x.strip().upper() for x in f.readlines()]
        if not limit:
            return lines # returns array of ALL the words in the .txt file (since limit = None by default)
        else:
            return lines[:limit] # returns an array of some words e.g. ['CIGAR', 'REBUT', 'SISSY']

class WordleEnvBase(gym.Env):
    """
    Actions:
        Can play any 5 letter word in vocabulary
        * 13k for full vocab
    State space is defined as:
        * 6 possibilities for turns (WORDLE_TURNS)
        * Each VALID_CHAR has a state of 0/1 for whether it's been guessed before
        * For each in VALID_CHARS [A-Z] can be in one of 3^WORDLE_N states: (No, Maybe, Yes)
        for full game, this is (3^5)^26
        Each state has 1 + 5*26 possibilities
    Reward:
        Reward is 10 for guessing the right word, -10 for not guessing the right word after 6 guesses.
    Starting State:
        Random goal word
        Initial state with turn 0, all chars Unvisited + Maybe
    """
    def __init__(self, words: List[str],
                 max_turns: int,
                 allowable_words: Optional[int] = None,
                 frequencies: Optional[List[float]]=None,
                 mask_based_state_updates: bool=False):
        assert all(len(w) == WORDLE_N for w in words), f'Not all words of length {WORDLE_N}, {words}' # This is a safety check to ensure each word in the words array is exactly 5 letters long. Note, f' is a formatted string (like template literals)
        self.words = words
        self.max_turns = max_turns
        self.allowable_words = allowable_words
        self.mask_based_state_updates = mask_based_state_updates
        if not self.allowable_words:
            self.allowable_words = len(self.words) # allowable_words is an int representing the NUMBER of allowable words. By default, this is the size of the words array.

        self.frequencies = None # By default, frequencies is none.
        if frequencies:
            assert len(words) == len(frequencies), f'{len(words), len(frequencies)}'
            self.frequencies = np.array(frequencies, dtype=np.float32) / sum(frequencies) # self.frequencies is an array of frequencies matching up with the array of words.
            # e.g. if words is ['ali', 'bob', 'cat', 'dog'], then self.frequencies = [0.1, 0.3, 0.2, 0.4], meaning 'dog' is the most frequent word. Note, frequencies must sum to 1.

        self.action_space = spaces.Discrete(len(self.words)) # A user can choose any word in the list of words as an action
        self.observation_space = spaces.MultiDiscrete(state.get_nvec(self.max_turns))

        # DISCRETE IS ONE OF THE FUNDAMENTAL SPACES
        # https://gymnasium.farama.org/api/spaces/fundamental/#discrete
        # It's a space consisting of finitely many elements.
        # e.g. observation_space = Discrete(2) # {0, 1}
        #      observation_space.sample() # 0

        # MULTIDISCRETE IS ONE OF THE FUNDAMENTAL SPACES
        # https://gymnasium.farama.org/api/spaces/fundamental/#gymnasium.spaces.MultiDiscrete
        # A nintendo controller can be conceptualized as 3 discrete action spaces:
        # 1) Arrow Keys: 0 [None], 1 [Up], 2[Right], 3[Down], 4[Left] 
        # 2) Button A: 0 [None], 1 [Pressed] 
        # 3) Button B: 0 [None], 1 [Pressed] 
        # You can represent a given action as spaces.MultiDiscrete([ 5, 2, 2 ]). A sample might be array([3, 1, 0])

        self.done = True # This means the episode / game has ended
        self.goal_word: int = -1

        self.state: state.WordleState = None
        self.state_updater = state.update # self.state_updater stores a function which is either update() or update_mask()
        if self.mask_based_state_updates:
            self.state_updater = state.update_mask

    def step(self, action: int):
        if self.done:
            raise ValueError(
                "You are calling 'step()' even though this "
                "environment has already returned done = True. You "
                "should always call 'reset()' once you receive 'done = "
                "True' -- any further steps are undefined behavior."
            )
        self.state = self.state_updater(state=self.state,
                                        word=self.words[action],
                                        goal_word=self.words[self.goal_word]) # This is changing the state from s to s', based on the action chosen

        reward = 0 # By default, we have reward of 0; most of the player's guesses won't be the goal_word
        if action == self.goal_word:
            self.done = True
            #reward = REWARD
            if state.remaining_steps(self.state) == self.max_turns-1: # RHS is 6 - 1 = 5. LHS is remaining_steps() = 5, if step() is called for first time (before which, remaining_steps() = 6)
                reward = 0  # No reward for guessing off the bat
            else:
                #reward = REWARD*(self.state.remaining_steps() + 1) / self.max_turns # We care about efficiency
                reward = REWARD # We don't care about efficiency, as long as we get there in the end
        elif state.remaining_steps(self.state) == 0:
            self.done = True
            reward = -REWARD # -10 if we run out of guesses

        return self.state.copy(), reward, self.done, {"goal_id": self.goal_word} # return observation, reward, terminated, info # https://gymnasium.farama.org/api/env/#gymnasium.Env.step

    def reset(self, seed: Optional[int] = None): # At the end of an episode, create a new starting state S_0, set done to false, and set the goal_word randomly.
        self.state = state.new(self.max_turns)
        self.done = False
        self.goal_word = int(np.random.random()*self.allowable_words)

        return self.state.copy()

    def set_goal_word(self, goal_word: str): # Setting goal_word using a string
        self.goal_word = self.words.index(goal_word)

    def set_goal_id(self, goal_id: int): # Setting goal_word using an int
        self.goal_word = goal_id


class WordleEnv10(WordleEnvBase):
    def __init__(self):
        super().__init__(words=_load_words(10), max_turns=6)


class WordleEnv100(WordleEnvBase):
    def __init__(self):
        super().__init__(words=_load_words(100), max_turns=6)


class WordleEnv100OneAction(WordleEnvBase):
    def __init__(self):
        super().__init__(words=_load_words(100), allowable_words=1, max_turns=6)


class WordleEnv100WithMask(WordleEnvBase):
    def __init__(self):
        super().__init__(words=_load_words(100), max_turns=6,
                         mask_based_state_updates=True)


class WordleEnv100TwoAction(WordleEnvBase):
    def __init__(self):
        super().__init__(words=_load_words(100), allowable_words=2, max_turns=6)


class WordleEnv100FullAction(WordleEnvBase):
    def __init__(self):
        super().__init__(words=_load_words(), allowable_words=100, max_turns=6)


class WordleEnv1000(WordleEnvBase):
    def __init__(self):
        super().__init__(words=_load_words(1000), max_turns=6)


class WordleEnv1000WithMask(WordleEnvBase):
    def __init__(self):
        super().__init__(words=_load_words(1000), max_turns=6,
                         mask_based_state_updates=True)


class WordleEnv1000FullAction(WordleEnvBase):
    def __init__(self):
        super().__init__(words=_load_words(), allowable_words=1000, max_turns=6)


class WordleEnvFull(WordleEnvBase):
    def __init__(self):
        super().__init__(words=_load_words(), max_turns=6)


class WordleEnvReal(WordleEnvBase):
    def __init__(self):
        super().__init__(words=_load_words(), allowable_words=2315, max_turns=6)


class WordleEnvRealWithMask(WordleEnvBase):
    def __init__(self):
        super().__init__(words=_load_words(), allowable_words=2315, max_turns=6,
                         mask_based_state_updates=True)
