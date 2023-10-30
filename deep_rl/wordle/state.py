"""
Keep the state in a 1D int array

index[0] = remaining steps
Rest of data is laid out as binary array

[1..27] = whether char has been guessed or not

[[status, status, status, status, status]
 for _ in "ABCD..."]
where status has codes
 [1, 0, 0] - char is definitely not in this spot
 [0, 1, 0] - char is maybe in this spot
 [0, 0, 1] - char is definitely in this spot
"""
import collections # Use the various data structures / utility functions provided by the collections module. e.g. OrderedDict, Deque
from typing import List # typing is a module used for type hinting in Python. 
# The "List" type hint specifies that a variable should be a list containing elements of a specified type e.g. List[int]
import numpy as np

# Adding these consts instead of importing them
WORDLE_CHARS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
WORDLE_N = 5
# Commenting this out to avoid import errors
# from wordle.const import WORDLE_CHARS, WORDLE_N


WordleState = np.ndarray # (short for "n-dimensional array")


# Looks similar to the "new" function, except it's a normal list instead of a numpy array. This array looks like [6, 2, 2, 2...] and is used to define the multidiscrete observation space in wordle.py. First element is 6 becaues a state can have 6 possible values for this element, other numbers are 2 because a state can have take 2 possible values for this element (0 or 1).
def get_nvec(max_turns: int):
    return [max_turns] + [2] * len(WORDLE_CHARS) + [2] * 3 * WORDLE_N * len(WORDLE_CHARS)


def new(max_turns: int) -> WordleState: # Takes in an int, returns a (starting) WordleState which is a really long Numpy array. This is S_0
    return np.array(
        [max_turns] + [0] * len(WORDLE_CHARS) + [0, 1, 0] * WORDLE_N * len(WORDLE_CHARS),
        dtype=np.int32)
    # Refer to https://wandb.ai/andrewkho/wordle-solver/reports/Solving-Wordle-with-Reinforcement-Learning--VmlldzoxNTUzOTc4?galleryTag=gaming#state-and-action-representation
    # and refer to https://wandb.ai/andrewkho/wordle-solver/reports/Solving-Wordle-with-Reinforcement-Learning--VmlldzoxNTUzOTc4?galleryTag=gaming#reinforcement-learning 
    # This numpy array is [max_turns, 0, 0... (repeats for a total of 26 0s),   0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0... (this sequence repeats 26 times) ]
    # Breaking this down, a state contains info about 
    # 1) How many turns we have left: 1 ELEMENT, WHOSE VALUE RANGES FROM 6 TO 0 (I think).
    # 2) Whether each letter from a - z is attempted or not: 26 ELEMENTS, WHOSE VALUES RANGE FROM 0 to 1.
    # 3) If a letter has been attempted, then for each of the 5 slots, is it a) NOT IN THE SLOT, b) MAYBE IN THE SLOT or c) IN THE SLOT? 3 * 5 * 26 = 390 ELEMENTS, WHOSE VALUES RANGE FROM 0 TO 1.
    # 3 example) E.g. if target word is HELLO and we guess HANKY, then for the letter 'H', in the 1st slot, we have 0, 0, 1 since 'H' is defo in the 1st slot. 
    # 3 example) Meanwhile for the letter 'H', in the 2nd, 3rd, 4th and 5th slots, we have 0, 1, 0, since the letter 'H' may be in the remaining 4 slots.
    # 3 example) Meanwhile for the letter 'A', in the 2nd slot, we have 1, 0, 0 since the letter 'A' is definitely not in the 2nd slot.


# Returns the remaining steps left in the game (for a given wordle state)
def remaining_steps(state: WordleState) -> int:
    return state[0] # Accesses the 1st element of the long WordleState array (this element is max_turns to begin with)


NO = 0
SOMEWHERE = 1
YES = 2

# I'm not reading the details, but I assume this is how the environment updates its state from s to s' using only the mask data (can't use the goal word as we're not supposed to know that!). 
def update_from_mask(state: WordleState, word: str, mask: List[int]) -> WordleState:
    """
    return a copy of state that has been updated to new state

    From a mask we need slighty different logic since we don't know the
    goal word.

    :param state:
    :param word:
    :param goal_word:
    :return:
    """
    state = state.copy()

    prior_yes = []
    prior_maybe = []
    # We need two passes because first pass sets definitely yesses
    # second pass sets the no's for those who aren't already yes
    state[0] -= 1 # This is most obvious- we decrement the element representing the remaining steps we have
    for i, c in enumerate(word):
        cint = ord(c) - ord(WORDLE_CHARS[0])
        offset = 1 + len(WORDLE_CHARS) + cint * WORDLE_N * 3
        state[1 + cint] = 1
        if mask[i] == YES:
            prior_yes.append(c)
            # char at position i = yes, all other chars at position i == no
            state[offset + 3 * i:offset + 3 * i + 3] = [0, 0, 1]
            for ocint in range(len(WORDLE_CHARS)):
                if ocint != cint:
                    oc_offset = 1 + len(WORDLE_CHARS) + ocint * WORDLE_N * 3
                    state[oc_offset + 3 * i:oc_offset + 3 * i + 3] = [1, 0, 0]

    for i, c in enumerate(word):
        cint = ord(c) - ord(WORDLE_CHARS[0])
        offset = 1 + len(WORDLE_CHARS) + cint * WORDLE_N * 3
        if mask[i] == SOMEWHERE:
            prior_maybe.append(c)
            # Char at position i = no, other chars stay as they are
            state[offset + 3 * i:offset + 3 * i + 3] = [1, 0, 0]
        elif mask[i] == NO:
            # Need to check this first in case there's prior maybe + yes
            if c in prior_maybe:
                # Then the maybe could be anywhere except here
                state[offset+3*i:offset+3*i+3] = [1, 0, 0]
            elif c in prior_yes:
                # No maybe, definitely a yes, so it's zero everywhere except the yesses
                for j in range(WORDLE_N):
                    # Only flip no if previously was maybe
                    if state[offset + 3 * j:offset + 3 * j + 3][1] == 1:
                        state[offset + 3 * j:offset + 3 * j + 3] = [1, 0, 0]
            else:
                # Just straight up no
                state[offset:offset+3*WORDLE_N] = [1, 0, 0]*WORDLE_N

    return state

# TLDR: Returns a list which represents the colouring of our guess word w.r.t the target word (0 is grey, 1 is yellow, 2 is green)
def get_mask(word: str, goal_word: str) -> List[int]:
    
    mask = [0, 0, 0, 0, 0] # think of this mask as a green/yellow/grey colouring of our guess word. Green is 2, yellow is 1, grey is 0.
    counts = collections.Counter(goal_word) # collections.Counter("hello") returns a Counter obj (essentially a dictionary) where keys are letters and values are counts: {'h': 1, 'e': 1, 'l': 2, 'o': 1}
    
    # Definite yesses first
    for i, c in enumerate(word): # i represents the index of the character in the guess word, starting from 0 for the first character. c represents the character at the current index in the guess word.
        if goal_word[i] == c: # if one of our guess letters is in the right spot (green)
            mask[i] = 2 # set the value for the corresponding slot in the mask to 2 (meaning green).
            counts[c] -= 1 # If we got a green letter, update our counter object.

    for i, c in enumerate(word):
        if mask[i] == 2:
            continue # we already took care of the definite yesses in the previous loop
        elif c in counts: # i.e. if this char of our guess word is in our goal word
            if counts[c] > 0: # if this char of our guess word could be in the goal word (but in a different position)
                mask[i] = 1 # color that char yellow
                counts[c] -= 1 # If we got a yellow letter, update our counter object.
            else: # if this char of our guess word (e.g. the 3rd 'E' in 'EEEAB') must be coloured grey because there are no more unaccounted Es in the goal word (e.g. 'JEWEL') because we've shaded all relevant chars (e.g. shaded the 1st and 2nd 'E's in 'EEEAB' yellow and green, respectively)
                for j in range(i+1, len(mask)): # Then colour the 4th and 5th letters in our guess word ('A' and 'B') grey, if they were not already green. It sounds weird colouring 'A' and 'B' grey when theyre unrelated to 'E', but it's fine as we'll just colour them differently (if we need to) once we reach them in the enumerate(word) for-loop
                    if mask[j] == 2:
                        continue
                    mask[j] = 0 # TBH I dont see why you need to do this, since the mask is initialized with all 0s. But i'm not questioning it.
    return mask

# When our agent from its current "state", picks a "word" as an action, our environment gets a colouring of the "word" by comparing it to the "goal_word". 
# This colouring is called a "mask". The current "state" and the "mask" + "word" are used by the environment to define the next state, s' for the agent.
# This function returns that s'.
def update_mask(state: WordleState, word: str, goal_word: str) -> WordleState:
    """
    return a copy of state that has been updated to new state

    :param state:
    :param word:
    :param goal_word:
    :return:
    """
    mask = get_mask(word, goal_word)
    return update_from_mask(state, word, mask)

# I haven't examined code in detail so idk how this differs from update_mask apart from the fact that the latter uses the mask to update the state. 
# Honestly it seems very similar. I think for a given input, update() will always return the same thing as update_mask()?
def update(state: WordleState, word: str, goal_word: str) -> WordleState:
    state = state.copy()

    state[0] -= 1
    for i, c in enumerate(word):
        cint = ord(c) - ord(WORDLE_CHARS[0]) # cint is a number from 0 - 25. If c is 'a', cint is 0. If c is 'b', cint is 1, etc.    ASIDE: # ord('c') returns the Unicode code point of the character 'c' e.g. 99.
        offset = 1 + len(WORDLE_CHARS) + cint * WORDLE_N * 3
        state[1 + cint] = 1
        if goal_word[i] == c:
            # char at position i = yes, all other chars at position i == no
            state[offset + 3 * i:offset + 3 * i + 3] = [0, 0, 1]
            for ocint in range(len(WORDLE_CHARS)):
                if ocint != cint:
                    oc_offset = 1 + len(WORDLE_CHARS) + ocint * WORDLE_N * 3
                    state[oc_offset + 3 * i:oc_offset + 3 * i + 3] = [1, 0, 0]
        elif c in goal_word:
            # Char at position i = no, other chars stay as they are
            state[offset + 3 * i:offset + 3 * i + 3] = [1, 0, 0]
        else:
            # Char at all positions = no
            state[offset:offset + 3 * WORDLE_N] = [1, 0, 0] * WORDLE_N

    return state

# I tested get_mask out on today's wordle, and the masks look as expected.
# print(get_mask(word= "CRATE",goal_word="TEMPO")) # [0, 0, 0, 1, 1]
# print(get_mask(word= "TEPID",goal_word="TEMPO")) # [2, 2, 1, 0, 0]
# print(get_mask(word= "TEMPO",goal_word="TEMPO")) # [2, 2, 2, 2, 2]