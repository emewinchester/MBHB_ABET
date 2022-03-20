import numpy as np
from pkg.constants import MAX_CAPACITY

def greedy(initial_state):
    return np.round(initial_state / initial_state.sum() * MAX_CAPACITY)