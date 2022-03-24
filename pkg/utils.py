import numpy as np
from pkg.constants import *



def generate_random_solution():

    capacity = np.random.randint(2, 10, TOTAL_STATIONS)

    capacity = np.round(capacity * MAX_CAPACITY / np.sum(capacity))
    total_gaps = np.sum(capacity)

    if total_gaps > MAX_CAPACITY:
        greater_station = np.argmax(capacity)
        capacity[greater_station] -= total_gaps - MAX_CAPACITY
    
    return capacity



