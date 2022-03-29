import numpy as np
from pkg.constants import *



def generate_random_solution():

    capacity = np.random.randint(2, 10, TOTAL_STATIONS)
    total_gaps = np.sum(capacity)

    capacity = np.round(capacity * MAX_CAPACITY / total_gaps)
    
    if total_gaps > MAX_CAPACITY:
        greater_station = np.argmax(capacity)
        capacity[greater_station] -= total_gaps - MAX_CAPACITY
    
    return capacity


def update_frequency_matrix(matrix, vector, base_values):

    for column in range(len(vector)):
        value = vector[column]
        row = len(base_values[base_values < value]) - 1

        matrix[row, column] +=1

    return matrix



