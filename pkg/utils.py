import numpy as np
from pkg.constants import *



def generate_random_solution():

    capacity = np.random.randint(2, 10, TOTAL_STATIONS)
    total_gaps = np.sum(capacity)

    capacity = np.round(capacity * MAX_CAPACITY / total_gaps)
    total_gaps = np.sum(capacity)
    
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


def generate_greedy_solution(matrix, base_values):
    #####  SOLUCION GREEDY #####

    solution = np.empty(matrix.shape[1])

    inverse = 1/matrix

   
    total_per_column = inverse.sum(axis=0)

    for column in range(len(solution)):

        # normalizamos la columna
        inverse[:,column] = inverse[:,column] / total_per_column[ column ]

        num = np.random.rand()
        suma = 0

        for row in range(matrix.shape[0]):
            suma += inverse[row,column]

            if num < suma:
                solution[column] = base_values[row]
                break


    # añadimos de 0 a 2 slots más a cada estacion
    increase  = np.random.randint(0, 3, len(solution))
    solution += increase

    # normalizamos solucion para evitar que supere los 220 slots
    total_slots = np.sum(solution)
    solution = np.round(solution * MAX_CAPACITY / total_slots)
    total_slots = np.sum(solution)

    if total_slots > MAX_CAPACITY:
        greater_station = np.argmax(solution)
        solution[greater_station] -= total_slots - MAX_CAPACITY

    return solution



