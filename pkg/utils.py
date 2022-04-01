from operator import index
import numpy as np
import pandas as pd
from pkg.constants import *
import time



def generate_random_solution():
    """
    Generates a valid random solution for the problem. Total capacity between
    MIN_CAPACITY and MAX_CAPACITY, constants from constants.py
    """

    solution   = np.random.randint(2, 10, TOTAL_STATIONS)
    total_gaps = np.sum(solution)

    solution = np.int64( np.round(solution * MAX_CAPACITY / total_gaps) )
    total_gaps = np.sum(solution)
    
    # Readjustment of capacity in case it exceeds MAX_CAPACITY
    if total_gaps > MAX_CAPACITY:
        greater_station = np.argmax(solution)
        solution[greater_station] -= total_gaps - MAX_CAPACITY
    
    return solution


def update_frequency_matrix(matrix, vector, base_values):

    for column in range(len(vector)):
        value = vector[column]
        row = len(base_values[base_values < value]) - 1

        matrix[row, column] +=1

    return matrix


def generate_greedy_solution(matrix, base_values):
    #####  SOLUCION GREEDY #####

    # vector solucion de capacidades
    solution = np.empty(matrix.shape[1])

    # inversa de la matriz de frecuencias
    inverse = 1/matrix


    incremento = np.random.randint(0, 3, matrix.shape[1])
    slots_repartidos = 0

    index_order = list(range(matrix.shape[1]))
    np.random.shuffle(index_order)

    

    for column in index_order:

        # desactivamos buckets si es necesario
        slots_por_repartir = MAX_CAPACITY - slots_repartidos

        # vector base_values + 2 -> para comparar con el mayor elemento del rango
        bv_2 = base_values + 2
        buckets_disponibles = len( bv_2[bv_2 < slots_por_repartir] )

        # normalizamos la columna
        total_column = inverse[0:buckets_disponibles,column].sum()
        inverse[0:buckets_disponibles,column] /= total_column

        num = np.random.rand()
        suma = 0

        for row in range(buckets_disponibles):
            suma += inverse[row,column]

            if num < suma:
                solution[column] = base_values[row] + incremento[column]
                slots_repartidos += solution[column]
                
                break


    # normalizamos solucion SI NOS QUEDAMOS POR DEBAJO DEL MINIMO
    if slots_repartidos < MIN_CAPACITY:
        total_slots = np.sum(solution)
        solution = np.int64(np.round(solution * MAX_CAPACITY / total_slots))
        total_slots = np.sum(solution)

        if total_slots > MAX_CAPACITY:
            greater_station = np.argmax(solution)
            solution[greater_station] -= total_slots - MAX_CAPACITY

    return solution



