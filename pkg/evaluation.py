import pandas as pd
import numpy as np
from pkg.constants import *


class Evaluation():
    """
    clase que evalua una solucion dada
    """
    def __init__(self):

        # DATA LOADING
        index_df    = pd.read_csv(CERCANAS_INDICES_PATH)
        distance_df = pd.read_csv(CERCANAS_KMS_PATH)
        deltas_df   = pd.read_csv(DELTAS_5M_PATH) 

        # DATA PREPROCESSING
        self.index_matrix    = index_df.to_numpy()
        self.distance_matrix = distance_df.to_numpy()
        deltas_matrix   = deltas_df.to_numpy()
        deltas_matrix[1:,:] *= 2

        # Add row of zeros to count initial state as a movement
        zero_row = np.zeros((1, deltas_matrix.shape[1]))
        self.extended_deltas = np.vstack([zero_row, deltas_matrix.copy()])

        # Matrix of movements, includes the initial state as movement
        self.movements = self.extended_deltas[1:,:]
        

    
    def _get_jumps(self,station, movement, jump_sequence, bikes, capacity):
        """
        DESCROPCION DE LA FUNCION
        """

        jumps = np.zeros(TOTAL_STATIONS)


        if movement < 0: # coger bicis

            for i in range(TOTAL_STATIONS):
                
                bikes[station] += movement

                if bikes[station] >= 0:
                    jumps[i] = movement
                    break
                else:
                    jumps[i] = movement - bikes[station]
                    movement = bikes[station]
                    bikes[station] = 0

                    # actualizamos la estacion siguiente
                    if i+1 < len(jump_sequence): 
                        station = jump_sequence[i+1]

        if movement > 0:

            for i in range(TOTAL_STATIONS):

                bikes[station] += movement

                if bikes[station] <= capacity[station]:
                    jumps[i] = movement
                    break
                else:
                    jumps[i] = capacity[station] - (bikes[station] - movement)
                    movement -= jumps[i]
                    bikes[station] = capacity[station]

                    if i+1 < len(jump_sequence): 
                        station = jump_sequence[i+1]


        return jumps



    def _get_timestamp_distance(self, bikes, capacity, current_time, index_matrix,count_km):

        jump_matrix = None


        for station in range(TOTAL_STATIONS):

            movement = current_time[station]
            jump_sequence = index_matrix[station,:]
            
            jumps = self._get_jumps(
                station       = station,
                movement      = movement,
                jump_sequence = jump_sequence,
                bikes         = bikes,
                capacity      = capacity
            )


            if jump_matrix is None:
                jump_matrix = jumps
            else:
                jump_matrix = np.vstack((jump_matrix,jumps))

        if count_km:
            # ya tenemos la matriz 16x16

            # aplicamos las ponderaciones
            jump_matrix[jump_matrix < 0] *= WALKING_WEIGHT
            jump_matrix[jump_matrix > 0] *= CYCLING_WEIGHT

            # pasamos la matriz a valor absoluto
            jump_matrix = np.abs(jump_matrix)

            # calculamos distancias
            return (jump_matrix * self.distance_matrix).sum()

        else:
            return 0

    

    def evaluate(self, solution):

        total_distance = 0
        bikes = self.extended_deltas[0,:].copy()
        count_km = False

        for row in range(self.movements.shape[0]):

            # flujo en una franja de tiempo de 5 min 
            current_time = self.movements[row,:] 

            total_distance += self._get_timestamp_distance(
                bikes        = bikes,
                capacity     = solution,
                current_time = current_time,
                index_matrix = self.index_matrix,
                count_km     = count_km
            )

            # First movement doesn't count km
            count_km = True

        return total_distance
