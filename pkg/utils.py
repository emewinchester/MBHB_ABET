import numpy as np
from pkg.constants import MAX_CAPACITY, BA_TOTAL_ITERATIONS, SEEDS



def generate_random_solution(vector, limit_capacity):
    """
    Generates random solution from a vector. 
    """

    capacity = vector.copy()

    capacity = np.round(capacity * limit_capacity / np.sum(capacity))
    total_gaps = np.sum(capacity)

    if total_gaps > limit_capacity:
        greater_station = np.argmax(capacity)
        capacity[greater_station] -= total_gaps - limit_capacity
    
    return capacity





def generate_capacities_for_RS(sequence, stations):


    # cada fila de capacities es una posible solucion de la BA
    capacities = np.empty((SEEDS, BA_TOTAL_ITERATIONS, stations))

    # Relleneamos capacities con soluciones viables


    for seed_index in range(SEEDS):

        initial, leap = 0, stations

        for i in range(BA_TOTAL_ITERATIONS):
    
            vector = sequence[seed_index,initial:(initial+leap)].copy()
            initial +=leap
            

            capacities[seed_index,i,:] = generate_random_solution(
                vector         = vector,
                limit_capacity = MAX_CAPACITY
            )

    
    return capacities









