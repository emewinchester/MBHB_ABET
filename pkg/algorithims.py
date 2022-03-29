import numpy as np
from pkg.neighborhood import Neighborhood
from pkg.constants import *
from pkg.utils import generate_random_solution
from pkg.tabu_list import *



def greedy(initial_state, evaluation):

    solution = np.round(initial_state / initial_state.sum() * MAX_CAPACITY)
    solution_cost = evaluation.evaluate(solution)

    return solution, solution_cost




def random_search(evaluation):

    initial_solution = generate_random_solution()
    current_solution = initial_solution
    best_solution = current_solution

    best_solution_cost = evaluation.evaluate(best_solution)

    for i in range(BA_TOTAL_ITERATIONS):
        current_solution = generate_random_solution()
        current_solution_cost = evaluation.evaluate(current_solution)

        if current_solution_cost < best_solution_cost:
            best_solution      = current_solution
            best_solution_cost = current_solution_cost

    return best_solution, best_solution_cost





def local_search(granularity, evaluation):

    current_solution = generate_random_solution()
    current_solution_cost = evaluation.evaluate(current_solution)

    evaluations = 0


    while True and evaluations < BL_EVALLUATION_CALLS:


        neigbhorhood = Neighborhood(
            current_solution = current_solution, 
            slots            = granularity
        )


        neighbor = neigbhorhood.get_neighbor()


        while neighbor is not None:
            neighbor_cost = evaluation.evaluate(neighbor)
            evaluations += 1

            if neighbor_cost < current_solution_cost:
                current_solution      = neighbor
                current_solution_cost = neighbor_cost
                break
            else:
                neighbor = neigbhorhood.get_neighbor()

            if evaluations > BL_EVALLUATION_CALLS:
                break

        if neighbor is None:
            break

    return current_solution, current_solution_cost, evaluations




def simulated_an(t0,L,tf,slots,evaluation):

    t = t0

    current_solution      = generate_random_solution()
    current_solution_cost = evaluation.evaluate(current_solution)

    neighbors = Neighborhood(
        current_solution = current_solution,
        slots            = slots
    )

    k = 0

    while t >= tf:
        for cont in range(L):

            candidate_solution      = neighbors.get_neighbor()
            candidate_solution_cost = evaluation.evaluate(candidate_solution)

            cost_difference = candidate_solution_cost - current_solution_cost

            if cost_difference < 0 or \
                np.random.rand() < np.exp( (-cost_difference) / t ):

                current_solution      = candidate_solution
                current_solution_cost = candidate_solution_cost 

                neighbors = Neighborhood(
                    current_solution = current_solution,
                    slots            = slots
                )
            
        t = t0 / (1 + k)
        k += 1

    return current_solution, current_solution_cost



def tabu_search(tenure, reboots, total_iterations, slots, total_neighbors, evaluation):
    
    current_solution      = generate_random_solution()
    current_solution_cost = evaluation.evaluate(current_solution)

    global_solution      = current_solution
    global_solution_cost = current_solution_cost


    # Matriz de frecuencias: inicializacion
    # frequency = np.ones((MAX_CAPACITY+1,TOTAL_STATIONS))

    # Matriz frecuencias: actualizacion
    # frequency = update_frecuency(frequency, current_solution)

    # tabu list
    tl = TabuList(tenure)




    for reboot in range(reboots):

        for iteration in range(int(total_iterations/reboots)):

            # genero vecinos -> 40 vecinos

            neighbors = Neighborhood(current_solution, slots)
            candidates = \
                [ neighbors.get_neighbor_ts() for n in range(total_neighbors) ]

            # ( solucion, movimiento, coste, esta_en_tabu_list )
            candidates_cost = [ 
                (c[0], c[1], evaluation.evaluate(c[0]), c[1] in tl.tabu_list )
                for c in candidates 
            ]

            candidates_cost.sort(key= lambda x:x[2])

            # se cumple criterio de aspiracion -> mejorar la global siendo tabu
            if candidates_cost[0][2] < global_solution_cost:

                # aceptamos solucion
                current_solution      = candidates_cost[0][0]
                current_solution_cost = candidates_cost[0][2]

                # actualizamos global
                global_solution      = current_solution
                global_solution_cost = current_solution_cost

                # actualizamos tabu
                tl.add_movement(candidates_cost[0][1])

            else:

                # eliminamos los movimientos tabu-activos
                candidates_clean = [c for c in candidates_cost if not c[3]]

                # aceptamos solucion
                current_solution      = candidates_clean[0][0]
                current_solution_cost = candidates_clean[0][2]

                # actualizamos tabu
                tl.add_movement(candidates_clean[0][1])


            # actualizamos matriz de frecuencias
            # frequency = update_frecuency(frequency, current_solution) 

        

        n = np.random.rand()

        if   n < RANDOM_SOL_PROB:
            current_solution      = generate_random_solution()
            current_solution_cost = evaluation.evaluate(current_solution)
        
        elif n < GREEDY_SOL_PROB:
            current_solution      = generate_random_solution()
            current_solution_cost = evaluation.evaluate(current_solution)
        
        else:
            current_solution      = global_solution
            current_solution_cost = global_solution_cost

        # Redimensionamos la tabu-list
        if np.random.rand() < 0.5:
            tenure += int(0.5 * tenure) # incrementamos la lista

        else:
            tenure -= int(0.5 * tenure) # decrementamos la lista

        # No admitimos listas menores a 2 elementos
        if tenure < 2:
            tl = TabuList(2)
        else:
            tl = TabuList(tenure)


    return global_solution, global_solution_cost
            


