import numpy as np
import pandas as pd

from pkg.neighborhood import Neighborhood
from pkg.constants import *
from pkg.utils import *
from pkg.tabu_list import *



def greedy(initial_state, evaluation):
    """
    Greedy Algorithm 

    Parameters
    ----------
    initial_state: array of elements from which the greedy solution is calculated. 

    evaluation: Evaluation object. Contains the data loaded and the evaluation function

    Returns
    -------
    solution: array of elements that represents the capacity of the stations.

    solution_cost: cost of the solution
    """

    solution = \
        np.int64(np.round(initial_state / initial_state.sum() * MAX_CAPACITY))
    solution_cost = evaluation.evaluate(solution)

    return solution, solution_cost




def random_search(evaluation):
    """
    Function that implements the Random Search Algorithm

    Parameters
    ---------
    evaluation: Evaluation object. Contains the data loaded and the evaluation function.

    Returns
    -------
    best_solution: array of elements that represents the capacity of the stations.

    best_solution_cost: cost of best_solution.
    """

    initial_solution = generate_random_solution()
    current_solution = initial_solution
    best_solution    = current_solution

    best_solution_cost = evaluation.evaluate(best_solution)

    for i in range(RS_TOTAL_ITERATIONS - 1):
        current_solution      = generate_random_solution()
        current_solution_cost = evaluation.evaluate(current_solution)

        if current_solution_cost < best_solution_cost:
            best_solution      = current_solution
            best_solution_cost = current_solution_cost

    return best_solution, best_solution_cost





def local_search(granularity, evaluation, path=None):
    """
    Implements the Local Search Algorithm


    Parameters
    ----------
    granularity: Number of slots that can be moved between stations.

    evaluation: Evaluation object. Contains the data loaded and the evaluation function.

    path: Optional parameter. Name of the file in which the data is saved.


    Returns
    -------
    current_solution: array of elements that represents the capacity of the stations.

    current_solution_cost: cost of current_solution
    """

    # estructura de datos para volcar a csv
    costs = np.array([])

    current_solution = generate_random_solution()
    current_solution_cost = evaluation.evaluate(current_solution)

    evaluations = 0


    while True and evaluations < LS_EVALLUATION_CALLS:


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
                costs = np.append(costs, current_solution_cost)
                break
            else:
                neighbor = neigbhorhood.get_neighbor()

            if evaluations > LS_EVALLUATION_CALLS:
                break

        if neighbor is None:
            break
    

    if path is not None:
        df = pd.DataFrame({
            'Coste': costs
        })

        df.to_excel(path)

    return current_solution, current_solution_cost




def simulated_an(t0,L,tf,slots,evaluation):
    """
    Implementation of the Simulated Annealing Algorithm


    Parameters
    ----------
    t0: Initial temperature.

    L: Number of neighbors to generate

    tf: Final temperature.

    slots: granularity.

    evaluation: Evaluation object. Contains the data loaded and the evaluation function.


    Returns
    -------
    global_solution: array of elements that represents the capacity of the stations.

    global_solution_cost: cost of global_solution
    """

    t = t0

    current_solution      = generate_random_solution()
    current_solution_cost = evaluation.evaluate(current_solution)

    global_solution      = current_solution
    global_solution_cost = current_solution_cost

    neighbors = Neighborhood(
        current_solution = current_solution,
        slots            = slots
    )

    # number of iterations
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

                if current_solution_cost < global_solution_cost:
                    global_solution      = current_solution
                    global_solution_cost = current_solution_cost

                neighbors = Neighborhood(
                    current_solution = current_solution,
                    slots            = slots
                )
            
        t = t0 / (1 + k)
        k += 1

    return global_solution, global_solution_cost



def tabu_search(tenure, reboots, total_iterations, slots, total_neighbors, evaluation, path=None):
    """
    Implementation of Tabu Search Algorithm

    Parameters
    ----------
    tenure: Tabu List length.

    reboots: Times the algorithm is reseted.

    total_iterations: 
    slots:
    total_neighbors:
    evaluation: Evaluation object. Contains the data loaded and the evaluation function.
    """
    
    # data structure for the study of optimal number of iterations
    

    current_solution      = generate_random_solution()
    current_solution_cost = evaluation.evaluate(current_solution)

    global_solution      = current_solution
    global_solution_cost = current_solution_cost
    costs = np.array([global_solution_cost])
    iterations = np.array([0])


    # Matriz de frecuencias: inicializacion y actualizacion
    base_values = np.array(range(0,39,5))
    frequency = np.ones( (len( base_values ), TOTAL_STATIONS) )
    frequency = update_frequency_matrix(frequency,current_solution, base_values)

    # tabu list
    tl = TabuList(tenure)

    iteracion = 0

    for reboot in range(reboots):

        for i in range(int(total_iterations/reboots)):

            #print(f'reboot {reboot}, iteracion {i}')

            iteracion +=1

            # genero vecinos -> 40 vecinos

            # (solucion, (origen, lo_que_tenia_origen_antes_del_cambio )
            neighbors = Neighborhood(current_solution, slots)
            candidates = \
                [ neighbors.get_neighbor_ts() for n in range(total_neighbors) ]

            # ( solucion, movimiento, coste, esta_en_tabu_list )
            candidates_cost = [ 
                (c[0], c[1], evaluation.evaluate(c[0]), c[1] in tl.tabu_list )
                for c in candidates 
            ]

            
            candidates_cost.sort(key= lambda x:x[2])
            # print(candidates_cost)

            

            # se cumple criterio de aspiracion -> mejorar la global siendo tabu
            if candidates_cost[0][2] < global_solution_cost:

                #print('ASPIRA')
                #print(f'coste candidata {candidates_cost[0][2]}')
                # print(f'coste global {global_solution_cost}')

                # aceptamos solucion
                current_solution      = candidates_cost[0][0]
                current_solution_cost = candidates_cost[0][2]

                # actualizamos global
                global_solution      = current_solution
                global_solution_cost = current_solution_cost

                # almacenamos los datos
                costs = np.append(costs, global_solution_cost)
                iterations = np.append(iterations, iteracion)

                # actualizamos tabu
                tl.add_movement(candidates_cost[0][1])

                
            else:

                # print('No aspira')

                # eliminamos los movimientos tabu-activos
                candidates_clean = [c for c in candidates_cost if not c[3]]

                # print(f'veces tabu: {np.array(list(map(lambda x:x[3], candidates_cost))).sum()}')

                # segundo criterio de aspiracion: si la lista tabu elimina a todos los vecinos, 
                # candidatos, aceptamos al que mejor coste tenga
                if np.array(list(map(lambda x:x[3], candidates_cost))).sum() >= total_neighbors:
                    # aceptamos solucion
                    current_solution      = candidates_cost[0][0]
                    current_solution_cost = candidates_cost[0][2]

                    # actualizamos tabu
                    tl.add_movement(candidates_cost[0][1])
                else:
                    # aceptamos solucion
                    current_solution      = candidates_clean[0][0]
                    current_solution_cost = candidates_clean[0][2]

                    # actualizamos tabu
                    tl.add_movement(candidates_clean[0][1])

            
            
            frequency = update_frequency_matrix(frequency,current_solution, base_values)

        

        n = np.random.rand()

        if   n < RANDOM_SOL_PROB:
            # print('genera solucion aleatoria')
            current_solution      = generate_random_solution()
            current_solution_cost = evaluation.evaluate(current_solution)
        elif n < GREEDY_SOL_PROB:
            current_solution      = generate_greedy_solution(frequency, base_values)
            current_solution_cost = evaluation.evaluate(current_solution)   
            # print(f'genera solucion por matriz frecuencias, total bicis{current_solution.sum()}')     
        else:
            current_solution      = global_solution
            current_solution_cost = global_solution_cost
            # print('reboot a partir de la global')


        # actualizamos matriz de frecuencias
        frequency = update_frequency_matrix(frequency,current_solution, base_values)

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

    if path is not None:
        df = pd.DataFrame({
            'Iteracion' : iterations,
            'Coste'     : costs
        })

        df.to_excel(path)


    return global_solution, global_solution_cost
            


