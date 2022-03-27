import numpy as np
from pkg.neighborhood import Neighborhood
from pkg.constants import *
from pkg.utils import generate_random_solution



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





def local_search(slots_to_move, evaluation):

    current_solution = generate_random_solution()
    current_solution_cost = evaluation.evaluate(current_solution)

    evaluations = 0


    while True and evaluations < BL_EVALLUATION_CALLS:


        neigbhorhood = Neighborhood(
            current_solution = current_solution, 
            slots            = slots_to_move
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


        if neighbor is None:
            break

    return current_solution, current_solution_cost



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

            candidate_solution      = neighbors.get_neighbor_sa()
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