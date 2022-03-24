from pkg.neighborhood import Neighborhood
from pkg.constants import *
from pkg.utils import generate_random_solution

def local_search(slots_to_move, evaluation):

    current_solution = generate_random_solution()
    cost_current_solution = evaluation.evaluate(current_solution)

    evaluations = 0


    while True and evaluations < BL_EVALLUATION_CALLS:


        neigbhorhood = Neighborhood(
            current_solution = current_solution, 
            slots            = slots_to_move
        )


        neighbor = neigbhorhood.get_neighbor()


        while neighbor is not None:
            cost_neighbor = evaluation.evaluate(neighbor)
            evaluations += 1

            if cost_neighbor < cost_current_solution:
                current_solution      = neighbor
                cost_current_solution = cost_neighbor
                break
            else:
                neighbor = neigbhorhood.get_neighbor()


        if neighbor is None:
            break

    return current_solution



  