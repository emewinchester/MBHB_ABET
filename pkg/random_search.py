from pkg.constants import *


def random_search(capacities, evaluation):

    initial_solution = capacities[0,:]
    current_solution = initial_solution
    best_solution = current_solution

    best_solution_cost = evaluation.evaluate(best_solution)

    for iteration in range(1,BA_TOTAL_ITERATIONS):
        current_solution = capacities[iteration,:]
        current_solution_cost = evaluation.evaluate(current_solution)

        if current_solution_cost < best_solution_cost:
            best_solution      = current_solution
            best_solution_cost = current_solution_cost

    return best_solution

