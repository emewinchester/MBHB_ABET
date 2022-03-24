from pkg.constants import *
from pkg.utils import generate_random_solution


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

    return best_solution

