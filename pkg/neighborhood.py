from itertools import permutations


class Neighborhood():

    def __init__(self, current_solution, slots):
        self.solution = current_solution.copy()
        self.slots = slots
        
        # list of pemutations
        perm = permutations(range(len(self.solution)), 2)
        self.neighbors = list(perm)

    
    def get_neighbor(self):

        if len(self.neighbors) > 0:
            neighbor = self.solution.copy()

            origin, destiny = self.neighbors[0]
            self.neighbors.remove((origin,destiny))

            # movimiento (origen,destino)
            # coge slots de origen y los añade a destino
            if neighbor[origin] < self.slots:
                neighbor[destiny] += (self.slots - neighbor[origin])
                neighbor[origin]  = 0
            else:
                neighbor[destiny] += self.slots
                neighbor[origin]  -= self.slots

            return neighbor

        else:
            return None

        
