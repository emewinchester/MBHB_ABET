import numpy as np
from random        import shuffle
from pkg.constants import PERMUTATIONS


class Neighborhood():

    def __init__(self, current_solution, slots):
        self.solution = current_solution.copy()
        self.slots = slots
        
        # list of pemutations
        perm = PERMUTATIONS
        self.neighbors = perm.copy()
        np.random.shuffle(self.neighbors)



    def _adjust_neighbor(self, neighbor, origin, destiny):

        # movimiento (origen,destino)
        # coge slots de origen y los añade a destino
        if neighbor[origin] < self.slots:
            neighbor[destiny] += (self.slots - neighbor[origin])
            neighbor[origin]  = 0
        else:
            neighbor[destiny] += self.slots
            neighbor[origin]  -= self.slots
        
        return neighbor



    
    def get_neighbor(self):

        if len(self.neighbors) > 0:

            neighbor = self.solution.copy()

            o, d = self.neighbors[0]
            self.neighbors.remove((o,d))

            return self._adjust_neighbor(neighbor, o,d)

        else:
            return None

    
    def get_neighbor_ts(self):

        if len(self.neighbors) > 0:

            neighbor = self.solution.copy()

            o, d = self.neighbors[0]
            self.neighbors.remove((o,d))

            return self._adjust_neighbor(neighbor, o,d), (o,d)

        else:
            return None









