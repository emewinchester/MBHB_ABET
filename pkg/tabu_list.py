import numpy as np

class TabuList():

    def __init__(self, tenure):
        self.tenure    = tenure
        self.tabu_list = []
        self.cont      = 0
    

    def add_movement(self, movement):
        if len(self.tabu_list) < self.tenure:
            self.tabu_list.append(movement)
        else:
            self.tabu_list[self.cont % self.tenure] = movement
            self.cont += 1


    def reset(self):
        self.tabu_list = []
        self.cont = 0
    
    
    



