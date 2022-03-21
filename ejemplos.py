
from itertools import permutations
from time import time
import numpy as np

# ejemplo de itertools
vector = [0,1,2,3,4]
comb = permutations(vector,2)
for origen, destino in comb:
    print(f'{origen}, {destino}')

print(time())

print(range(3))


print('*' * 10)


a = np.array([1,2,3,4])
print(np.argmax(a))