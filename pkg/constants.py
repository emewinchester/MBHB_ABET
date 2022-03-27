CERCANAS_INDICES_PATH = './00_datos/cercanas_indices.csv'
CERCANAS_KMS_PATH     = './00_datos/cercanas_kms.csv'
DELTAS_5M_PATH        = './00_datos/deltas_5m.csv'

MAX_CAPACITY = 220
MIN_CAPACITY = 205

# Los km extra andando penalizan el triple
CYCLING_WEIGHT = 1
WALKING_WEIGHT = 3 * CYCLING_WEIGHT

BA_TOTAL_ITERATIONS  = 100
BL_EVALLUATION_CALLS = 3000

SEEDS = 5

TOTAL_STATIONS = 16

# rango de probabilidades para la reinicializacion en Tabu Search

RANDOM_SOL_PROB = 0.25
GREEDY_SOL_PROB = RANDOM_SOL_PROB + 0.5


