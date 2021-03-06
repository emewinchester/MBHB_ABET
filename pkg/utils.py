import numpy as np
import pandas as pd
from sklearn import cross_decomposition
from pkg.constants import *




def generate_random_solution():
    """
    Generates a valid random solution for the problem. Total capacity between
    MIN_CAPACITY and MAX_CAPACITY, constants from constants.py
    """

    solution   = np.random.randint(2, 10, TOTAL_STATIONS)
    total_gaps = np.sum(solution)

    solution = np.int64( np.round(solution * MAX_CAPACITY / total_gaps) )
    total_gaps = np.sum(solution)
    
    # Readjustment of capacity in case it exceeds MAX_CAPACITY
    if total_gaps > MAX_CAPACITY:
        greater_station = np.argmax(solution)
        solution[greater_station] -= total_gaps - MAX_CAPACITY
    
    return solution


def update_frequency_matrix(matrix, vector, base_values):

    for column in range(len(vector)):
        value = vector[column]
        row = len(base_values[base_values < value]) - 1

        matrix[row, column] +=1

    return matrix


def generate_greedy_solution(matrix, base_values):
    #####  SOLUCION GREEDY #####

    # vector solucion de capacidades
    solution = np.empty(matrix.shape[1])

    # inversa de la matriz de frecuencias
    inverse = 1/matrix


    incremento = np.random.randint(0, 3, matrix.shape[1])
    slots_repartidos = 0

    index_order = list(range(matrix.shape[1]))
    np.random.shuffle(index_order)

    

    for column in index_order:

        # desactivamos buckets si es necesario
        slots_por_repartir = MAX_CAPACITY - slots_repartidos

        # vector base_values + 2 -> para comparar con el mayor elemento del rango
        bv_2 = base_values + 2
        buckets_disponibles = len( bv_2[bv_2 < slots_por_repartir] )

        # normalizamos la columna
        total_column = inverse[0:buckets_disponibles,column].sum()
        inverse[0:buckets_disponibles,column] /= total_column

        num = np.random.rand()
        suma = 0

        for row in range(buckets_disponibles):
            suma += inverse[row,column]

            if num < suma:
                solution[column] = base_values[row] + incremento[column]
                slots_repartidos += solution[column]
                
                break


    total_slots = np.sum(solution)
    print(f'slots asignados: {total_slots}')

    # normalizamos solucion SI NOS QUEDAMOS POR DEBAJO DEL MINIMO
    if total_slots < MIN_CAPACITY or total_slots > MAX_CAPACITY:

        print('normaliza')
        
        solution = np.int64(np.round(solution * MAX_CAPACITY / total_slots))
        total_slots = np.sum(solution)

        if total_slots > MAX_CAPACITY:
            greater_station = np.argmax(solution)
            solution[greater_station] -= total_slots - MAX_CAPACITY

    return solution



def neighbor_generation_operator(solution, k, sizes_Ek, granularity):

    neighbor = solution.copy()
    
    # select index from which we start selecting items
    index = np.random.randint(TOTAL_STATIONS)
    # print(f'indice: {index}')

    # size of the sublist
    s = sizes_Ek[k-1]
    # print(f'tama??o sublista: {s}')

    
    
    sublist = np.zeros(s)

    # items selection
    index_aux = index
    for i in range(s):
        sublist[i] = solution[index_aux % TOTAL_STATIONS]
        index_aux += 1
        
    # print(sublist)
    # np.random.shuffle(sublist)
    # print(sublist)

    for i in range(s):

        origin  = i
        destiny = i+1

        # solo en indices pares
        if origin%2 == 0:
                
            # movimiento (origen,destino)
            # coge granularity de slots de origen y los a??ade a destino
            if sublist[origin] < granularity:
                sublist[destiny] += sublist[origin]
                sublist[origin]  = 0
            else:
                sublist[destiny] += granularity
                sublist[origin]  -= granularity
        


    # modify neighbor
    index_aux = index
    for i in range(s):
        neighbor[index_aux % TOTAL_STATIONS] = sublist[i]
        index_aux += 1
    
    return neighbor



def inicializa_poblacion(tam, conocidos=None):

    if conocidos is None:
        cromosoma = generate_random_solution()
        poblacion = np.array([cromosoma])
    else:
        poblacion = conocidos

    for i in range(tam- poblacion.shape[0]):
        cromosoma = generate_random_solution()
        poblacion = np.append(poblacion,[cromosoma],0)  
    
    return poblacion



def evalua_poblacion(poblacion, evaluation, alpha = None):
    fitness_poblacion = np.empty(poblacion.shape[0])
    km_poblacion      = np.empty(poblacion.shape[0])
    slots_poblacion   = np.empty(poblacion.shape[0])

    for individuo in range(poblacion.shape[0]):
        if alpha is None:
            fitness, km, slots = evaluation.fitness(poblacion[individuo,:])
        else:
            fitness, km, slots = evaluation.fitness(poblacion[individuo,:], alpha)
        
        fitness_poblacion[individuo] = fitness
        km_poblacion[individuo]      = km
        slots_poblacion[individuo]   = slots
    
    return fitness_poblacion, km_poblacion, slots_poblacion



def operador_cruce(padres, ev, alpha):

    padre1 = padres[0,:].copy()
    padre2 = padres[1,:].copy()

    candidatos = np.empty((6, TOTAL_STATIONS))
    candidato  = np.zeros(TOTAL_STATIONS)
    
    for c in range(len(candidatos)):

        # nos aseguramos de tener hijos validos (m??s de 205 slots)
        while candidato.sum() < 205 :
            
            for i in range(TOTAL_STATIONS):

                if np.random.rand() < 0.5:
                    candidato[i] = padre1[i]
                else:
                    candidato[i] = padre2[i]

        candidatos[c,:] = candidato.copy()

        candidato = np.zeros(TOTAL_STATIONS)
        

    # Nos quedamos con los hijos con mejores fitness
    fitness, km, slots = evalua_poblacion(candidatos, ev, alpha)

    # nos quedamos solo con 2 hijos
    hijos         = np.empty((2,TOTAL_STATIONS))
    fitness_hijos = np.empty(2)
    km_hijos      = np.empty(2)

    for i in range(2):

        mejor = np.argmin(fitness)

        hijos[i,:]       = candidatos[mejor,:].copy()
        fitness_hijos[i] = fitness[mejor]
        km_hijos[i]      = km[mejor]

        # eliminamos ese candidato y su fitness
        fitness    = np.delete(fitness,mejor)
        candidatos = np.delete(candidatos, mejor, axis=0)
        km         = np.delete(km, mejor)

    
    return hijos, fitness_hijos, km_hijos




def seleccion_padres(poblacion):

    tam_poblacion = poblacion.shape[0]

    i_padre1 = np.random.randint(0,tam_poblacion)
    i_padre2 = np.random.randint(0,tam_poblacion)

    # T??cnica de diversidad en el cruce: 
    # un padre no puede cruzarse consigo mismo
    while(i_padre1 == i_padre2):
        i_padre2 = np.random.randint(0,tam_poblacion)


    padre1 = poblacion[i_padre1,:].copy()
    padre2 = poblacion[i_padre2,:].copy()

    padres = np.array([padre1,padre2])

    return padres


def operador_movimiento(cromosoma, origin, destiny, granularidad):

    # movimiento (origen,destino)
    # coge slots de origen y los a??ade a destino


    cromosoma = cromosoma.copy()

    if cromosoma[origin] < granularidad:
        cromosoma[destiny] += cromosoma[origin]
        cromosoma[origin]  = 0
    else:
        cromosoma[destiny] += granularidad
        cromosoma[origin]  -= granularidad

    return cromosoma


def muta_cromosoma(cromosoma, granularidad):

    origen = np.random.randint(0,TOTAL_STATIONS)
    destino = np.random.randint(0,TOTAL_STATIONS)

    while origen == destino:
        destino = np.random.randint(0,TOTAL_STATIONS)

    mutado = operador_movimiento(cromosoma, origen, destino, granularidad)

    return mutado


def mutacion(f_poblacion, hijos, f_hijos, granularidad):

    f_pob = f_poblacion.copy()
    f_pob_sorted = np.sort(f_pob) # ordenado ascendentemente

    # decidimos cuantos genes mutar por cada hijo
    for hijo in range(len(hijos)):


        if np.random.rand() > 0.05:

            if f_hijos[hijo] > f_pob_sorted[10]:

                # mutamos 3 genes
                for i in range(3):
                    hijos[hijo,:] = muta_cromosoma(hijos[hijo,:],granularidad)

            elif f_hijos[hijo] > f_pob_sorted[10]:

                # mutamos 2 genes
                for i in range(1):
                    hijos[hijo,:] = muta_cromosoma(hijos[hijo,:],granularidad)
            else:
                # mutamos 1 gen
                hijos[hijo,:] = muta_cromosoma(hijos[hijo,:],granularidad)
    
    return hijos



def reemplazo(k, poblacion, f_poblacion, hijos, f_hijos):

    
    participantes = int((poblacion.shape[0]*k))

    if participantes < 3:
        participantes = 3

    indices = list(range(poblacion.shape[0]))
    np.random.shuffle(indices)


    i_participantes = indices[:participantes]
    f_participantes = f_poblacion[i_participantes]

  

    # sustituimos el peor de los participantes
    peor_participantes = np.argmax(f_participantes)
    i_peor_poblacion = i_participantes[peor_participantes]

    # por el mejor de los hijos
    i_mejor_hijo = np.argmin(f_hijos)

    mejor_hijo         = hijos[i_mejor_hijo]
    mejor_hijo_fitness = f_hijos[i_mejor_hijo]
    

    
    f_poblacion[i_peor_poblacion] = mejor_hijo_fitness
    poblacion[i_peor_poblacion,:] = mejor_hijo.copy()

    return poblacion, f_poblacion



def distancia_hamming(padre1,padre2):

    if len(padre1) == len(padre2):
        
        distancia = 0

        for i in range(len(padre1)):

            if padre1[i] != padre2[i]:
                distancia +=1 
        
        return distancia
    
    else:
        return -1




def cruce_parent_centered(padre1, padre2):

    index_distintos = []

    for i in range(len(padre2)):
        # print(f'padre1, pos {i}: {padre1[i]}')
        # print(f'padre2, pos {i}: {padre2[i]}\n')

        if padre2[i] != padre1[i]: index_distintos.append(i)

    # print(index_distintos)

    # numero de cambios a realizar
    cambios = int(np.floor(len(index_distintos)/2))

    np.random.shuffle(index_distintos)

    # generamos 2 hijos
    hijo1 = np.copy(padre1)
    hijo2 = np.copy(padre2)

    for i in range(cambios):

        pos1 = index_distintos[i]
        pos2 = index_distintos[i+cambios]
        # print(pos)

        # mutamos hijos
        hijo1[pos1] = np.round(np.random.normal(loc=padre2[pos1], scale=2))
        hijo2[pos2] = np.round(np.random.normal(loc=padre1[pos2], scale=2))

        # corregimos si tenemos valores negativos
        if hijo1[pos1] < 0: hijo1[pos1] = 0
        if hijo2[pos2] < 0: hijo2[pos2] = 0

    # comprobamos que los hijos son soluciones validas (mayores de 205 bicis)
    if hijo1.sum() < 205: 
        # corregimos al hijo uniformemente
        multiplicador = 205/hijo1.sum()
        
        # redondeamos por arriba para asegurarnos no bajar de 205
        hijo1 = np.ceil(hijo1*multiplicador)


    if hijo2.sum() < 205: 
        multiplicador = 205/hijo2.sum()
        hijo2 = np.ceil(hijo2*multiplicador)


    return hijo1, hijo2




def recombinar(poblacion, umbral):

    hijos = None

    for i in range(0,poblacion.shape[0],2):

        padre1 = poblacion[i,:].copy()
        padre2 = poblacion[i+1,:].copy()

        

        if distancia_hamming(padre1, padre2)/2 > umbral:
            
            # swap half the differing bits at random
            hijo1, hijo2 = cruce_parent_centered(padre1, padre2) # crea hijo

            if hijos is None:
                hijos = np.array([hijo1, hijo2])
            else:
                hijos = np.append(hijos,[hijo1],0)  
                hijos = np.append(hijos,[hijo2],0)  

            
    return hijos




def select_s(P, hijos, df_padres, df_hijos):


    P_nueva = P.copy()

    df_P_nueva = df_padres.copy()

    cambios = False

    total_hijos = len(hijos)
    
    indice = 0

    while(total_hijos > indice):

        fp = df_padres.iloc[indice,1]
        fh = df_hijos.iloc[indice,1]

        if fp > fh:
            
            indice_padre = df_padres.iloc[indice,0]
            indice_hijo  = df_hijos.iloc[indice,0]

            P_nueva[indice_padre,:] = hijos[indice_hijo,:].copy()

            df_P_nueva.iloc[indice,1] = df_hijos.iloc[indice,1]

            cambios = True

        else:
            break
    
        indice += 1



    return P_nueva, df_P_nueva, cambios



def aclarado(poblacion, f_poblacion, radio_nicho, kappa):

    df_poblacion = pd.DataFrame({
            'indices' : list(range(len(poblacion))),
            'fitness' : f_poblacion
    })

    # Ordenamos los padres (los de mejor fitness arriba)
    df_poblacion.sort_values(by='fitness', ascending=True, inplace=True)

    

    for i in range(len(poblacion)):

        

        fitness_i = df_poblacion.iloc[i,1]
        p_i = poblacion[df_poblacion.iloc[i,0],:]

        if fitness_i > 0:

                num_ganadores = 1
                

                for j in range(i+1,len(poblacion)):

                    

                    fitness_j = df_poblacion.iloc[i,1]
                    p_j = poblacion[df_poblacion.iloc[j,0],:]
                    
                    if (fitness_j > 0) and \
                        (distancia_hamming(p_i,p_j) < radio_nicho):

                        

                        if num_ganadores < kappa:
                            
                            num_ganadores += 1
                        else:
                          df_poblacion.iloc[i,1] = 0
                          
    
    
    
    poblacion_nueva = None
    

    for i in range(len(df_poblacion)):

        
        cromosoma_i = poblacion[df_poblacion.iloc[i,0],:].copy()

        if fitness_i > 0:

            

            if poblacion_nueva is None: 
                poblacion_nueva = np.array([cromosoma_i])
            else:
                poblacion_nueva = np.append(poblacion_nueva,[cromosoma_i],0) 


    
    return poblacion_nueva

     
     