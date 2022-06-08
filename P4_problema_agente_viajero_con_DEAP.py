# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 23:51:05 2022

@author: germa
"""

#problema del agente viajero con algoritmos genéticos
#pasos a seguir
# 1) inicializar la población (las ciudades)
# 2) determinar el largo de los caminos
# 3) Hasta que este listo, repetir:
    #a) seleccionar Padres
    #b) Realizar el cruce y la mutación
    #c) calcular el largo de la nueva población
    #d) añadirlo al stock de genes
    
#cargamos las librerias
import random
import numpy as np

from deap import base, creator,tools
from deap import algorithms

#introducimos las distancias
numero_ciudades=5 #(A,B,C,D,E)
distancias= np.zeros((numero_ciudades,numero_ciudades)) #ponemos en cero las distancias
print (distancias)

#añadimos las distancias a la matriz
distancias[0][1]=7
distancias[0][2]=9
distancias[0][3]=8
distancias[0][4]=20
distancias[1][0]=7
distancias[1][2]=10
distancias[1][3]=4
distancias[1][4]=11
distancias[2][0]=9
distancias[2][1]=10
distancias[2][3]=15
distancias[2][4]=5
distancias[3][0]=8
distancias[3][1]=4
distancias[3][2]=15
distancias[3][4]=17
distancias[4][0]=20
distancias[4][1]=11
distancias[4][2]=5
distancias[4][3]=17

print(distancias)

#establecemos el algoritmo genetico
# minimizamos el camino (queremos encontrar el camino más corto)
creator.create("FitnessMin", base.Fitness,weights=(-1.0,) )
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox=base.Toolbox()

#establecemos la permutación por individuo
toolbox.register("indices", random.sample, range(numero_ciudades), numero_ciudades)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)

#establecemos la población
toolbox.register("population",tools.initRepeat,list,toolbox.individual)

# métodos:cruce, mutación, fitness
def evaluateAGENTE_VIAJERO(individual):
    print("individiual: ", individual)
    suma=0
    start=individual[0]
    for i in range (1, len(individual)):
        end=individual[i]
        suma += distancias[start][end]
        start =end
    print("suma: ",suma)    
    return suma #devolvemos la suma
        
toolbox.register("evaluate", evaluateAGENTE_VIAJERO)
toolbox.register("mate", tools.cxPartialyMatched)
#toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=2.0/numero_ciudades)
toolbox.register("select", tools.selTournament, tournsize=3)

#hacemos correr y examinamos los resultados
def main(seed=0):
    random.seed(seed)

    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("Avg", np.mean)
    stats.register("Std", np.std)
    stats.register("Min", np.min)
    stats.register("Max", np.max)

    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=100, stats=stats,
                        halloffame=hof, verbose=True)

    return pop, stats, hof

if __name__ == "__main__":
    main()

'''
tamano_poblacion=200
numero_iteraciones= 1000
numero_cruces=50
a = Runner(toolbox)
a.set_parameters(tamano_poblacion,numero_iteraciones,numero_cruces)
stats,population=a.Run()
'''