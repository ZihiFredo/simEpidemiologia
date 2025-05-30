#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 12:17:08 2025

@author: erick
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# Parámetros del modelo
NOrg = 200  # Número de nodos
beta = 0.3  # Probabilidad de infección
steps = 200  # Número de pasos de simulación (Reducimos de 500 para optimizar)
gamma = 0.1  # Probabilidad de recuperación
probContacto = 0.3 # Probabilidad de Contacto entre nodos conectados

# Función para simular el modelo SIS
def simulate_SIS(G, beta, gamma, probContacto, steps):
    states = {node: 'S' for node in G.nodes}
    initial_infected = np.random.choice(G.nodes, size=10, replace=False)
    for node in initial_infected:
        states[node] = 'I'
    num_infected = []

    for _ in range(steps):
        new_states = states.copy()
        for node in G.nodes:
            if states[node] == 'I':  # Nodo infectado
                if np.random.rand() < gamma:
                    new_states[node] = 'S'
                else:
                    for neighbor in G.neighbors(node): #Podemos hacerlo un while para hacerlo mas rapido
                        if states[neighbor] == 'S' and np.random.rand() < beta*probContacto:
                            new_states[neighbor] = 'I'
        states = new_states
        num_infected.append(sum(1 for state in states.values() if state == 'I'))

    return num_infected


# Definir rangos de valores

# N int The number of nodes

# k int Each node is joined with its k nearest neighbors in a ring topology.

# p float The probability of rewiring each edge

N_values = np.arange(50, 1050, 50)
k_values = np.arange(2, 21, 1)  
p_values = np.arange(0.05, 1, 0.05)  
num_experiments = 1000  # Número de repeticiones por combinación

'''
resultados_nodos = []

for N in N_values:
    
    total_infected_ratio = 0
    
    for _ in range(num_experiments):
        watts_strogatz_graph = nx.watts_strogatz_graph(N, 4, 0.1, seed = 1)
        infected_counts = simulate_SIS(watts_strogatz_graph, beta, gamma, probContacto, steps)
        total_infected_ratio += infected_counts[-1] / N  # Proporción de infectados al final
            
    # Promediar sobre las repeticiones
    resultados_nodos.append(total_infected_ratio / num_experiments)

resultados_k = []
 
for k in k_values:
    
    total_infected_ratio = 0
    
    for _ in range(num_experiments):
        watts_strogatz_graph = nx.watts_strogatz_graph(NOrg, k, 0.1, seed = 1)
        infected_counts = simulate_SIS(watts_strogatz_graph, beta, gamma, probContacto, steps)
        total_infected_ratio += infected_counts[-1] / NOrg  # Proporción de infectados al final
            
    # Promediar sobre las repeticiones
    resultados_k.append(total_infected_ratio / num_experiments) 
'''
resultados_p = []

for p in p_values:
    
    total_infected_ratio = 0
    
    for _ in range(num_experiments):
        watts_strogatz_graph = nx.watts_strogatz_graph(NOrg, 4, p, seed = 1)
        infected_counts = simulate_SIS(watts_strogatz_graph, beta, gamma, probContacto, steps)
        total_infected_ratio += infected_counts[-1] / NOrg  # Proporción de infectados al final
            
    # Promediar sobre las repeticiones
    resultados_p.append(total_infected_ratio / num_experiments)


# Graficar resultados
plt.figure(figsize=(12, 8))
'''
# Gráfico 1: Infección vs Número de Nodos
plt.subplot(2, 2, 1)
plt.plot(N_values, resultados_nodos, marker='o', linestyle='-', color='b')
plt.xlabel("Número de Nodos en la Red Strogatz")
plt.ylabel("Probabilidad de infeccion")
plt.title("Infección vs Número de Nodos de la Red Strogatz")
plt.grid()

# Gráfico 2: Infección vs Grado de la Red Regular
plt.subplot(2, 2, 2)
plt.plot(k_values, resultados_k, marker='o', linestyle='-', color='r')
plt.xlabel("Vecinos de la Red Strogatz")
plt.ylabel("Probabilidad de infeccion")
plt.title("Infección vs Vecinos de la Red Strogatz")
plt.grid()
'''
# Gráfico 3: Infección vs Grado de la Red Regular
plt.subplot(2, 2, 3)
plt.plot(p_values, resultados_p, marker='o', linestyle='-', color='r')
plt.xlabel("Probabilidad de reconexion de la Red Strogatz")
plt.ylabel("Probabilidad de infeccion")
plt.title("Infección vs Probabilidad de reconexion")
plt.grid()

# Mostrar gráficas
plt.tight_layout()
plt.show()