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
N_values = np.arange(50, 1050, 50)
grados_values = np.arange(1, 21, 1)  
num_experiments = 100 # Número de repeticiones por combinación

resultados_nodos = []

for N in N_values:
    
    total_infected_ratio = 0
    
    for _ in range(num_experiments):
        regular_graph = nx.random_regular_graph(4, N, seed = 1)
        infected_counts = simulate_SIS(regular_graph, beta, gamma, probContacto, steps)
        total_infected_ratio += infected_counts[-1] / N  # Proporción de infectados al final
            
    # Promediar sobre las repeticiones
    resultados_nodos.append(total_infected_ratio / num_experiments)


resultados_grados = []

for grad in grados_values:
    
    total_infected_ratio = 0
    
    for _ in range(num_experiments):
        regular_graph = nx.random_regular_graph(grad, NOrg, seed = 1)
        infected_counts = simulate_SIS(regular_graph, beta, gamma, probContacto, steps)
        total_infected_ratio += infected_counts[-1] / NOrg  # Proporción de infectados al final
            
    # Promediar sobre las repeticiones
    resultados_grados.append(total_infected_ratio / num_experiments)


# Graficar resultados
plt.figure(figsize=(12, 5))

# Gráfico 1: Infección vs Número de Nodos
plt.subplot(1, 2, 1)
plt.plot(N_values, resultados_nodos, marker='o', linestyle='-', color='b')
plt.xlabel("Número de Nodos en la Red Regular")
plt.ylabel("Probabilidad de infeccion")
plt.title("Infección vs Número de Nodos de la Red Regular")
plt.grid()

# Gráfico 2: Infección vs Grado de la Red Regular
plt.subplot(1, 2, 2)
plt.plot(grados_values, resultados_grados, marker='o', linestyle='-', color='r')
plt.xlabel("Grado de la Red Regular")
plt.ylabel("Probabilidad de infeccion")
plt.title("Infección vs Grado de la Red Regular")
plt.grid()

# Mostrar gráficas
plt.tight_layout()
plt.show()
