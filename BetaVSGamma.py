#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 18:39:19 2025

@author: erick
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# Parámetros del modelo
N = 100  # Número de nodos
#beta = 0.1  # Probabilidad de infección
steps = 200  # Número de pasos de simulación (Reducimos de 500 para optimizar)
#gamma = 0.1  # Probabilidad de recuperación

# Creamos las redes
regular_graph = nx.random_regular_graph(4, N, seed = 1)
watts_strogatz_graph = nx.watts_strogatz_graph(N, 4, 0.5, seed = 1)
Barabasi_Albert = nx.barabasi_albert_graph(N, 2, seed = 1)

# Función para simular el modelo SIS
def simulate_SIS(G, beta, gamma, steps):
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
                        if states[neighbor] == 'S' and np.random.rand() < beta:
                            new_states[neighbor] = 'I'
        states = new_states
        num_infected.append(sum(1 for state in states.values() if state == 'I'))

    return num_infected


# Definir rangos de valores para beta y probabilidad de contacto
beta_values = np.arange(0.2, 0.52, 0.02)  # De 0.1 a 1.0 con incrementos de 0.1
gamma_values = np.arange(0.2, 0.52, 0.02)  # De 0.1 a 1.0 con incrementos de 0.1
num_experiments = 100  # Número de repeticiones por combinación

# Crear matriz para almacenar resultados
heatmap_data_regular = np.zeros((len(beta_values), len(gamma_values)))
heatmap_data_strogatz = np.zeros((len(beta_values), len(gamma_values)))
heatmap_data_barabbasi = np.zeros((len(beta_values), len(gamma_values)))

# Simulación para cada combinación de parámetros
for i, beta in enumerate(beta_values):
    for j, gamma in enumerate(gamma_values):
        total_infected_ratio_regular = 0
        total_infected_ratio_strogatz = 0
        total_infected_ratio_barabbasi = 0

        for _ in range(num_experiments):
            infected_counts_regular = simulate_SIS(regular_graph, beta, gamma, steps)
            total_infected_ratio_regular+= infected_counts_regular[-1] / N  # Proporción de infectados al final
            
            infected_counts_strogatz = simulate_SIS(watts_strogatz_graph, beta, gamma, steps)
            total_infected_ratio_strogatz+= infected_counts_strogatz[-1] / N  # Proporción de infectados al final
            
            infected_counts_barabbasi = simulate_SIS(Barabasi_Albert, beta, gamma, steps)
            total_infected_ratio_barabbasi+= infected_counts_barabbasi[-1] / N  # Proporción de infectados al final
            
        # Promediar sobre las repeticiones
        heatmap_data_regular[i, j] = total_infected_ratio_regular / num_experiments
        heatmap_data_strogatz[i, j] = total_infected_ratio_strogatz / num_experiments
        heatmap_data_barabbasi[i, j] = total_infected_ratio_barabbasi / num_experiments


# Mapa de calor Red Regular
plt.figure(figsize=(8, 6))
plt.imshow(heatmap_data_regular, cmap="hot", origin="lower", aspect="auto",
           extent=[0.05, 1.0, 0.05, 1.0], interpolation="bicubic")  # Interpolación para suavizar
plt.colorbar(label="Proporción de nodos infectados al final")
plt.xlabel("Probabilidad de Recuperacion (Gamma)")
plt.ylabel("Probabilidad de Contagio (Beta)")
plt.title("Mapa de Calor: Probabilidad de Infección en Red Regular")
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()


# Mapa de Calor Watts Strogatz
plt.figure(figsize=(8, 6))
plt.imshow(heatmap_data_strogatz, cmap="hot", origin="lower", aspect="auto",
           extent=[0.05, 1.0, 0.05, 1.0], interpolation="bicubic")  # Interpolación para suavizar
plt.colorbar(label="Proporción de nodos infectados al final")
plt.xlabel("Probabilidad de Recuperacion (Gamma)")
plt.ylabel("Probabilidad de Contagio (Beta)")
plt.title("Mapa de Calor: Probabilidad de Infección Red Strogatz")
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# Mapa de Calor Barabbasi Albert
plt.figure(figsize=(8, 6))
plt.imshow(heatmap_data_barabbasi, cmap="hot", origin="lower", aspect="auto",
           extent=[0.05, 1.0, 0.05, 1.0], interpolation="bicubic")  # Interpolación para suavizar
plt.colorbar(label="Proporción de nodos infectados al final")
plt.xlabel("Probabilidad de Recuperacion (Gamma)")
plt.ylabel("Probabilidad de Contagio (Beta)")
plt.title("Mapa de Calor: Probabilidad de Infección en Red Barabbasi")
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
