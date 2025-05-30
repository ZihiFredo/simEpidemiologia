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
gamma = 0.1  # Probabilidad de recuperación
probContacto = 0.2 # Probabilidad de Contacto entre nodos conectados

# Creamos las redes
regular_graph = nx.random_regular_graph(4, N, seed = 1)
watts_strogatz_graph = nx.watts_strogatz_graph(N, 4, 0.5, seed = 1)
Barabasi_Albert = nx.barabasi_albert_graph(N, 2, seed = 1)

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


# Definir rangos de valores para beta y probabilidad de contacto
beta_values = np.arange(0.005, 0.255, 0.005)
num_experiments = 250  # Número de repeticiones por combinación

# Diccionario para almacenar datos de infección por red
infection_data = {
    "Regular": {"metric": [], "infection_ratio": []},
    "Watts-Strogatz": {"metric": [], "infection_ratio": []},
    "Barabási-Albert": {"metric": [], "infection_ratio": []}
}

# Obtener las matrices de adyacencia como arrays de NumPy
A_regular = nx.adjacency_matrix(regular_graph).toarray()
A_watts_strogatz = nx.adjacency_matrix(watts_strogatz_graph).toarray()
A_barabasi_albert = nx.adjacency_matrix(Barabasi_Albert).toarray()

# Calcular el eigenvalor más grande (valor propio dominante)
lambda_max_regular = max(np.linalg.eigvals(A_regular))
lambda_max_watts_strogatz = max(np.linalg.eigvals(A_watts_strogatz))
lambda_max_barabasi_albert = max(np.linalg.eigvals(A_barabasi_albert))


# Simulación para cada combinación de parámetros
for i, beta in enumerate(beta_values):
        total_infected_ratio_regular = 0
        total_infected_ratio_strogatz = 0
        total_infected_ratio_barabbasi = 0

        for _ in range(num_experiments):
            infected_counts_regular = simulate_SIS(regular_graph, beta, gamma, probContacto, steps)
            total_infected_ratio_regular+= infected_counts_regular[-1] / N  # Proporción de infectados al final
            
            infected_counts_strogatz = simulate_SIS(watts_strogatz_graph, beta, gamma, probContacto, steps)
            total_infected_ratio_strogatz+= infected_counts_strogatz[-1] / N  # Proporción de infectados al final
            
            infected_counts_barabbasi = simulate_SIS(Barabasi_Albert, beta, gamma, probContacto, steps)
            total_infected_ratio_barabbasi+= infected_counts_barabbasi[-1] / N  # Proporción de infectados al final
            
        # Calcular la métrica de propagación
        metric_regular = lambda_max_regular * (beta * probContacto) / gamma
        metric_watts_strogatz = lambda_max_watts_strogatz * (beta * probContacto) / gamma
        metric_barabasi_albert = lambda_max_barabasi_albert * (beta * probContacto) / gamma
            
        # Almacenar datos
        infection_data["Regular"]["metric"].append(metric_regular.real)
        infection_data["Regular"]["infection_ratio"].append(total_infected_ratio_regular / num_experiments)

        infection_data["Watts-Strogatz"]["metric"].append(metric_watts_strogatz.real)
        infection_data["Watts-Strogatz"]["infection_ratio"].append(total_infected_ratio_strogatz / num_experiments)

        infection_data["Barabási-Albert"]["metric"].append(metric_barabasi_albert.real)
        infection_data["Barabási-Albert"]["infection_ratio"].append(total_infected_ratio_barabbasi / num_experiments)


# Graficar los resultados
plt.figure(figsize=(15, 5))

# Regular Graph
plt.subplot(1, 3, 1)
plt.scatter(infection_data["Regular"]["metric"], infection_data["Regular"]["infection_ratio"], alpha=0.5, color="b")
plt.axvline(x=1, color='k', linestyle='--', label="Threshold = 1")
plt.xlabel("Métrica de Propagación")
plt.ylabel("Proporción de Nodos Infectados")
plt.title("Propagación en Red Regular")
plt.legend()
plt.grid()

# Watts-Strogatz Graph
plt.subplot(1, 3, 2)
plt.scatter(infection_data["Watts-Strogatz"]["metric"], infection_data["Watts-Strogatz"]["infection_ratio"], alpha=0.5, color="r")
plt.axvline(x=1, color='k', linestyle='--', label="Threshold = 1")
plt.xlabel("Métrica de Propagación")
plt.ylabel("Proporción de Nodos Infectados")
plt.title("Propagación en Red Watts-Strogatz")
plt.legend()
plt.grid()

# Barabási-Albert Graph
plt.subplot(1, 3, 3)
plt.scatter(infection_data["Barabási-Albert"]["metric"], infection_data["Barabási-Albert"]["infection_ratio"], alpha=0.5, color="g")
plt.axvline(x=1, color='k', linestyle='--', label="Threshold = 1")
plt.xlabel("Métrica de Propagación")
plt.ylabel("Proporción de Nodos Infectados")
plt.title("Propagación en Red Barabási-Albert")
plt.legend()
plt.grid()

# Mostrar gráficas
plt.tight_layout()
plt.show()