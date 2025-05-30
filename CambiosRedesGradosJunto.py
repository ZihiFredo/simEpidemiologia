#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  5 12:36:06 2025

@author: erick
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Parámetros del modelo
N = 200
beta = 0.2
steps = 200
gamma = 0.1
probContacto = 0.2
grados = range(2, 21)
num_experiments = 100

# Función SIS
def simulate_SIS(G, beta, gamma, probContacto, steps):
    states = {node: 'S' for node in G.nodes}
    initial_infected = np.random.choice(G.nodes, size=10, replace=False)
    for node in initial_infected:
        states[node] = 'I'
    for _ in range(steps):
        new_states = states.copy()
        for node in G.nodes:
            if states[node] == 'I':
                if np.random.rand() < gamma:
                    new_states[node] = 'S'
                else:
                    for neighbor in G.neighbors(node):
                        if states[neighbor] == 'S' and np.random.rand() < beta * probContacto:
                            new_states[neighbor] = 'I'
        states = new_states
    return sum(1 for s in states.values() if s == 'I') / len(G.nodes)

# Inicializar resultados
resultados_regular = []
resultados_barabasi = []
resultados_strogatz = []

for grado in grados:
    print(f"Simulando para grado = {grado}...")
    
    # Red Regular
    total_inf_r = 0
    G = nx.random_regular_graph(grado, N)
    for _ in range(num_experiments):
        total_inf_r += simulate_SIS(G, beta, gamma, probContacto, steps)
    resultados_regular.append(total_inf_r / num_experiments)
    
    # Barabási-Albert (m ≈ grado / 2)
    m = max(1, grado // 2)
    total_inf_ba = 0
    G = nx.barabasi_albert_graph(N, m)
    for _ in range(num_experiments):
        total_inf_ba += simulate_SIS(G, beta, gamma, probContacto, steps)
    resultados_barabasi.append(total_inf_ba / num_experiments)
    
    # Watts-Strogatz
    k = grado
    total_inf_ws = 0
    G = nx.watts_strogatz_graph(N, k, 0.5)
    for _ in range(num_experiments):
        total_inf_ws += simulate_SIS(G, beta, gamma, probContacto, steps)
    resultados_strogatz.append(total_inf_ws / num_experiments)

# 1. Extraer valores solo para grados pares en Barabási y Strogatz
grados_pares = np.array(grados)[np.array(grados) % 2 == 0]
barabasi_pares = [resultados_barabasi[i] for i, g in enumerate(grados) if g % 2 == 0]
strogatz_pares = [resultados_strogatz[i] for i, g in enumerate(grados) if g % 2 == 0]

# 2. Interpolación (lineal)
interp_barabasi = interp1d(grados_pares, barabasi_pares, kind='linear', fill_value='extrapolate')
interp_strogatz = interp1d(grados_pares, strogatz_pares, kind='linear', fill_value='extrapolate')

# 3. Graficar
plt.figure(figsize=(10, 6))

# Red Regular: todos los puntos
plt.plot(grados, resultados_regular, label='Red Regular', marker='o', linestyle='-')

grados_array = np.array(grados)

# Barabási-Albert
plt.plot(grados_array, interp_barabasi(grados_array), label='Barabási-Albert', linestyle='-', color='orange', marker='^')
# Watts-Strogatz
plt.plot(grados_array, interp_strogatz(grados_array), label='Watts-Strogatz', linestyle='-', color='green', marker='*')

plt.xlabel("Grado de los nodos")
plt.ylabel("Probabilidad de Infección")
plt.title("Infección final vs Grado en topologías")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()