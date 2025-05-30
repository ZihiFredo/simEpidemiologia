#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 18:33:57 2025

@author: erick
"""

import numpy as np
import random

def generar_poblacion(N, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    edad = 100 * np.random.beta(2, 3, N)
    muerte = 100 * np.random.beta(5, 2, N)
    
    while np.any(edad > muerte):
        indices = edad > muerte
        edad[indices] = 100 * np.random.beta(2, 3, np.sum(indices))
        muerte[indices] = 100 * np.random.beta(5, 2, np.sum(indices))
    
    return edad, muerte


def difundir_antivacunas(G, theta=0.3, semilla=None):
    if semilla is not None:
        random.seed(semilla)
        np.random.seed(semilla)

    N = G.number_of_nodes()
    nodos = list(G.nodes)
    
    # Inicialización: 1 de cada 80 tiene la opinión anti-vacuna
    A = np.zeros(N, dtype=bool)  # Adoptaron la opinión
    E = np.zeros(N, dtype=bool)  # Expuestos pero no decididos aún
    D = np.zeros(N, dtype=bool)  # Rechazaron la opinión
    S = np.ones(N, dtype=bool)   # Susceptibles

    iniciales = random.sample(nodos, N // 80)
    A[iniciales] = True
    S[iniciales] = False

    # Exponer a vecinos de los A actuales
    for i in range(N):
        if not A[i] and not D[i]:
            vecinos = list(G.neighbors(i))
            if any(A[v] for v in vecinos):
                E[i] = True
                S[i] = False

    tiempo = 0
    historia = []
    while E.any():
        nuevos_A = (np.random.rand(N) < theta) & E
        nuevos_D = E & (~nuevos_A)

        A[nuevos_A] = True
        D[nuevos_D] = True
        E[:] = False

        # Actualizar exposición para la siguiente ronda
        for i in range(N):
            if not A[i] and not D[i]:
                vecinos = list(G.neighbors(i))
                if any(A[v] for v in vecinos):
                    E[i] = True
                    S[i] = False

        historia.append({
            "tiempo": tiempo,
            "adoptaron": A.sum(),
            "rechazaron": D.sum(),
            "expuestos": E.sum(),
            "susceptibles": S.sum()
        })

        tiempo += 1

    return {
        "A": A,
        "D": D,
        "E": E,
        "S": S,
        "historia": historia,
        "tiempo_total": tiempo,
        "porcentaje_antivac": A.sum() / N
    }

'''
import matplotlib.pyplot as plt

# Parámetros para prueba
N = 200  # Red pequeña para visualizar
theta = 0.4  # Probabilidad de adoptar la idea
semilla = 42

# Crear red de tipo pequeño mundo
G = nx.watts_strogatz_graph(n=N, k=6, p=0.1, seed=semilla)

# Generar población
edad, muerte = generar_poblacion(N, seed=semilla)

# Simular difusión de opiniones anti-vacunas
resultado = difundir_antivacunas(G, theta=theta, semilla=semilla)

# Visualizar resultados finales
A = resultado['A']
D = resultado['D']
S = resultado['S']

color_map = []
for i in range(N):
    if A[i]:
        color_map.append('red')  # Adoptaron la opinión
    elif D[i]:
        color_map.append('blue')  # La rechazaron
    elif S[i]:
        color_map.append('green')  # No fueron expuestos

plt.figure(figsize=(8, 6))
nx.draw_spring(G, node_color=color_map, node_size=50, with_labels=False)
plt.title(f"Red final con θ = {theta}\nRojo=Antivacunas, Azul=Rechazaron, Verde=No expuestos")
plt.show()

# Información general
print(f"Tiempo total de difusión: {resultado['tiempo_total']}")
print(f"Nodos que adoptaron la opinión: {sum(A)}")
print(f"Nodos que la rechazaron: {sum(D)}")
print(f"Nodos no expuestos: {sum(S)}")
'''