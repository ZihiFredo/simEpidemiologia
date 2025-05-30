#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 18:50:41 2025

@author: erick
"""

import matplotlib.pyplot as plt
from Reprogramacion import crear_red, reconectar_aristas
from AntivaxSim import generar_poblacion, difundir_antivacunas
from SVIR import simular_virus
import networkx as nx

# -----------------------
# 1. Crear red original
# -----------------------
N = 300  # Red pequeña para prueba visual
G_opiniones = crear_red('pequeno_mundo', N=N, k=6, p=0.5, distancia_umbral=0.05)

# -----------------------
# 2. Generar población
# -----------------------
edad, muerte = generar_poblacion(N, seed=42)

# -----------------------
# 3. Difusión de opiniones anti-vacunas
# -----------------------
resultado_antiv = difundir_antivacunas(G_opiniones, theta=0.4, semilla=42)
A_antivacunas = resultado_antiv["A"]  # nodos que rechazan vacuna

# -----------------------
# 4. Modificar red para la propagación del virus
# -----------------------
G_contacto = reconectar_aristas(G_opiniones, porcentaje=0.3, semilla=42)

# -----------------------
# 5. Simular SVIR
# -----------------------
resultado_virus = simular_virus(
    G_contacto,
    A_antivacunas=A_antivacunas,
    edad=edad,
    muerte=muerte,
    tasa_vac=0.3,
    beta=0.05,
    tau=14,
    estrategia='grado',
    semilla=42
)

# -----------------------
# 6. Mostrar resultados
# -----------------------
print("⏱ Tiempo total de epidemia:", resultado_virus["tiempo_total"])
print("🧍‍♂️ Porcentaje recuperados:", round(100 * resultado_virus["porcentaje_recuperados"], 2), "%")
print("⚰️ Porcentaje muertos:", round(100 * resultado_virus["porcentaje_muertos"], 2), "%")
print("📉 Años de vida perdidos:", round(resultado_virus["años_de_vida_perdidos"], 4))

# -----------------------
# 7. Visualización final de opiniones (opcional)
# -----------------------
color_map = []
for i in range(N):
    if A_antivacunas[i]:
        color_map.append('red')  # Adoptaron la opinión
    elif resultado_antiv["D"][i]:
        color_map.append('blue')  # Rechazaron la opinión
    elif resultado_antiv["S"][i]:
        color_map.append('green')  # No fueron expuestos

plt.figure(figsize=(8, 6))
nx.draw_spring(G_opiniones, node_color=color_map, node_size=50, with_labels=False)
plt.title("Red de opiniones final (Rojo: antivacunas, Azul: rechazo, Verde: no expuestos)")
plt.show()