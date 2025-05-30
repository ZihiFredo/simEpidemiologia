#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 10:37:41 2025

@author: erick
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from Reprogramacion import crear_red, reconectar_aristas
from AntivaxSim import generar_poblacion, difundir_antivacunas
from SVIR import simular_virus

# Parámetros del experimento
tipos_red = ['pequeno_mundo', 'libre_escala', 'proximidad']
porcentajes = [0.0, 0.3, 0.6, 0.9]
thetas = np.arange(0.0, 1.05, 0.05)
resultados = {tipo: {p: [] for p in porcentajes} for tipo in tipos_red}

# Simulaciones
for tipo in tipos_red:
    G_opiniones = crear_red(tipo, N=300, k=6, p=0.5, distancia_umbral=0.082)
    edad, muerte = generar_poblacion(300)
    for p in porcentajes:
        for theta in tqdm(thetas, desc=f"{tipo} p={p}"):
            años_prom = []
            for _ in range(500):
                resultado_antivac = difundir_antivacunas(G_opiniones, theta=theta)
                A_antivac = resultado_antivac["A"]
                G_contacto = reconectar_aristas(G_opiniones, porcentaje=p)
                resultado_virus = simular_virus(G_contacto, A_antivac, edad, muerte)
                años_prom.append(resultado_virus["años_de_vida_perdidos"])
            resultados[tipo][p].append(np.mean(años_prom))

# Graficar
fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
for i, tipo in enumerate(tipos_red):
    for p in porcentajes:
        axs[i].plot(thetas, resultados[tipo][p], label=f"{int(p*100)}% aristas reconectadas")
    axs[i].set_title(f"Red: {tipo.replace('_', ' ').title()}")
    axs[i].set_xlabel("Persuasión anti-vacunas (θ)")
    axs[i].set_ylabel("Años de vida perdidos")
    axs[i].legend()
    axs[i].grid(True)
plt.tight_layout()
plt.show()