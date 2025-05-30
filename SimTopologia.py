#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 13:28:22 2025

@author: erick
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from Reprogramacion import crear_red
from AntivaxSim import generar_poblacion, difundir_antivacunas
from SVIR import simular_virus

# Ejecutamos simulaciones para cambio de topología desde diferentes redes de opiniones iniciales

tipos_red_inicial = ['pequeno_mundo', 'libre_escala', 'proximidad']
tipos_red_topologia = ['pequeno_mundo', 'libre_escala', 'proximidad']
thetas = np.arange(0.0, 1.05, 0.05)
N = 300

# Resultados organizados por red inicial → tipo red de contacto → valores
resultados_topologia_multi = {
    tipo_ini: {tipo_topo: [] for tipo_topo in tipos_red_topologia}
    for tipo_ini in tipos_red_inicial
}

for tipo_ini in tipos_red_inicial:
    for tipo_topo in tipos_red_topologia:
        G_opiniones = crear_red(tipo_ini, N=N, k=6, p=0.5, distancia_umbral=0.082)
        edad, muerte = generar_poblacion(N)
        G_contacto = crear_red(tipo_topo, N=N, k=6, p=0.5, distancia_umbral=0.082)
        for theta in tqdm(thetas, desc=f"{tipo_ini} → {tipo_topo}"):
            años_prom = []
            for _ in range(500):
                resultado_antivac = difundir_antivacunas(G_opiniones, theta=theta)
                A_antivac = resultado_antivac["A"]
                resultado_virus = simular_virus(G_contacto, A_antivac, edad, muerte)
                años_prom.append(resultado_virus["años_de_vida_perdidos"])
            resultados_topologia_multi[tipo_ini][tipo_topo].append(np.mean(años_prom))

# Graficar una figura por tipo de red inicial
fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

for idx, tipo_ini in enumerate(tipos_red_inicial):
    ax = axs[idx]
    for tipo_topo in tipos_red_topologia:
        ax.plot(thetas, resultados_topologia_multi[tipo_ini][tipo_topo],
                label=f"Red contacto: {tipo_topo.replace('_', ' ')}")
    ax.set_title(f"Red de opiniones: {tipo_ini.replace('_', ' ').title()}")
    ax.set_xlabel("Persuasión anti-vacunas (θ)")
    if idx == 0:
        ax.set_ylabel("Años de vida perdidos")
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()
