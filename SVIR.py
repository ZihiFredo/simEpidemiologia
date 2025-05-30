#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 18:49:15 2025

@author: erick
"""

import numpy as np
import random

def simular_virus(G, A_antivacunas, edad, muerte, tasa_vac=0.3, beta=0.05, tau=14, estrategia='grado', semilla=None):
    if semilla is not None:
        random.seed(semilla)
        np.random.seed(semilla)
    
    N = G.number_of_nodes()
    nodos = list(G.nodes)

    # 1. Infección inicial (1 de cada 80)
    infectados = set(random.sample(nodos, N // 80))
    vacunados = set()
    
    # 2. Vacunación (solo si no está infectado ni es antivacunas)
    candidatos = list(set(nodos) - infectados - set(np.where(A_antivacunas)[0]))

    if estrategia == 'grado':
        grados = dict(G.degree)
        ordenados = sorted(candidatos, key=lambda x: grados[x], reverse=True)
    elif estrategia == 'edad_mayor':
        ordenados = sorted(candidatos, key=lambda x: edad[x], reverse=True)
    elif estrategia == 'edad_menor':
        ordenados = sorted(candidatos, key=lambda x: edad[x])
    elif estrategia == 'aleatoria':
        ordenados = random.sample(candidatos, len(candidatos))
    else:
        raise ValueError("Estrategia no reconocida.")

    n_vacunar = int(tasa_vac * N)
    vacunados = set(ordenados[:n_vacunar])

    susceptibles = set(nodos) - infectados - vacunados
    recuperados = set()
    dias_infectado = {i: 0 for i in infectados}
    t = 0

    while infectados:
        nuevos_infectados = set()
        a_retirar = set()

        for nodo in infectados:
            dias_infectado[nodo] += 1
            if dias_infectado[nodo] > tau:
                recuperados.add(nodo)
                a_retirar.add(nodo)
            else:
                for vecino in G.neighbors(nodo):
                    if vecino in susceptibles and random.random() < beta:
                        nuevos_infectados.add(vecino)

        for nodo in nuevos_infectados:
            dias_infectado[nodo] = 1
        infectados |= nuevos_infectados #Union de Conjuntos
        infectados -= a_retirar
        susceptibles -= nuevos_infectados
        t += 1

    # Cálculo de muertes y años de vida perdidos
    muertos = []
    for nodo in recuperados:
        if random.random() < edad[nodo] / 200: # 0.05 es el factor de muerte por edad
            muertos.append(nodo)

    años_perdidos = sum((muerte[nodo] - edad[nodo]) for nodo in muertos) / N

    return {
        "tiempo_total": t,
        "porcentaje_recuperados": len(recuperados) / N,
        "porcentaje_muertos": len(muertos) / N,
        "años_de_vida_perdidos": años_perdidos
    }