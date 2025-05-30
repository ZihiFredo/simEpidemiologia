#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 18:42:56 2025

@author: erick
"""

import numpy as np
import networkx as nx
import random

def eliminar_aristas(G, porcentaje=0.3, semilla=None):
    if semilla is not None:
        random.seed(semilla)
    G_mod = G.copy()
    edges = list(G_mod.edges)
    n_eliminar = int(porcentaje * len(edges))
    edges_a_remover = random.sample(edges, n_eliminar)
    G_mod.remove_edges_from(edges_a_remover)
    return G_mod


def reconectar_aristas(G, porcentaje=0.3, semilla=None):
    if semilla is not None:
        random.seed(semilla)
    G_mod = G.copy()
    edges = list(G_mod.edges)
    nodes = list(G_mod.nodes)
    n_rewire = int(porcentaje * len(edges))
    edges_a_reemplazar = random.sample(edges, n_rewire)
    G_mod.remove_edges_from(edges_a_reemplazar)

    for u, v in edges_a_reemplazar:
        extremo_fijo = u if random.random() < 0.5 else v
        nuevo_vecino = random.choice(nodes)
        while nuevo_vecino == extremo_fijo or G_mod.has_edge(extremo_fijo, nuevo_vecino):
            nuevo_vecino = random.choice(nodes)
        G_mod.add_edge(extremo_fijo, nuevo_vecino)

    return G_mod


def crear_red(tipo_red, N=5000, k=6, p=0.1, distancia_umbral=0.05, semilla=None):
    if semilla is not None:
        random.seed(semilla)
    if tipo_red == 'pequeno_mundo':
        return nx.watts_strogatz_graph(N, k, p)
    elif tipo_red == 'libre_escala':
        return nx.barabasi_albert_graph(N, k // 2)
    elif tipo_red == 'proximidad':
        pos = {i: (np.random.rand(), np.random.rand()) for i in range(N)}
        G = nx.Graph()
        G.add_nodes_from(pos)
        for i in range(N):
            for j in range(i + 1, N):
                if np.linalg.norm(np.array(pos[i]) - np.array(pos[j])) < distancia_umbral:
                    G.add_edge(i, j)
        return G
    else:
        raise ValueError("Tipo de red no reconocido.")


def cambiar_topologia(N, tipo_nueva='libre_escala', k=6, p=0.1, distancia=0.05, semilla=None):
    return crear_red(tipo_nueva, N, k, p, distancia)

'''
# Red original de opiniones
red_opiniones = nx.watts_strogatz_graph(n=5000, k=6, p=0.1, seed=1)

# 1. Eliminar 30% de las aristas
red_contacto_drop = eliminar_aristas(red_opiniones, porcentaje=0.3, semilla=1)

# 2. Rewirear 30% de las aristas
red_contacto_rewire = reconectar_aristas(red_opiniones, porcentaje=0.3, semilla=1)

# 3. Nueva topología completamente diferente
red_contacto_nueva = cambiar_topologia(5000, tipo_nueva='libre_escala', k=6, semilla=1)
'''

