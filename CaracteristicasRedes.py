#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 11:38:09 2025

@author: erick
"""

# Reimportar librerías necesarias después del reinicio
import networkx as nx
import numpy as np

# Crear las redes nuevamente
N = 100  # Número de nodos


for i in range(10):
    regular_graph = nx.random_regular_graph((i+1)*2, N, seed=1)
    watts_strogatz_graph = nx.watts_strogatz_graph(N, (i+1)*2, 0.5, seed=1)
    barabasi_albert = nx.barabasi_albert_graph(N, (i+1), seed=1)
    
    # Obtener las matrices de adyacencia como arrays de NumPy
    A_regular = nx.adjacency_matrix(regular_graph).toarray()
    A_watts_strogatz = nx.adjacency_matrix(watts_strogatz_graph).toarray()
    A_barabasi_albert = nx.adjacency_matrix(barabasi_albert).toarray()
    
    '''
    # Calcular el eigenvalor más grande (valor propio dominante)
    lambda_max_regular = max(np.linalg.eigvals(A_regular))
    lambda_max_watts_strogatz = max(np.linalg.eigvals(A_watts_strogatz))
    '''
    lambda_max_barabasi_albert = max(np.linalg.eigvals(A_barabasi_albert))
    '''
    print("Regular " + str(i) + ":" + str(lambda_max_regular))
    print("Watts " + str(i) + ":" + str(lambda_max_watts_strogatz))
    '''
    grado_promedio = sum(dict(barabasi_albert.degree()).values()) / barabasi_albert.number_of_nodes()
    print(f"Grado promedio: {grado_promedio}")
    print("Barabasi " + str(i) + ":" + str(lambda_max_barabasi_albert))
    print("===========================")
    
    
    
    '''
    print(nx.average_clustering(regular_graph))
    print(nx.average_clustering(watts_strogatz_graph))
    print(nx.average_clustering(barabasi_albert))
    '''
    
    '''
    Grafo Regular: 
    𝜆max = 3.0
    
    Watts-Strogatz: 
    𝜆max = 4.173
    
    Barabási-Albert: 
    𝜆max = 6.896
    '''

