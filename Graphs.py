import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Parámetros del modelo
N = 100  # Número de nodos

#d: Grado de cada nodo en un grafo regular.
d = 3

#p: Probabilidad de reconectar enlaces en el modelo de mundo pequeño.
p = 0.1

#k: Número de vecinos iniciales en el modelo de Watts-Strogatz.
k = 4

#m: Número de conexiones que cada nuevo nodo hace en el modelo de Barabási-Albert.
m = 2

# Crear redes
networks = {
    "Barabási-Albert": nx.barabasi_albert_graph(N, m, seed = 1),
    "Watts-Strogatz": nx.watts_strogatz_graph(N, k, p, seed = 1),
    "Regular": nx.random_regular_graph(d, N, seed = 1)
}

# Graficar las redes
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (name, G) in zip(axes, networks.items()):
    nx.draw(G, ax=ax, node_size=30, edge_color="gray", with_labels=False)
    ax.set_title(name)
plt.show()


# Revisar y corregir el cálculo de SWI y omega

def calculate_network_properties_fixed(G):
    N = len(G.nodes)
    
    # Coeficiente de clustering promedio
    C = nx.average_clustering(G)

    # Longitud promedio del camino más corto
    L = nx.average_shortest_path_length(G)
    
    # Comparación con una red aleatoria (asegurando que sea conexa)
    G_random = nx.erdos_renyi_graph(N, p=nx.density(G))
    while not nx.is_connected(G_random):  # Asegurar que la red sea conexa
        G_random = nx.erdos_renyi_graph(N, p=nx.density(G))

    C_rand = nx.average_clustering(G_random)
    L_rand = nx.average_shortest_path_length(G_random)

    # Generar red tipo "rejilla" (lattice) y asegurar conexión
    G_lattice = nx.watts_strogatz_graph(N, k=int(np.mean([d for _, d in G.degree()])), p=0)
    while not nx.is_connected(G_lattice):  # Asegurar que la red sea conexa
        G_lattice = nx.watts_strogatz_graph(N, k=int(np.mean([d for _, d in G.degree()])), p=0)

    C_lattice = nx.average_clustering(G_lattice)
    L_lattice = nx.average_shortest_path_length(G_lattice)
    
    # Small-World Index (SWI)
    SWI = ((L - L_lattice) / (L_rand - L_lattice)) * ((C - C_rand) / (C_lattice - C_rand))

    # Small-World Measure (ω)
    omega = (L_rand / L) - (C / C_lattice)

    return {
        "Clustering Coefficient (C)": C,
        "Average Shortest Path Length (L)": L,
        "Small-World Index (SWI)": SWI,
        "Small-World Measure (ω)": omega
    }

# Calcular métricas corregidas para las redes
network_metrics_fixed = {name: calculate_network_properties_fixed(G) for name, G in networks.items()}

def print_network_metrics(metrics):
    print("\nMétricas de Redes:")
    print("-" * 50)
    for name, data in metrics.items():
        print(f"\n{name} Network:")
        for key, value in data.items():
            print(f"  {key}: {value:.4f}" if not pd.isna(value) else f"  {key}: NaN")
        print("-" * 50)

# Mostrar resultados corregidos en una tabla de texto
print_network_metrics(network_metrics_fixed)

