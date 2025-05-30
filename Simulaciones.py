import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# Parámetros del modelo
N = 100  # Número de nodos
beta = 0.3  # Probabilidad de infección
gamma = 0.1  # Probabilidad de recuperación
mu = 0.05  # Probabilidad de pérdida de inmunidad en SIRS
delta = 0.05  # Probabilidad de incubación en SEIR
steps = 500  # Número de pasos de simulación

# Crear redes
networks = {
    "Barabási-Albert": nx.barabasi_albert_graph(N, 2),
    "Watts-Strogatz": nx.watts_strogatz_graph(N, 4, 0.1),
    "Regular": nx.random_regular_graph(3, N)
}

# Graficar las redes
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (name, G) in zip(axes, networks.items()):
    nx.draw(G, ax=ax, node_size=30, edge_color="gray", with_labels=False)
    ax.set_title(name)
plt.show()

# Función para simular el modelo SIS
def simulate_SIS(G, beta, gamma, steps):
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
                        if states[neighbor] == 'S' and np.random.rand() < beta:
                            new_states[neighbor] = 'I'
        states = new_states
        num_infected.append(sum(1 for state in states.values() if state == 'I'))

    return num_infected

# Función para simular el modelo SIR
def simulate_SIR(G, beta, gamma, steps):
    states = {node: 'S' for node in G.nodes}
    initial_infected = np.random.choice(G.nodes, size=10, replace=False)
    for node in initial_infected:
        states[node] = 'I'
    num_infected = []

    for _ in range(steps):
        new_states = states.copy()
        for node in G.nodes:
            if states[node] == 'I':
                if np.random.rand() < gamma:
                    new_states[node] = 'R'
                else:
                    for neighbor in G.neighbors(node):
                        if states[neighbor] == 'S' and np.random.rand() < beta:
                            new_states[neighbor] = 'I'
        states = new_states
        num_infected.append(sum(1 for state in states.values() if state == 'I'))

    return num_infected

# Función para simular el modelo SIRS
def simulate_SIRS(G, beta, gamma, mu, steps):
    states = {node: 'S' for node in G.nodes}
    initial_infected = np.random.choice(G.nodes, size=10, replace=False)
    for node in initial_infected:
        states[node] = 'I'
    num_infected = []

    for _ in range(steps):
        new_states = states.copy()
        for node in G.nodes:
            if states[node] == 'I':
                if np.random.rand() < gamma:
                    new_states[node] = 'R'
                else:
                    for neighbor in G.neighbors(node):
                        if states[neighbor] == 'S' and np.random.rand() < beta:
                            new_states[neighbor] = 'I'
            elif states[node] == 'R' and np.random.rand() < mu:
                new_states[node] = 'S'
        states = new_states
        num_infected.append(sum(1 for state in states.values() if state == 'I'))

    return num_infected

# Función para simular el modelo SEIR
def simulate_SEIR(G, beta, gamma, delta, steps):
    states = {node: 'S' for node in G.nodes}
    initial_infected = np.random.choice(G.nodes, size=10, replace=False)
    for node in initial_infected:
        states[node] = 'E'
    num_infected = []

    for _ in range(steps):
        new_states = states.copy()
        for node in G.nodes:
            if states[node] == 'E':
                if np.random.rand() < delta:
                    new_states[node] = 'I'
            elif states[node] == 'I':
                if np.random.rand() < gamma:
                    new_states[node] = 'R'
                else:
                    for neighbor in G.neighbors(node):
                        if states[neighbor] == 'S' and np.random.rand() < beta:
                            new_states[neighbor] = 'E'
        states = new_states
        num_infected.append(sum(1 for state in states.values() if state == 'I'))

    return num_infected

# Simular el modelo SIS, SIR, SIRS y SEIR para cada red
results = {name: {
    "SIS": simulate_SIS(G, beta, gamma, steps),
    "SIR": simulate_SIR(G, beta, gamma, steps),
    "SIRS": simulate_SIRS(G, beta, gamma, mu, steps),
    "SEIR": simulate_SEIR(G, beta, gamma, delta, steps)
} for name, G in networks.items()}

# Graficar resultados de la simulación
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
models = ["SIS", "SIR", "SIRS", "SEIR"]
for ax, model in zip(axes.flatten(), models):
    for name, data in results.items():
        ax.plot(data[model], label=name)
    ax.set_xlabel("Tiempo")
    ax.set_ylabel("Número de infectados")
    ax.set_title(f"Modelo {model} en diferentes redes")
    ax.legend()
plt.tight_layout()
plt.show()
