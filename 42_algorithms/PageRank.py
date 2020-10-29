import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def PageRank(graph, alpha=0.9):
    
    n = len(graph)

    # Remove self loops
    graph[range(n), range(n)] = 0
    
    # Ensure stochasticity
    graph[:, graph.sum(0) == 0] = 1
    graph /= graph.sum(0)
    
    # Add random teleports
    graph = alpha * graph + (1 - alpha) / n * np.ones((n, n))

    # Power iteration
    eps = 1e-8
    prev = np.zeros(n)
    rank = prev + 1 / n
    while (rank - prev) @ (rank - prev) > eps:
        prev = rank
        rank = graph @ rank

    return rank


# Testing

n = 10
graph = nx.DiGraph()
graph.add_nodes_from(range(n))
graph.add_edges_from(np.random.randint(0, n, (3 * n, 2)))
nx.draw_networkx(graph, node_color='lightgreen')
plt.show()


ranks = PageRank(np.array(nx.adjacency_matrix(graph).todense(), dtype=np.float32))
print(ranks.round(2))

nx.draw_networkx(graph, node_color='lightgreen', node_size=ranks * 5000)
plt.show()