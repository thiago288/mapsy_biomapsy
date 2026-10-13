import numpy as np

def cluster_graph(clusters, neighbors):
    clusters = np.asarray(clusters, dtype=int)
    if not len(clusters):
        return np.zeros((0, 0), dtype=np.int64)
    if np.any(clusters < 0):
        raise ValueError("Cluster labels must be nonnegative")
    G = np.zeros((int(clusters.max()) + 1,) * 2, dtype=np.int64)
    if neighbors is None:
        return G
    if len(neighbors) != len(clusters):
        raise ValueError("Neighbor rows and cluster labels must have matching lengths")
    seen = set()
    for i, row in enumerate(neighbors):
        for j in row:
            j = int(j)
            if j < 0 or j == i:
                continue
            if j >= len(clusters):
                raise ValueError("Neighbor index is outside the contact space")
            edge = tuple(sorted((i, j)))
            if edge in seen:
                continue
            seen.add(edge)
            ci, cj = clusters[i], clusters[j]
            G[ci, cj] += 1
            if ci != cj:
                G[cj, ci] += 1
    return G
