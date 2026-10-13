"""Population-weighted RBF spectral analysis of repeated periodic descriptors.

No contact points are sampled. Rounding groups near-identical descriptors;
representatives retain their original values. The quotient accounts for every
point's multiplicity and removes self-edges exactly as sklearn's Laplacian does.
"""
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances

def quotient_embedding(X, maxclusters=14, decimals=12):
    """Return group embedding, membership and population; raw RBF gamma=1."""
    X = np.asarray(X, dtype=float)
    _, first, inverse, counts = np.unique(np.round(X,
            decimals),
        axis=0,
        return_index=True,
        return_inverse=True,
        return_counts=True)
    U = X[first]
    w = counts.astype(float)
    maxdiff = float(np.max(np.abs(X - U[inverse])))
    if len(U) > 8000:
        raise ValueError(f'{len(U)} distinct environments exceed the 8000-group memory guard; no automatic sampling or precision reduction')
    affinity = rbf_kernel(U, gamma=1.0)
    degree = affinity @ w - 1.0
    if np.any(degree <= 0):
        raise ValueError('Zero-degree RBF graph')
    weight = np.sqrt(w / degree)
    affinity *= weight[:, None]
    affinity *= weight[None, :]
    affinity.flat[::len(U) + 1] -= 1.0 / degree
    # Largest normalized adjacency eigenvectors = smallest Laplacian vectors.
    count = min(maxclusters, len(U) - 1)
    vals, vectors = eigsh(affinity,
        k=count,
        which='LA',
        tol=1e-10,
        v0=np.random.default_rng(20261009).normal(size=len(U)))
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vectors = vectors[:, order]
    residual = float(np.max(np.linalg.norm(affinity @ vectors - vectors * vals, axis=0)))
    if vals[-1] <= 0:
        raise ValueError('Requested eigenvectors intersect omitted within-group modes; quotient cannot reproduce this spectral subspace')
    # Lift group eigenvectors to all repeated vertices, then undo degree scaling
    # as in sklearn spectral_embedding(normalized_laplacian, drop_first=False).
    embedding = vectors / np.sqrt(w * degree)[:, None]
    evidence = {'all_points': len(X),
        'descriptor_features': X.shape[1],
        'groups': len(U),
        'rounding_decimals': decimals,
        'max_group_descriptor_difference': maxdiff,
        'dense_affinity_estimate_bytes': len(X) ** 2 * 8,
        'quotient_affinity_bytes': len(U) ** 2 * 8,
        'gamma': 1.0,
        'diagonal_self_edges_removed': True,
        'maximum_eigenvector_residual': residual,
        'adjacency_eigenvalues': vals.tolist()}
    return (embedding, inverse, counts, U, evidence)

def weighted_silhouette(distances, group_labels, counts):
    """Exact all-point silhouette for repeated representative descriptors."""
    labels = np.asarray(group_labels)
    k = int(labels.max() + 1)
    sizes = np.bincount(labels, weights=counts, minlength=k)
    weights = np.zeros((len(counts), k))
    weights[np.arange(len(counts)), labels] = counts
    sums = distances @ weights
    within = sums[np.arange(len(counts)), labels] / np.maximum(sizes[labels] - 1, 1)
    means = sums / sizes
    means[np.arange(len(counts)), labels] = np.inf
    between = means.min(axis=1)
    values = (between - within) / np.maximum(within, between)
    values[sizes[labels] == 1] = 0
    return float(np.average(np.nan_to_num(values), weights=counts))

def analyze(X, ntries=1):
    t = time.perf_counter()
    embedding, inverse, counts, U, evidence = quotient_embedding(X)
    distances = euclidean_distances(U)
    np.fill_diagonal(distances, 0)
    rows = []
    solutions = {}
    rng = np.random.default_rng(20261009)
    for k in range(2, 15):
        for trial in range(ntries):
            seed = int(rng.integers(0, 1000))
            # Full expanded KMeans retains sklearn's population weighting and
            # initialization behavior; a 81k × 14 embedding is small.
            labels = KMeans(n_clusters=k,
                n_init=10,
                random_state=seed).fit_predict(embedding[inverse,
                    :k])
            group_labels = labels[np.unique(inverse, return_index=True)[1]]
            if not np.array_equal(labels, group_labels[inverse]):
                raise AssertionError('KMeans split identical descriptor groups')
            rows.append({'nclusters': k,
                    'random_state': seed,
                    'silhouette_score': weighted_silhouette(distances,
                        group_labels,
                        counts),
                    'db_index': float(davies_bouldin_score(X,
                            labels))})
            solutions[k, seed] = labels
        print('SCREEN', k, rows[-1], flush=True)
    evidence['clustering_seconds'] = time.perf_counter() - t
    evidence['silhouette_definition'] = 'Population-weighted all-point score on grouped representative descriptors; no sample'
    evidence['db_definition'] = 'Exact score on all original descriptor rows'
    return (pd.DataFrame(rows), solutions, evidence)

def save_decorated_clustering(data, atoms, output, ntries=1):
    """Add a complete clustering result in a new directory; preserve descriptors."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from itertools import combinations
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    data = data.copy()
    features = [c for c in data if c.startswith('ACSF_')]
    X = data[features].to_numpy(float)
    table, solutions, evidence = analyze(X, ntries)
    table.to_csv(out / 'cluster_screening.csv', index=False)
    sil = table.loc[table.silhouette_score.idxmax()]
    db = table.loc[table.db_index.idxmin()]
    fixed = table[table.nclusters == 10].sort_values('db_index').iloc[0]
    choices = {'best_silhouette': sil, 'best_davies_bouldin': db, 'reference_k10': fixed}

    def save(fig, name):
        fig.savefig(out / (name + '.png'), dpi=170, bbox_inches='tight')
        fig.savefig(out / (name + '.pdf'), bbox_inches='tight')
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
    for ax, key, best, title in [(axes[0],
            'silhouette_score',
            sil,
            'Silhouette — higher is better'),
        (axes[1],
            'db_index',
            db,
            'Davies–Bouldin — lower is better')]:
        ax.scatter(table.nclusters, table[key])
        ax.plot(table.groupby('nclusters')[key].agg('max' if key == 'silhouette_score' else 'min'))
        ax.axvline(int(best.nclusters),
            color='green',
            ls='--',
            label=f'Best k = {int(best.nclusters)}')
        ax.set(xlabel='Number of clusters', ylabel=title)
        ax.set_xticks(range(2, 15))
        ax.legend()
    save(fig, 'clustering_metrics')
    pt = atoms.positions[np.asarray(atoms.symbols) == 'Pt']
    top = pt[abs(pt[:, 2] - pt[:, 2].max()) < 0.3]
    oxygen = atoms.positions[np.asarray(atoms.symbols) == 'O']
    pc = data[['pca0', 'pca1', 'pca2', 'pca3']].to_numpy()
    evidence['choices'] = {}
    for name, row in choices.items():
        k, seed = (int(row.nclusters), int(row.random_state))
        labels = solutions[k, seed]
        data['Cluster_' + name] = labels
        cmap = ListedColormap(plt.get_cmap('tab20').colors[:k])
        norm = BoundaryNorm(np.arange(k + 1) - 0.5, k)
        evidence['choices'][name] = {key: int(row[key]) if key in ['nclusters',
                'random_state'] else float(row[key]) for key in table.columns}
        evidence['choices'][name]['cluster_sizes'] = np.bincount(labels, minlength=k).tolist()
        fig, ax = plt.subplots(figsize=(10, 8), layout='constrained')
        sc = ax.scatter(data.x,
            data.y,
            c=labels,
            s=4,
            marker='s',
            cmap=cmap,
            norm=norm,
            rasterized=True)
        ax.scatter(top[:,
                0],
            top[:,
                1],
            s=18,
            facecolors='none',
            edgecolors='black',
            lw=0.5,
            label='Top-layer Pt')
        ax.scatter(oxygen[:, 0], oxygen[:, 1], s=35, c='red', marker='x', label='O')
        ax.set(xlabel='x (Å)',
            ylabel='y (Å)',
            aspect='equal',
            xlim=(0,
                atoms.cell[0,
                    0]),
            ylim=(0,
                atoms.cell[1,
                    1]),
            title=f'Pt(111) decorated — {name.replace('_',
                    ' ')}, k={k}')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13), ncol=2)
        fig.colorbar(sc, ax=ax, ticks=range(k), label='Cluster label', shrink=0.75)
        save(fig, 'clusters_xy_' + name)
        # A smaller view makes the two vacancy circles and O patches readable.
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
        for ax, (xlim,
            ylim,
            title) in zip(axes,
            [((0.15 * atoms.cell[0,
                            0],
                        0.36 * atoms.cell[0,
                            0]),
                    (0.12 * atoms.cell[1,
                            1],
                        0.87 * atoms.cell[1,
                            1]),
                    'Vacancy patches'),
                ((0.55 * atoms.cell[0,
                            0],
                        0.9 * atoms.cell[0,
                            0]),
                    (0.15 * atoms.cell[1,
                            1],
                        0.85 * atoms.cell[1,
                            1]),
                    'O decorations')]):
            ax.scatter(data.x,
                data.y,
                c=labels,
                s=8,
                marker='s',
                cmap=cmap,
                norm=norm,
                rasterized=True)
            ax.scatter(top[:, 0], top[:, 1], s=35, facecolors='none', edgecolors='black', lw=0.6)
            ax.scatter(oxygen[:, 0], oxygen[:, 1], s=45, c='red', marker='x')
            ax.set(xlim=xlim,
                ylim=ylim,
                xlabel='x (Å)',
                ylabel='y (Å)',
                aspect='equal',
                title=title)
        save(fig, 'clusters_xy_details_' + name)
        fig, axes = plt.subplots(2, 3, figsize=(14, 9), layout='constrained')
        for ax, (i, j) in zip(axes.flat, combinations(range(4), 2)):
            ax.scatter(pc[:,
                    i],
                pc[:,
                    j],
                c=labels,
                cmap=cmap,
                norm=norm,
                s=2,
                alpha=0.2,
                rasterized=True)
            ax.set(xlabel=f'PC{i + 1}', ylabel=f'PC{j + 1}')
        fig.suptitle(f'Decorated Pt(111): {name.replace('_', ' ')}, k={k}')
        save(fig, 'pca_pairwise_clusters_' + name)
    data['Cluster'] = data.Cluster_best_davies_bouldin
    data.to_csv(out / 'descriptors_with_clusters.csv', index=False)
    (out / 'summary.json').write_text(json.dumps(evidence, indent=2))
    return evidence
