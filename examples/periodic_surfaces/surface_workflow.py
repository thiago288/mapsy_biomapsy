"""Periodic Pt workflows adapted from Thiago's planar surface notebook.

Run from an installed MapSy checkout: python surface_workflow.py --surface 111.
The planar boundary is a center-of-mass referenced probe plane, not an
atom-conforming boundary. ASE periodic=True retains all three periodic axes.
"""
from pathlib import Path
import argparse
import json
import time
from copy import copy
import numpy as np
from ase import Atoms
from ase.build import fcc111, fcc100, fcc110
from mapsy.data import Grid, System
from mapsy.maps import Maps
from mapsy.io.parser import ContactSpaceGenerator, ContactSpaceModel
from mapsy.symfunc.input import SymmetryFunctionsModel
from mapsy.symfunc.parser import SymmetryFunctionsParser
from mapsy.utils import setscalars

def make_surface(face='111', size=None):
    """ASE-built ideal faces or the notebook's vacancy/O decoration recipe."""
    size = tuple(size or ((20, 20, 5) if face == 'defects' else (4, 4, 3)))
    build = {'111': fcc111, '100': fcc100, '110': fcc110, 'defects': fcc111}[str(face)]
    slab = build('Pt', size=size, a=3.94, orthogonal=True, periodic=True, vacuum=10)
    if face != 'defects':
        return slab
    p = slab.positions.copy()
    Lx, Ly = (slab.cell[0, 0], slab.cell[1, 1])
    top = np.sort(np.unique(np.round(p[:, 2], 1)))[-1]
    idx = np.where(abs(p[:, 2] - top) < 0.3)[0]
    s = p[idx]

    def circle(c, r):
        return np.linalg.norm(s[:, :2] - c, axis=1) < r

    def rect(x0, x1, y0, y1):
        return (s[:, 0] > x0) & (s[:, 0] < x1) & (s[:, 1] > y0) & (s[:, 1] < y1)
    vac = circle([0.25 * Lx, 0.25 * Ly], 0.06 * Lx) | circle([0.25 * Lx, 0.75 * Ly], 0.05 * Lx)
    ox = rect(0.65 * Lx,
        0.85 * Lx,
        0.2 * Ly,
        0.45 * Ly) | rect(0.6 * Lx,
        0.8 * Lx,
        0.6 * Ly,
        0.8 * Ly)
    keep = np.ones(len(slab), dtype=bool)
    keep[idx[vac]] = False
    base = slab[keep]
    opos = s[ox].copy()
    opos[:, 2] += 1.8
    return Atoms(list(base.symbols) + ['O'] * len(opos),
        positions=np.vstack([base.positions,
                opos]),
        cell=slab.cell,
        pbc=slab.pbc)

def make_functions(compositional=False, radius=4.5):
    settings = SymmetryFunctionsModel.parse_obj({'functions': [{'type': 'ac',
                    'radius': radius,
                    'order': 10,
                    'compositional': compositional,
                    'structural': True,
                    'radial': True},
                {'type': 'ac',
                    'radius': radius,
                    'order': 10,
                    'compositional': compositional,
                    'structural': True,
                    'radial': False}]})
    return SymmetryFunctionsParser(settings).parse()

def make_contact_space(atoms, cutoff=80, max_grid_points=15000000):
    """Grid cutoff follows MapSy's Ry convention; limit bounds this example."""
    scalars = setscalars(np.array(atoms.cell), float(cutoff))
    count = int(np.prod(scalars))
    if count > max_grid_points:
        raise ValueError(f'Example grid would contain {count:,} points. Select an explicitly coarser cutoff or smaller slab; this changes sampling. Limit={max_grid_points:,}.')
    system = System(Grid(cell=atoms.cell), atoms, dimension=2, axis=2)
    model = ContactSpaceModel.parse_obj({'mode': 'system',
            'distance': 3.5,
            'spread': 1.0,
            'cutoff': float(cutoff),
            'threshold': -1,
            'side': 1})
    cs = ContactSpaceGenerator(model).generate(system)
    return (system, cs)
# Seeds and cluster counts recovered from the original notebook's saved outputs.
ORIGINAL_CLUSTERING = {'111': (10, 450), '100': (10, 371), '110': (4, 361)}
PT110_SCREEN_SEEDS = [326, 199, 361, 557, 528, 31, 972, 917, 766, 840, 111, 498, 903]

def save_analysis(m, atoms, out, face, ntries=1):
    """Full-surface analysis; raw descriptors for clustering, standardized PCA.

    Both clustering metrics are evaluated on every point. No subsampling.
    Original notebook cluster counts remain separate from metric-selected counts.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import SpectralClustering
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    import pandas as pd
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        fig.savefig(out / (name + '.png'), dpi=160, bbox_inches='tight')
        fig.savefig(out / (name + '.pdf'), bbox_inches='tight')
        plt.close(fig)
    xy = m.data[['x', 'y']].to_numpy()
    pt = atoms.positions[np.asarray(atoms.symbols) == 'Pt']
    top = pt[np.abs(pt[:, 2] - pt[:, 2].max()) < 0.3]

    def overlay(ax):
        ax.scatter(top[:,
                0],
            top[:,
                1],
            facecolors='none',
            edgecolors='black',
            s=60,
            lw=0.8,
            label='Top-layer Pt',
            zorder=4)
        oxygen = atoms.positions[np.asarray(atoms.symbols) == 'O']
        if len(oxygen):
            ax.scatter(oxygen[:, 0], oxygen[:, 1], c='red', s=14, marker='x', label='O', zorder=5)
        ax.set(xlabel='x (Å)',
            ylabel='y (Å)',
            aspect='equal',
            xlim=(0,
                atoms.cell[0,
                    0]),
            ylim=(0,
                atoms.cell[1,
                    1]))

    def field(ax, key):
        # Display every contact point in its physical position; no interpolation.
        sc = ax.scatter(xy[:,
                0],
            xy[:,
                1],
            c=m.data[key],
            s=10,
            marker='s',
            cmap='viridis',
            rasterized=True)
        overlay(ax)
        ax.set_title(key, fontsize=9)
        return sc
    # Every descriptor is saved individually and in readable ten-panel sheets.
    for start in range(0, len(m.features), 10):
        keys = list(m.features[start:start + 10])
        fig, axes = plt.subplots(2, 5, figsize=(20, 8), layout='constrained')
        for ax, key in zip(axes.flat, keys):
            fig.colorbar(field(ax, key), ax=ax, shrink=0.7)
        fig.suptitle(f'Pt({face}) — descriptors {start + 1}–{start + len(keys)}')
        save(fig, f'descriptor_panel_{start // 10 + 1:02}')
        for key in keys:
            fig, ax = plt.subplots(figsize=(7, 6), layout='constrained')
            fig.colorbar(field(ax, key), ax=ax, label=key)
            save(fig, 'descriptor_' + key)
    X = m.data[list(m.features)].to_numpy(float)
    standardized = StandardScaler().fit_transform(X)
    pca = PCA(svd_solver='full').fit(standardized)
    pc = pca.transform(standardized)[:, :4]
    for j in range(4):
        m.data[f'pca{j}'] = pc[:, j]
    pd.DataFrame({'component': np.arange(1,
                len(pca.explained_variance_ratio_) + 1),
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_)}).to_csv(out / 'pca_variance.csv',
        index=False)
    pd.DataFrame(pca.components_[:4].T,
        index=m.features,
        columns=['PC1',
            'PC2',
            'PC3',
            'PC4']).to_csv(out / 'pca_loadings.csv')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    axes[0].plot(np.arange(1,
            len(pca.explained_variance_ratio_) + 1),
        pca.explained_variance_ratio_,
        'o-')
    axes[0].set(xlabel='Principal component', ylabel='Explained variance fraction')
    axes[1].plot(np.arange(1,
            len(pca.explained_variance_ratio_) + 1),
        np.cumsum(pca.explained_variance_ratio_),
        'o-')
    axes[1].set(xlabel='Number of components',
        ylabel='Cumulative variance fraction',
        ylim=(0,
            1.02))
    save(fig, 'pca_scree')
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), layout='constrained')
    for j, ax in enumerate(axes.flat):
        fig.colorbar(field(ax, f'pca{j}'), ax=ax, label=f'PC{j + 1}')
    save(fig, 'pca_spatial_maps')
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), layout='constrained')
    from itertools import combinations
    for ax, (i, j) in zip(axes.flat, combinations(range(4), 2)):
        sc = ax.scatter(pc[:,
                i],
            pc[:,
                j],
            c=X[:,
                0],
            s=6,
            alpha=0.5,
            cmap='viridis',
            rasterized=True)
        ax.set(xlabel=f'PC{i + 1}', ylabel=f'PC{j + 1}')
        fig.colorbar(sc, ax=ax, label=m.features[0])
    save(fig, 'pca_pairwise_descriptors')
    result = {'cluster_sample_points': len(X),
        'clustering_features': 'raw, unscaled descriptors',
        'pca_features': 'standardized descriptors',
        'screening_trials_per_k': ntries}
    # All-point weighted RBF spectral clustering exploits repeated environments.
    if face == 'defects':
        from decorated_clustering import save_decorated_clustering
        m.data.to_csv(out / 'descriptors.csv', index=False)
        evidence = save_decorated_clustering(m.data, atoms, out / 'clustering', ntries=ntries)
        result.update(clustering='Population-weighted RBF spectral quotient; all points',
            cluster_choices=evidence['choices'],
            clustering_evidence=evidence)
        return result
    rows = []
    solutions = {}
    rng = np.random.default_rng(20261009)
    original_k, original_seed = ORIGINAL_CLUSTERING[face]
    for k in range(2, 15):
        for trial in range(ntries):
            seed = int(PT110_SCREEN_SEEDS[k - 2] if face == '110' and trial == 0 else original_seed if k == original_k and trial == 0 else rng.integers(0,
                    1000))
            labels = SpectralClustering(n_clusters=k, random_state=seed).fit_predict(X)
            rows.append(dict(nclusters=k,
                    random_state=seed,
                    silhouette_score=float(silhouette_score(X,
                            labels)),
                    db_index=float(davies_bouldin_score(X,
                            labels))))
            solutions[k, seed] = labels
    table = pd.DataFrame(rows)
    table.to_csv(out / 'cluster_screening.csv', index=False)
    sil = table.loc[table.silhouette_score.idxmax()]
    db = table.loc[table.db_index.idxmin()]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
    for ax, key, title, best in [(axes[0],
            'silhouette_score',
            'Silhouette — higher is better',
            sil),
        (axes[1],
            'db_index',
            'Davies–Bouldin — lower is better',
            db)]:
        ax.scatter(table.nclusters, table[key])
        ax.plot(table.groupby('nclusters')[key].agg('max' if key == 'silhouette_score' else 'min'))
        ax.axvline(int(best.nclusters),
            color='green',
            ls='--',
            label=f'Best k = {int(best.nclusters)}')
        ax.axvline(original_k, color='black', ls=':', label=f'Original k = {original_k}')
        ax.set(xlabel='Number of clusters', ylabel=title)
        ax.legend(fontsize=9)
        ax.set_xticks(range(2, 15))
    save(fig, 'clustering_metrics')
    choices = {'original': (original_k,
            original_seed),
        'best_silhouette': (int(sil.nclusters),
            int(sil.random_state)),
        'best_davies_bouldin': (int(db.nclusters),
            int(db.random_state))}
    result['cluster_choices'] = {}
    for name, (k, seed) in choices.items():
        # Use MapSy's analysis API, including its physical-site and graph logic.
        m.cluster(nclusters=k, random_state=seed)
        m.sites()
        labels = m.data.Cluster.to_numpy()
        assert np.array_equal(labels, solutions[k, seed])
        m.data[f'Cluster_{name}'] = labels
        row = table[(table.nclusters == k) & (table.random_state == seed)].iloc[0]
        result['cluster_choices'][name] = {'nclusters': k,
            'random_state': seed,
            'silhouette_score': float(row.silhouette_score),
            'db_index': float(row.db_index)}
        cmap = plt.get_cmap('tab20', k)
        norm = BoundaryNorm(np.arange(k + 1) - 0.5, k)
        fig, ax = plt.subplots(figsize=(8, 7), layout='constrained')
        sc = ax.scatter(xy[:,
                0],
            xy[:,
                1],
            c=labels,
            s=18,
            marker='s',
            cmap=cmap,
            norm=norm,
            rasterized=True)
        overlay(ax)
        centers = m.data.loc[m.centroids, ['x', 'y']].to_numpy()
        ax.scatter(centers[:,
                0],
            centers[:,
                1],
            marker='*',
            c='white',
            edgecolors='black',
            s=180,
            zorder=6,
            label='Representative MapSy site')
        ax.legend(loc='upper left', bbox_to_anchor=(1.12, 1), fontsize=8)
        ax.set_title(f"Pt({face}) — {name.replace('_', ' ')}, k={k}")
        fig.colorbar(sc, ax=ax, ticks=range(k), label='Cluster (arbitrary label)')
        save(fig, 'clusters_xy_' + name)
        np.savetxt(out / ('cluster_edges_' + name + '.csv'),
            m.cluster_edges,
            delimiter=',',
            fmt='%d')
        m.data.loc[m.centroids,
            ['x',
                'y',
                'z',
                f'Cluster_{name}']].to_csv(out / ('representative_sites_' + name + '.csv'),
            index=False)
        fig, axes = plt.subplots(2, 3, figsize=(14, 9), layout='constrained')
        for ax, (i, j) in zip(axes.flat, combinations(range(4), 2)):
            ax.scatter(pc[:,
                    i],
                pc[:,
                    j],
                c=labels,
                cmap=cmap,
                norm=norm,
                s=7,
                alpha=0.5,
                rasterized=True)
            ax.set(xlabel=f'PC{i + 1}', ylabel=f'PC{j + 1}')
        fig.suptitle(f"Pt({face}) — {name.replace('_', ' ')}, k={k}")
        save(fig, 'pca_pairwise_clusters_' + name)
        if name == 'original':
            fig, ax = plt.subplots(figsize=(8, 6), layout='constrained')
            ax.scatter(pc[:,
                    0],
                pc[:,
                    1],
                c=labels,
                cmap=cmap,
                norm=norm,
                s=8,
                alpha=0.4,
                rasterized=True)
            sitepc = m.data.loc[m.centroids, ['pca0', 'pca1']].to_numpy()
            edges = np.asarray(m.cluster_edges)
            for a, b in zip(*np.where(np.triu(edges, 1) > 0)):
                ax.plot(sitepc[[a, b], 0], sitepc[[a, b], 1], c='black', alpha=0.3, lw=1)
            ax.scatter(sitepc[:,
                    0],
                sitepc[:,
                    1],
                c=np.arange(k),
                cmap=cmap,
                norm=norm,
                edgecolors='black',
                s=150,
                zorder=3)
            ax.set(xlabel='PC1',
                ylabel='PC2',
                title=f'Original k={k}: representative-site connectivity')
            save(fig, 'pca_cluster_graph_original')
    # Keep Cluster consistent with the original choice, not the last alternative.
    m.data['Cluster'] = m.data.Cluster_original
    m.data.to_csv(out / 'descriptors.csv', index=False)
    return result

def run(face='111', size=None, cutoff=80, use_gpu=False, output=None, ntries=1):
    if output is None:
        raise ValueError('Provide a new output directory for complete analysis')
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    atoms = make_surface(face, size)
    system, cs = make_contact_space(atoms, cutoff)
    m = Maps(system, make_functions(face == 'defects'), cs, use_gpu=use_gpu, batch_size=256)
    t = time.perf_counter()
    m.atcontactspace()
    seconds = time.perf_counter() - t
    if not np.isfinite(m.data[list(m.features)].to_numpy(dtype=float)).all():
        raise AssertionError('Non-finite descriptors')
    out.mkdir(parents=True, exist_ok=True)
    summary = {'face': face,
        'size': list(size or ((20,
                    20,
                    5) if face == 'defects' else (4,
                    4,
                    3))),
        'atoms': len(atoms),
        'pbc': atoms.pbc.tolist(),
        'grid_cutoff_ry': cutoff,
        'contact_points': len(cs.data),
        'features': list(m.features),
        'descriptor_seconds': seconds,
        'backend': m.last_backend,
        'top_atomic_z_angstrom': float(atoms.positions[:,
                2].max()),
        'probe_plane_z_range_angstrom': [float(cs.data.z.min()),
            float(cs.data.z.max())],
        'center_of_mass_z_angstrom': float(atoms.get_center_of_mass()[2])}
    summary.update(save_analysis(m, atoms, out, face, ntries))
    cs.data.drop(columns=['nn']).to_csv(out / 'contact_space.csv', index=False)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    return summary
if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--surface', choices=['111', '100', '110', 'defects'], default='111')
    p.add_argument('--cutoff', type=float, default=80)
    p.add_argument('--size', nargs=3, type=int)
    p.add_argument('--gpu', action='store_true')
    p.add_argument('--ntries', type=int, default=1)
    p.add_argument('--output', type=Path, default=Path('periodic_surface_results'))
    a = p.parse_args()
    if a.ntries < 1:
        p.error('--ntries must be positive')
    print(json.dumps(run(a.surface, a.size, a.cutoff, a.gpu, a.output, a.ntries), indent=2))
