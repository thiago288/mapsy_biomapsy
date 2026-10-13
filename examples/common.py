"""Shared tutorial plotting and analysis; this is example code, not MapSy core."""
from pathlib import Path
from datetime import datetime
import json
import os
import platform
import importlib.metadata
import numpy as np
import pandas as pd
from mapsy.symfunc.input import SymmetryFunctionsModel
from mapsy.symfunc.parser import SymmetryFunctionsParser
EXAMPLES = Path(__file__).resolve().parent

def new_output(label, output=None):
    path = Path(output) if output is not None else Path(os.environ.get('MAPSY_EXAMPLES_OUTPUT',
            EXAMPLES / 'results')) / (label + '_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f'Refusing to overwrite results: {path}')
    path.mkdir(parents=True, exist_ok=True)
    return path

def environment():
    import mapsy
    versions = {}
    for name in ['numpy',
        'scipy',
        'pandas',
        'ase',
        'scikit-learn',
        'pydantic',
        'matplotlib',
        'cupy-cuda12x',
        'cupy']:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return dict(python=platform.python_version(),
        platform=platform.platform(),
        mapsy_module=str(Path(mapsy.__file__).resolve()),
        versions=versions,
        thread_environment={k: os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS',
                'OMP_NUM_THREADS',
                'MKL_NUM_THREADS']})

def make_functions(radius=4.5, order=10, compositional=False, descriptor_set='mixed'):
    if descriptor_set not in ('mixed', 'radial'):
        raise ValueError('descriptor_set must be mixed or radial')
    functions = [dict(type='ac',
            radius=radius,
            order=order,
            structural=True,
            compositional=compositional,
            radial=True)]
    if descriptor_set == 'mixed':
        functions.append(dict(type='ac',
                radius=radius,
                order=order,
                structural=True,
                compositional=compositional,
                radial=False))
    return SymmetryFunctionsParser(SymmetryFunctionsModel.parse_obj({'functions': functions})).parse()

def feature_columns(data):
    return [c for c in data if c.startswith('ACSF_') or c in ('electrostatic', 'hydrophobicity')]

def plot_protein_fields(data, keys, out, name, title, *, limits=None):
    """Opposite 3D views, no depth-shading; display subset is explicitly recorded."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    positions = data[['x', 'y', 'z']].to_numpy(float)
    selected = np.sort(np.random.default_rng(42).choice(len(data),
            min(12000,
                len(data)),
            replace=False))
    fig = plt.figure(figsize=(5 * len(keys), 10), layout='constrained')
    evidence = {}
    for j, key in enumerate(keys):
        values = data[key].to_numpy(float)
        low, high = limits[key] if limits else np.quantile(values, [0.02, 0.98])
        if high <= low:
            low, high = (float(values.min()) - 1e-08, float(values.max()) + 1e-08)
        norm = Normalize(low, high)
        evidence[key] = dict(color_min=float(low),
            color_max=float(high),
            full_min=float(values.min()),
            full_max=float(values.max()),
            std=float(values.std()))
        for view, azimuth in enumerate([-65, 115]):
            ax = fig.add_subplot(2, len(keys), view * len(keys) + j + 1, projection='3d')
            image = ax.scatter(*positions[selected].T,
                c=values[selected],
                s=2,
                cmap='coolwarm' if key.startswith('PC') or key == 'electrostatic' else 'viridis',
                norm=norm,
                depthshade=False,
                rasterized=True)
            ax.view_init(elev=20, azim=azimuth)
            ax.set_box_aspect(np.ptp(positions, axis=0))
            ax.set(xlabel='x (Å)', ylabel='y (Å)', zlabel='z (Å)', title=f'{key}: view {view + 1}')
            fig.colorbar(image, ax=ax, shrink=0.55, extend='both')
    fig.suptitle(title + '\nColors use 2–98% limits; opposite views; up to 12,000 displayed points',
        fontsize=12)
    out = Path(out)
    fig.savefig(out / (name + '.png'), dpi=150, bbox_inches='tight')
    fig.savefig(out / (name + '.pdf'), bbox_inches='tight')
    plt.close(fig)
    (out / (name + '_display.json')).write_text(json.dumps(dict(displayed_points=len(selected),
                total_points=len(data),
                display_seed=42,
                color_limits=evidence),
            indent=2))

def analyze(data,
    atoms,
    output,
    *,
    periodic,
    method='spectral',
    ks=tuple(range(2,
            15)),
    reference_k=None,
    ntries=2,
    title='MapSy',
    features=None):
    """Full-data fits, an explicit K sweep, and median-over-seeds metric choices.

    All rows enter PCA and clustering. Silhouette uses all <=6000 rows, otherwise
    a recorded deterministic 2000-point subset. Plotting subsets never alter fits.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm, Normalize
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import SpectralClustering, KMeans
    from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
    from itertools import combinations
    out = Path(output)
    data = data.copy()
    features = list(features or feature_columns(data))
    X = data[features].to_numpy(float)
    if len(X) < 15 or not np.isfinite(X).all():
        raise ValueError('Need at least 15 finite descriptor rows')
    if method not in ('spectral', 'kmeans'):
        raise ValueError(method)
    if method == 'spectral' and len(X) > 6000:
        raise ValueError('Tutorial dense spectral guard: 6000 points. Choose an explicit scalable analysis; no automatic subsampling.')
    if ntries < 1:
        raise ValueError('ntries must be positive')
    statistics = data[features].describe(percentiles=[0.02, 0.5, 0.98]).T
    statistics.to_csv(out / 'feature_summary.csv')
    scaled = StandardScaler().fit_transform(X)
    pca = PCA(svd_solver='full').fit(scaled)
    pcs = pca.transform(scaled)[:, :4]
    for j in range(pcs.shape[1]):
        data[f'PC{j + 1}'] = pcs[:, j]
    variance = pd.DataFrame(dict(component=np.arange(1,
                len(pca.explained_variance_ratio_) + 1),
            explained_variance=pca.explained_variance_ratio_,
            cumulative_variance=np.cumsum(pca.explained_variance_ratio_)))
    variance.to_csv(out / 'pca_variance.csv', index=False)
    pd.DataFrame(pca.components_[:4].T,
        index=features,
        columns=[f'PC{j + 1}' for j in range(pcs.shape[1])]).to_csv(out / 'pca_loadings.csv')

    def save(fig, name):
        previous = fig._suptitle.get_text() if fig._suptitle is not None else ''
        fig.suptitle(title + (' — ' + previous if previous else ''), fontsize=12)
        fig.savefig(out / (name + '.png'), dpi=150, bbox_inches='tight')
        fig.savefig(out / (name + '.pdf'), bbox_inches='tight')
        plt.close(fig)
    xy = data[['x', 'y']].to_numpy()
    symbols = np.array(atoms.get_chemical_symbols())
    pt = atoms.positions[symbols == 'Pt']
    oxygen = atoms.positions[symbols == 'O']
    pt = pt[abs(pt[:, 2] - pt[:, 2].max()) < 0.3] if len(pt) else pt
    # Explicit front-facing projection avoids superposing all protein depths.
    visible = np.arange(len(data)) if periodic else data.reset_index(drop=True).groupby(['x',
            'y'])['z'].idxmax().to_numpy()

    def field(ax, values, cmap=None, norm=None, robust=False):
        values = np.asarray(values)
        cmap = cmap or 'viridis'
        if robust and norm is None:
            lo, hi = np.quantile(values, [0.02, 0.98])
            norm = Normalize(float(lo), float(hi) if hi > lo else float(lo + 1e-08))
        if periodic and (not data.duplicated(['x', 'y']).any()):
            table = pd.DataFrame(dict(x=xy[:,
                        0],
                    y=xy[:,
                        1],
                    value=values)).pivot(index='y',
                columns='x',
                values='value')
            image = ax.pcolormesh(table.columns,
                table.index,
                table.to_numpy(),
                shading='nearest',
                cmap=cmap,
                norm=norm,
                rasterized=True)
        else:
            image = ax.scatter(xy[visible,
                    0],
                xy[visible,
                    1],
                c=values[visible],
                cmap=cmap,
                norm=norm,
                s=9,
                marker='s',
                rasterized=True)
        if periodic:
            if len(pt):
                ax.scatter(pt[:,
                        0],
                    pt[:,
                        1],
                    s=18,
                    facecolors='none',
                    edgecolors='black',
                    linewidths=0.4,
                    label='Top-layer Pt',
                    zorder=4)
            if len(oxygen):
                ax.scatter(oxygen[:,
                        0],
                    oxygen[:,
                        1],
                    c='black',
                    marker='x',
                    s=20,
                    label='O',
                    zorder=5)
        ax.set(xlabel='x (Å)', ylabel='y (Å)', aspect='equal')
        return image
    for start in range(0, len(features), 10):
        fig, axes = plt.subplots(2, 5, figsize=(20, 8), layout='constrained')
        keys = features[start:start + 10]
        for ax, key in zip(axes.flat, keys):
            fig.colorbar(field(ax,
                    data[key],
                    robust=not periodic),
                ax=ax,
                shrink=0.7,
                extend='both' if not periodic else 'neither')
            ax.set_title(key, fontsize=8)
        for ax in list(axes.flat)[len(keys):]:
            ax.set_visible(False)
        save(fig, f'descriptor_panel_{start // 10 + 1:02d}')
    # A compact, consistently selected descriptor gallery for every system.
    groups = {name: [c for c in features if c.startswith('ACSF_' + name)] for name in ['RS',
            'RC',
            'AS',
            'AC']}
    chosen = []
    for group in groups.values():
        if group:
            chosen.extend([group[0], group[len(group) // 2]])
    chosen = [key for key in ['electrostatic',
            'hydrophobicity'] if key in features] + [group[0] for group in groups.values() if group] + [group[len(group) // 2] for group in groups.values() if group]
    chosen = list(dict.fromkeys(chosen))[:6]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), layout='constrained')
    for ax, key in zip(axes.flat, chosen):
        fig.colorbar(field(ax,
                data[key],
                robust=not periodic),
            ax=ax,
            extend='both' if not periodic else 'neither')
        ax.set_title(key, fontsize=9)
    for ax in list(axes.flat)[len(chosen):]:
        ax.set_visible(False)
    save(fig, 'descriptor_overview')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    axes[0].plot(variance.component, variance.explained_variance, 'o-')
    axes[1].plot(variance.component, variance.cumulative_variance, 'o-')
    axes[0].set(xlabel='Principal component', ylabel='Explained variance')
    axes[1].set(xlabel='Principal component', ylabel='Cumulative variance')
    save(fig, 'pca_scree')
    fig, ax = plt.subplots(figsize=(8, max(6, 0.24 * len(features))), layout='constrained')
    image = ax.imshow(pca.components_[:4].T, cmap='coolwarm', aspect='auto')
    ax.set(xticks=range(4),
        xticklabels=['PC1',
            'PC2',
            'PC3',
            'PC4'],
        yticks=range(len(features)),
        yticklabels=features,
        title='Standardized PCA loadings')
    fig.colorbar(image, ax=ax, label='Loading')
    save(fig, 'pca_loadings')
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), layout='constrained')
    for j, ax in enumerate(axes.flat):
        fig.colorbar(field(ax,
                pcs[:,
                    j],
                cmap='coolwarm',
                robust=not periodic),
            ax=ax,
            extend='both' if not periodic else 'neither')
        ax.set_title(f'PC{j + 1} ({pca.explained_variance_ratio_[j]:.1%})')
    if not periodic:
        fig.suptitle('Front-facing projection; 2–98% color limits')
    save(fig, 'pca_spatial_maps')
    if not periodic:
        plot_protein_fields(data,
            [f'PC{j + 1}' for j in range(4)],
            out,
            'pca_3d_views',
            title + ' — standardized PCA')
        if 'electrostatic' in features:
            plot_protein_fields(data,
                ['electrostatic',
                    'hydrophobicity'],
                out,
                'physicochemical_3d_views',
                title + ' — notebook-derived fields')
    color_key = 'hydrophobicity' if 'hydrophobicity' in features else next((c for c in features if c.startswith('ACSF_RS')))
    colorvalues = data[color_key].to_numpy(float)
    lo, hi = np.quantile(colorvalues,
        [0.02,
            0.98]) if not periodic else (colorvalues.min(),
        colorvalues.max())
    norm = Normalize(lo, hi if hi > lo else lo + 1e-08)
    scatter_indices = np.sort(np.random.default_rng(42).choice(len(data),
            min(12000,
                len(data)),
            replace=False))
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), layout='constrained')
    for ax, (i, j) in zip(axes.flat, combinations(range(4), 2)):
        image = ax.scatter(pcs[scatter_indices,
                i],
            pcs[scatter_indices,
                j],
            c=colorvalues[scatter_indices],
            cmap='viridis',
            norm=norm,
            s=3,
            alpha=0.7,
            rasterized=True)
        ax.set(xlabel=f'PC{i + 1}', ylabel=f'PC{j + 1}')
        fig.colorbar(image, ax=ax, label=color_key, shrink=0.7)
    save(fig, 'pca_pairwise_descriptors')
    V = X if method == 'spectral' else scaled
    sample = np.arange(len(X)) if len(X) <= 6000 else np.sort(np.random.default_rng(42).choice(len(X),
            2000,
            replace=False))
    seeds = ([42, 371] + [1000 + i for i in range(max(0, ntries - 2))])[:ntries]

    def fit(k, seed):
        return (SpectralClustering(n_clusters=k,
                gamma=1,
                random_state=seed) if method == 'spectral' else KMeans(n_clusters=k,
                n_init=10,
                random_state=seed)).fit_predict(V)
    rows = []
    solutions = {}
    counts = sorted(set(ks) | ({reference_k} if reference_k is not None else set()))
    for k in counts:
        for seed in seeds:
            labels = fit(k, seed)
            labelsample = labels[sample]
            nc = len(np.unique(labelsample))
            score = float(silhouette_score(V[sample],
                    labelsample)) if 1 < nc < len(sample) else float('nan')
            rows.append(dict(nclusters=k,
                    random_state=seed,
                    silhouette_score=score,
                    db_index=float(davies_bouldin_score(V,
                            labels)),
                    largest_cluster_fraction=float(np.bincount(labels).max() / len(X)),
                    silhouette_points=len(sample),
                    fit_points=len(X)))
            solutions[k, seed] = labels
        if len(seeds) > 1:
            ari = float(adjusted_rand_score(solutions[k, seeds[0]], solutions[k, seeds[1]]))
            for row in rows[-len(seeds):]:
                row['seed_ARI'] = ari
    table = pd.DataFrame(rows)
    table.to_csv(out / 'clustering_metrics.csv', index=False)
    medians = table.groupby('nclusters')[['silhouette_score', 'db_index']].median()
    medians.to_csv(out / 'clustering_medians.csv')
    sil_k = int(medians.silhouette_score.idxmax())
    db_k = int(medians.db_index.idxmin())
    choices = {'best_silhouette': (sil_k, seeds[0]), 'best_davies_bouldin': (db_k, seeds[0])}
    if reference_k is not None:
        choices['comparison'] = (reference_k, seeds[0])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
    for ax, key, label, best in [(axes[0],
            'silhouette_score',
            'Silhouette (higher is better)',
            sil_k),
        (axes[1],
            'db_index',
            'Davies–Bouldin (lower is better)',
            db_k)]:
        grouped = table.groupby('nclusters')[key]
        ax.scatter(table.nclusters, table[key], s=15, alpha=0.6, label='Individual seeds')
        ax.plot(medians.index, medians[key], 'o-', label='Median over seeds')
        ax.fill_between(medians.index,
            grouped.min(),
            grouped.max(),
            alpha=0.15,
            label='Observed seed range')
        ax.axvline(best, color='green', ls='--', label=f'Metric-selected k={best}')
        ax.set(xlabel='Number of clusters', ylabel=label, xticks=counts)
        ax.legend(fontsize=8)
    save(fig, 'clustering_metrics')
    palette = list(plt.get_cmap('tab10').colors) + list(plt.get_cmap('tab20').colors)[1::2]
    choice_evidence = {}
    for name, (k, seed) in choices.items():
        caption = 'Selected by silhouette and Davies–Bouldin' if name in ('best_silhouette',
            'best_davies_bouldin') and choices['best_silhouette'] == choices['best_davies_bouldin'] else 'Comparison resolution' if name == 'comparison' else 'Selected by ' + ('silhouette' if name == 'best_silhouette' else 'Davies–Bouldin')
        labels = solutions[k, seed]
        data['Cluster_' + name] = labels
        cmap = ListedColormap(palette[:k])
        norm = BoundaryNorm(np.arange(k + 1) - 0.5, k)
        row = table[(table.nclusters == k) & (table.random_state == seed)].iloc[0]
        choice_evidence[name] = dict(nclusters=k,
            random_state=seed,
            median_silhouette=float(medians.loc[k,
                    'silhouette_score']),
            median_db=float(medians.loc[k,
                    'db_index']),
            silhouette_this_seed=float(row.silhouette_score),
            db_this_seed=float(row.db_index),
            cluster_sizes=np.bincount(labels).tolist())
        fig, ax = plt.subplots(figsize=(9, 7), layout='constrained')
        image = field(ax, labels, cmap, norm)
        fig.colorbar(image, ax=ax, ticks=range(k), label='Cluster (arbitrary label)')
        ax.set_title(f'{method}, {caption}, k={k}')
        if periodic and len(pt):
            ax.legend(loc='upper right', fontsize=8)
        save(fig, 'clusters_xy_' + name)
        fig, axes = plt.subplots(2, 3, figsize=(14, 9), layout='constrained')
        for ax, (i, j) in zip(axes.flat, combinations(range(4), 2)):
            ax.scatter(pcs[scatter_indices,
                    i],
                pcs[scatter_indices,
                    j],
                c=labels[scatter_indices],
                cmap=cmap,
                norm=norm,
                s=3,
                alpha=0.5,
                rasterized=True)
            ax.set(xlabel=f'PC{i + 1}', ylabel=f'PC{j + 1}')
        fig.suptitle(f'{caption}, k={k}')
        save(fig, 'pca_pairwise_clusters_' + name)
        if not periodic:
            positions = data[['x', 'y', 'z']].to_numpy(float)
            fig = plt.figure(figsize=(13, 6), layout='constrained')
            for view, azimuth in enumerate([-65, 115]):
                ax = fig.add_subplot(1, 2, view + 1, projection='3d')
                image = ax.scatter(*positions[scatter_indices].T,
                    c=labels[scatter_indices],
                    cmap=cmap,
                    norm=norm,
                    s=2,
                    depthshade=False,
                    rasterized=True)
                ax.view_init(elev=20, azim=azimuth)
                ax.set_box_aspect(np.ptp(positions, axis=0))
                ax.set(xlabel='x (Å)',
                    ylabel='y (Å)',
                    zlabel='z (Å)',
                    title=f'{caption}, k={k}, view {view + 1}')
                fig.colorbar(image, ax=ax, ticks=range(k), shrink=0.6)
            save(fig, 'clusters_3d_' + name)
    data.to_csv(out / 'descriptors.csv', index=False)
    summary = dict(title=title,
        points=len(X),
        features=features,
        clustering_method=method,
        clustering_space='raw' if method == 'spectral' else 'standardized',
        pca_space='standardized',
        cluster_choices=choice_evidence,
        selection_rule='Maximum median silhouette / minimum median DB across seeds; representative plotted seed is the first seed',
        screening_ks=counts,
        screening_seeds=seeds,
        silhouette_points=len(sample),
        silhouette_sampling='all rows' if len(sample) == len(X) else '2000 uniformly selected rows, seed 42',
        db_points=len(X),
        pca_pair_display_points=len(scatter_indices),
        protein_projected_front_points=len(visible) if not periodic else None,
        physical_validation='Cluster counts and assignments require validation for the intended scientific application')
    (out / 'analysis.json').write_text(json.dumps(summary, indent=2))
    return summary

def show_analysis(summary):
    """Notebook presentation: parameters -> descriptors -> PCA -> K -> partitions."""
    from IPython.display import display, Image, Markdown
    folder = Path(summary['output'])
    display(Markdown('## ' + summary['title']))
    parameters = {k: summary[k] for k in ['atoms',
            'points',
            'radius_angstrom',
            'orders',
            'source_state',
            'contact_mode',
            'reference',
            'requested_clearance_angstrom',
            'actual_probe_z',
            'backend'] if k in summary}
    display(pd.DataFrame([parameters]))
    display(Markdown('### 1. Descriptors\nEvery coefficient is saved; the compact overview selects several families.'))
    for name in ['descriptor_overview'] + [f'descriptor_panel_{i + 1:02d}' for i in range((len(summary['features']) + 9) // 10)]:
        display(Image(filename=str(folder / (name + '.png'))))
    if (folder / 'physicochemical_3d_views.png').exists():
        display(Image(filename=str(folder / 'physicochemical_3d_views.png')))
    display(Markdown('### 2. PCA\nStandardized inputs; loadings explain what each component combines. Protein views use 2–98% color limits and opposite 3D views.'))
    for name in ['pca_scree',
        'pca_loadings',
        'pca_3d_views' if (folder / 'pca_3d_views.png').exists() else 'pca_spatial_maps',
        'pca_pairwise_descriptors']:
        display(Image(filename=str(folder / (name + '.png'))))
    display(Markdown('### 3. Number of clusters\nSweep k=2–14, two seeds by default. Select each metric using its median over seeds; no fixed k=10 is imposed.'))
    display(Image(filename=str(folder / 'clustering_metrics.png')))
    display(pd.read_csv(folder / 'clustering_medians.csv'))
    display(pd.DataFrame(summary['cluster_choices']).T)
    display(Markdown('### 4. Selected partitions\nSilhouette and Davies–Bouldin can disagree. Cluster counts and assignments require validation for the intended scientific application.'))
    shown = set()
    for choice, info in summary['cluster_choices'].items():
        signature = (info['nclusters'], info['random_state'])
        if signature in shown:
            continue
        shown.add(signature)
        for name in ['clusters_3d_' + choice if (folder / ('clusters_3d_' + choice + '.png')).exists() else 'clusters_xy_' + choice,
            'pca_pairwise_clusters_' + choice]:
            display(Image(filename=str(folder / (name + '.png'))))
