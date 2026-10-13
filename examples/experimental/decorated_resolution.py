"""Compare decorated-surface partitions using descriptor reconstruction error.

Run from the repository root:
    python examples/experimental/decorated_resolution.py
A saved descriptor CSV can be supplied with --descriptors. Default calculations
use the same structure, probe plane and descriptor settings as the tutorial.
"""
from pathlib import Path
from itertools import combinations
import argparse
import hashlib
import json
import sys
import time

import numpy as np
import pandas as pd
from ase.build import fcc111
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

EXAMPLES = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLES / 'periodic_surfaces'))
from structures import make_surface


def retention(values, labels, centers=None):
    """Return the fraction of each channel's variance retained by the labels.

    When centers are omitted, use conditional means within the supplied rows.
    Constant channels have no variance to resolve and are returned as NaN.
    """
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if centers is None:
        centers = np.zeros((labels.max() + 1, values.shape[1]))
        np.add.at(centers, labels, values)
        counts = np.bincount(labels, minlength=len(centers))
        centers /= np.maximum(counts, 1)[:, None]
    total = np.sum((values - values.mean(axis=0)) ** 2, axis=0)
    residual = np.sum((values - centers[labels]) ** 2, axis=0)
    result = np.full(values.shape[1], np.nan)
    keep = total > len(values) * 1e-20
    result[keep] = 1 - residual[keep] / total[keep]
    return result


def geometry(data, atoms, radius):
    """Mark points whose cutoff sphere intersects O or a missing top-layer Pt.

    Vacancy positions are recovered by comparing the decorated fixture with
    its ideal parent. These masks diagnose the result; they do not enter fits.
    """
    box = atoms.cell.lengths()[:2]
    symbols = np.array(atoms.get_chemical_symbols())
    pt = atoms.positions[symbols == 'Pt']
    oxygen = atoms.positions[symbols == 'O']
    ideal = fcc111('Pt', size=(20, 20, 5), a=3.94,
                   orthogonal=True, periodic=True, vacuum=10)
    missing = ideal.positions[cKDTree(pt).query(ideal.positions)[0] > 1e-5]
    xy = data[['x', 'y']].to_numpy() % box
    height = float(data.z.iloc[0])

    def within(points):
        distances = cKDTree(points[:, :2] % box, boxsize=box).query(xy)[0]
        return distances ** 2 + (height - points[0, 2]) ** 2 < radius ** 2

    near_oxygen = within(oxygen)
    near_vacancy = within(missing)
    masks = {
        'terrace': ~near_oxygen & ~near_vacancy,
        'vacancy': near_vacancy & ~near_oxygen,
        'oxygen': near_oxygen & ~near_vacancy,
        'overlap': near_oxygen & near_vacancy,
    }
    return masks, missing, oxygen


def study(data, output, *, seeds=(42, 371, 829),
          ks=tuple(range(2, 41, 2)) + (50, 64),
          global_budget=0.90, regional_budget=0.80, seed_budget=0.90):
    """Find the first tested k that meets stated fidelity budgets for every seed.

    Budgets are user choices about retained descriptor information. They are
    not universal physical thresholds or estimates of the number of site types.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    features = [column for column in data if column.startswith('ACSF_')]
    raw = data[features].to_numpy(float)
    if raw.shape != (81000, 40) or data.z.nunique() != 1:
        raise ValueError('Requires the full 81,000-point, 40-channel decorated tutorial')
    if not np.isfinite(raw).all():
        raise ValueError('Descriptors must be finite')
    scaler = StandardScaler().fit(raw)
    values = scaler.transform(raw)
    families = [name.split('_')[1] for name in features]
    atoms = make_surface('defects')
    masks, missing, oxygen = geometry(data, atoms, 6.5)
    sample = np.sort(np.random.default_rng(42).choice(len(values), 2000, replace=False))
    rows, channel_rows, regional_rows = [], [], []
    solutions, models = {}, {}
    for k in sorted(ks):
        started = time.perf_counter()
        for seed in seeds:
            model = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(values)
            labels = model.labels_
            solutions[k, seed], models[k, seed] = labels, model
            retained = retention(values, labels, model.cluster_centers_)
            regional_minima = []
            for region, mask in masks.items():
                if not mask.any():
                    continue
                local = retention(values[mask], labels[mask])
                for family in sorted(set(families)):
                    selected = np.array(families) == family
                    fraction = float(np.nanmean(local[selected]))
                    regional_minima.append(fraction)
                    regional_rows.append(dict(k=k, seed=seed, region=region,
                                              family=family, retained=fraction,
                                              points=int(mask.sum())))
                for feature, fraction in zip(features, local):
                    channel_rows.append(dict(k=k, seed=seed, region=region,
                                             feature=feature, retained=float(fraction)))
            for feature, fraction in zip(features, retained):
                channel_rows.append(dict(k=k, seed=seed, region='global',
                                         feature=feature, retained=float(fraction)))
            rows.append(dict(k=k, seed=seed,
                             global_retained=float(np.nanmean(retained)),
                             worst_channel_retained=float(np.nanmin(retained)),
                             worst_region_family_retained=min(regional_minima),
                             silhouette=float(silhouette_score(values[sample], labels[sample])),
                             db=float(davies_bouldin_score(values, labels)),
                             inertia=float(model.inertia_),
                             largest_fraction=float(np.bincount(labels).max() / len(values)),
                             smallest_cluster=int(np.bincount(labels).min())))
        agreement = [adjusted_rand_score(solutions[k, a], solutions[k, b])
                     for a, b in combinations(seeds, 2)]
        for row in rows[-len(seeds):]:
            row['seed_ari_mean'] = float(np.mean(agreement))
            row['seed_ari_min'] = float(np.min(agreement))
        pd.DataFrame(rows).to_csv(output / 'resolution_trials.csv', index=False)
        print(f'k={k}: {time.perf_counter() - started:.1f} s; '
              f'worst-seed global retention={min(row["global_retained"] for row in rows[-len(seeds):]):.3f}',
              flush=True)
    table = pd.DataFrame(rows)
    summary = table.groupby('k').agg(
        global_retained_median=('global_retained', 'median'),
        global_retained_min=('global_retained', 'min'),
        regional_retained_min=('worst_region_family_retained', 'min'),
        worst_channel_retained_min=('worst_channel_retained', 'min'),
        silhouette_median=('silhouette', 'median'),
        db_median=('db', 'median'),
        seed_ari_mean=('seed_ari_mean', 'first'),
        seed_ari_min=('seed_ari_min', 'first'),
    )
    eligible = summary.index[(summary.global_retained_min >= global_budget)
                             & (summary.regional_retained_min >= regional_budget)
                             & (summary.seed_ari_min >= seed_budget)]
    if len(eligible) == 0:
        raise ValueError('No tested k meets the budgets; inspect the recorded curves')
    selected = int(eligible[0])
    summary.to_csv(output / 'resolution_summary.csv')
    pd.DataFrame(channel_rows).to_csv(output / 'channel_retention.csv', index=False)
    pd.DataFrame(regional_rows).to_csv(output / 'regional_retention.csv', index=False)
    # Repeat fitting with 20% of rows withheld. Keep the full-plane scaler fixed
    # so this checks partition/reconstruction stability in the same coordinates.
    holdout_rows = []
    for split_seed in (103, 211, 307):
        test = np.random.default_rng(split_seed).choice(len(values), len(values) // 5, replace=False)
        train_mask = np.ones(len(values), dtype=bool)
        train_mask[test] = False
        for k in sorted({2, 10, selected}):
            model = KMeans(n_clusters=k, n_init=10, random_state=42).fit(values[train_mask])
            labels = model.predict(values)
            global_retained = retention(values[test], labels[test], model.cluster_centers_)
            regional_minima = []
            for region, mask in masks.items():
                if not mask.any():
                    continue
                train_region = mask & train_mask
                counts = np.bincount(labels[train_region], minlength=k)
                centers = np.zeros((k, values.shape[1]))
                np.add.at(centers, labels[train_region], values[train_region])
                centers /= np.maximum(counts, 1)[:, None]
                centers[counts == 0] = model.cluster_centers_[counts == 0]
                held_region = test[mask[test]]
                retained = retention(values[held_region], labels[held_region], centers)
                for family in sorted(set(families)):
                    regional_minima.append(float(np.nanmean(retained[np.array(families) == family])))
            holdout_rows.append(dict(k=k, split_seed=split_seed,
                                     test_points=len(test),
                                     global_retained=float(np.nanmean(global_retained)),
                                     regional_retained_min=min(regional_minima),
                                     assignment_ari=float(adjusted_rand_score(
                                         solutions[k, 42][test], labels[test]))))
    pd.DataFrame(holdout_rows).to_csv(output / 'holdout_checks.csv', index=False)
    conditions = dict(points=len(values), features=features, seeds=list(seeds),
                      n_init=10, ks=sorted(ks), selected_k=selected,
                      global_budget=global_budget, regional_budget=regional_budget,
                      seed_budget=seed_budget,
                      selection='Smallest tested k meeting both fidelity budgets for every fitted seed and the minimum pairwise seed-ARI budget',
                      silhouette_points=2000, silhouette_seed=42,
                      geometry_region_counts={name: int(mask.sum()) for name, mask in masks.items()},
                      radius_angstrom=6.5, probe_z=float(data.z.iloc[0]),
                      scope='Descriptor fidelity comparison; no claim of a unique physical cluster count',
                      holdout_scope='Random spatial-grid rows; scaler frozen on the full plane. This is an interpolation/refit check, not independence across structures.',
                      device='CPU', thread_environment={key: __import__('os').environ.get(key)
                      for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']})
    (output / 'conditions.json').write_text(json.dumps(conditions, indent=2))
    np.savez_compressed(output / 'plot_labels.npz',
                        **{f'k{k}': solutions[k, 42] for k in sorted({2, 10, selected})})
    make_figures(data, atoms, missing, oxygen, summary, regional_rows,
                 solutions, selected, output, global_budget, regional_budget)
    print(json.dumps(conditions, indent=2), flush=True)
    return conditions


def make_figures(data, atoms, missing, oxygen, summary, regional_rows,
                 solutions, selected, output, global_budget, regional_budget):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm

    xyz = atoms.positions
    symbols = np.array(atoms.get_chemical_symbols())
    top = xyz[(symbols == 'Pt') & (np.abs(xyz[:, 2] - xyz[symbols == 'Pt', 2].max()) < 0.3)]
    box = atoms.cell.lengths()[:2]
    plotted = sorted({2, 10, selected})

    def field(ax, k, limits=None):
        labels = solutions[k, 42]
        # Deterministic colors; labels are nominal and do not denote site types.
        colors = plt.get_cmap('turbo')(np.linspace(0.03, 0.97, k))
        cmap, norm = ListedColormap(colors), BoundaryNorm(np.arange(k + 1) - 0.5, k)
        grid = pd.DataFrame(dict(x=data.x, y=data.y, label=labels)).pivot(index='y', columns='x', values='label')
        image = ax.pcolormesh(grid.columns, grid.index, grid.to_numpy(),
                              shading='nearest', cmap=cmap, norm=norm, rasterized=True)
        ax.scatter(top[:, 0], top[:, 1], s=9, facecolors='none', edgecolors='black', linewidths=0.35, label='Top Pt')
        ax.scatter(oxygen[:, 0], oxygen[:, 1], s=14, marker='x', c='black', label='O')
        ax.scatter(missing[:, 0], missing[:, 1], s=20, marker='*', c='white', edgecolors='black', linewidths=0.4, label='Missing Pt')
        ax.set(xlabel='x (Å)', ylabel='y (Å)', aspect='equal', title=f'k={k}')
        if limits:
            ax.set_xlim(*limits[:2]); ax.set_ylim(*limits[2:])
        return image

    fig, axes = plt.subplots(1, len(plotted), figsize=(5 * len(plotted), 5.5), layout='constrained')
    for ax, k in zip(np.atleast_1d(axes), plotted):
        image = field(ax, k)
        fig.colorbar(image, ax=ax, ticks=np.arange(k), label='Cluster label', shrink=0.7).ax.tick_params(labelsize=6)
    np.atleast_1d(axes)[0].legend(loc='lower left', fontsize=7)
    fig.suptitle('Decorated Pt(111): metric optimum, k=10, and descriptor-fidelity comparison')
    fig.savefig(output / 'resolution_comparison.png', dpi=180)
    plt.close(fig)
    limits = [(0.13 * box[0], 0.38 * box[0], 0.12 * box[1], 0.38 * box[1]),
              (0.53 * box[0], 0.92 * box[0], 0.13 * box[1], 0.52 * box[1])]
    fig, axes = plt.subplots(2, len(plotted), figsize=(5 * len(plotted), 10), layout='constrained')
    for row, bounds in enumerate(limits):
        for col, k in enumerate(plotted):
            image = field(axes[row, col], k, bounds)
            fig.colorbar(image, ax=axes[row, col], ticks=np.arange(k), label='Cluster label', shrink=0.7).ax.tick_params(labelsize=6)
    axes[0, 0].legend(loc='upper right', fontsize=7)
    fig.suptitle('Vacancy region (top) and oxygen region (bottom); all grid points plotted')
    fig.savefig(output / 'defect_zoom.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout='constrained')
    axes[0, 0].plot(summary.index, summary.silhouette_median, 'o-')
    axes[0, 0].set(ylabel='Median silhouette')
    axes[0, 1].plot(summary.index, summary.db_median, 'o-')
    axes[0, 1].set(ylabel='Median Davies–Bouldin')
    axes[1, 0].plot(summary.index, summary.global_retained_min, 'o-', label='Global mean, worst seed')
    axes[1, 0].plot(summary.index, summary.regional_retained_min, 's-', label='Worst region/family/seed')
    axes[1, 0].axhline(global_budget, ls='--', color='C0', label=f'Global budget {global_budget:.0%}')
    axes[1, 0].axhline(regional_budget, ls='--', color='C1', label=f'Regional budget {regional_budget:.0%}')
    axes[1, 0].set(ylabel='Retained descriptor variance', ylim=(-0.03, 1.03))
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].plot(summary.index, summary.seed_ari_mean, 'o-', label='Mean pairwise ARI')
    axes[1, 1].plot(summary.index, summary.seed_ari_min, 's-', label='Minimum pairwise ARI')
    axes[1, 1].set(ylabel='Agreement across random seeds', ylim=(0, 1.03))
    axes[1, 1].legend(fontsize=8)
    for ax in axes.flat:
        ax.axvline(selected, color='black', ls=':', label=f'Comparison k={selected}')
        ax.set_xlabel('Number of clusters')
        ax.grid(alpha=0.2)
    fig.suptitle('Compactness scores and descriptor fidelity answer different questions')
    fig.savefig(output / 'resolution_curves.png', dpi=180)
    plt.close(fig)
    regional = pd.DataFrame(regional_rows)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), layout='constrained')
    for ax, region in zip(axes, ['terrace', 'vacancy', 'oxygen']):
        view = regional[regional.region == region].groupby(['k', 'family']).retained.median().unstack()
        for family in view:
            ax.plot(view.index, view[family], 'o-', ms=3, label=family)
        ax.axhline(regional_budget, color='gray', ls='--')
        ax.axvline(selected, color='black', ls=':')
        ax.set(title=region, xlabel='Number of clusters', ylabel='Within-region variance retained', ylim=(-0.02, 1.02))
        ax.legend(fontsize=8)
    fig.savefig(output / 'regional_fidelity.png', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--descriptors', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--global-budget', type=float, default=0.90)
    parser.add_argument('--regional-budget', type=float, default=0.80)
    parser.add_argument('--seed-budget', type=float, default=0.90)
    args = parser.parse_args()
    if args.descriptors:
        data = pd.read_csv(args.descriptors)
    else:
        from corrected_surfaces import build
        from common import make_functions
        from mapsy.maps import Maps
        _, system, contact, radius = build('defects')
        data = Maps(system, make_functions(radius, compositional=True), contact,
                    use_gpu=False, batch_size=256).atcontactspace()
    study(data, args.output, global_budget=args.global_budget,
          regional_budget=args.regional_budget, seed_budget=args.seed_budget)
    if args.descriptors:
        (args.output / 'input_integrity.json').write_text(json.dumps(
            dict(descriptor_filename=args.descriptors.name,
                 sha256=hashlib.sha256(args.descriptors.read_bytes()).hexdigest()), indent=2))


if __name__ == '__main__':
    main()
