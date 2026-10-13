"""Known Pt111 site geometry and descriptor symmetry; no energy prerequisites."""
from pathlib import Path
import sys
import argparse
import json
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay, cKDTree
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'periodic_surfaces'))
from structures import make_surface
from common import make_functions, new_output
from mapsy.data import Grid, System
from mapsy.maps import Maps

def anchors(atoms):
    L = np.diag(atoms.cell)[:2]
    p = atoms.positions
    zs = np.unique(np.round(p[:, 2], 8))
    top = p[np.isclose(p[:, 2], zs[-1]), :2]
    second = p[np.isclose(p[:, 2], zs[-2]), :2]
    tile = np.vstack([top + np.array([i, j]) * L for i in [-1, 0, 1] for j in [-1, 0, 1]])
    tri = tile[Delaunay(tile).simplices]
    a = 3.94 / np.sqrt(2)
    tri = tri[np.max(np.linalg.norm(tri - np.roll(tri, 1, axis=1), axis=2), axis=1) < a * 1.01]

    def unique(c):
        return np.unique(np.round(c / L, 10) % 1, axis=0) * L
    hole = unique(tri.mean(axis=1))
    delta = hole[:, None] - second[None]
    delta -= np.rint(delta / L) * L
    hcp = np.linalg.norm(delta, axis=2).min(1) < a * 0.1
    sites = {'atop': top,
        'bridge': np.vstack([(tri[:,
                        i] + tri[:,
                        (i + 1) % 3]) / 2 for i in range(3)]),
        'hcp': hole[hcp],
        'fcc': hole[~hcp]}
    return pd.DataFrame([dict(site=s, x=q[0], y=q[1]) for s, c in sites.items() for q in unique(c)])

def run(csv, output=None, stability=False):
    csv = Path(csv)
    meta = json.loads((csv.parent / 'summary.json').read_text())
    if meta.get('face') != '111' or meta.get('atoms') != 48 or meta.get('radius_angstrom') != 4.5:
        raise ValueError('Requires the bundled ideal Pt111 4x4x3 model at radius4.5, with summary.json')
    df = pd.read_csv(csv)
    out = new_output('Pt111_site_check', output)
    at = make_surface('111')
    sites = anchors(at)
    if df.z.nunique() != 1:
        raise ValueError('Expected a single Pt111 plane')
    sites['z'] = df.z.iloc[0]
    system = System(Grid(cell=at.cell), at, dimension=2, axis=2)
    model = Maps(system, make_functions(4.5), use_gpu=False, batch_size=256)
    exact = model.atpoints(sites[['x', 'y', 'z']].to_numpy(float))
    features = [c for c in exact if c.startswith('ACSF_')]
    sc = StandardScaler().fit(df[features])
    X = sc.transform(exact[features])
    names = ['atop', 'bridge', 'hcp', 'fcc']
    means = np.vstack([X[sites.site == s].mean(0) for s in names])
    pd.DataFrame(np.linalg.norm(means[:,
                None] - means[None,
                :],
            axis=2),
        index=names,
        columns=names).to_csv(out / 'site_descriptor_distances.csv')
    L = np.diag(at.cell)[:2]
    dist, idx = cKDTree(df[['x',
                'y']].to_numpy() % L,
        boxsize=L).query(sites[['x',
                'y']].to_numpy() % L)
    sites['nearest_grid_index'] = idx
    sites['offset_A'] = dist
    rows = []
    for c in [c for c in df if c.startswith('Cluster_')]:
        tab = pd.crosstab(sites.site, df[c].to_numpy()[idx])
        tab.to_csv(out / (c + '_contingency.csv'))
        weights = tab.div(tab.sum(1), axis=0)
        rows.append(dict(partition=c,
                site_discrimination=float(weights.max(0).sum() / 4),
                site_equivalence=float(weights.max(1).mean())))
    sites.to_csv(out / 'site_anchors.csv', index=False)
    pd.DataFrame(rows).to_csv(out / 'site_assignments.csv', index=False)
    result = dict(site_counts=sites.site.value_counts().to_dict(),
        within_site_max_standardized_distance={s: float(np.linalg.norm(X[sites.site == s] - means[i],
                    axis=1).max()) for i,
            s in enumerate(names)},
        max_grid_offset_A=float(dist.max()),
        probe_z=float(df.z.iloc[0]),
        scope='Ideal Pt111 fixture, radius4.5. Descriptors at exact sites; cluster assignments at nearest grid points, with offsets recorded.')
    if stability:
        points = sites[['x', 'y', 'z']].to_numpy(float)

        def compute(at, p):
            m = Maps(System(Grid(cell=at.cell),
                    at,
                    dimension=2,
                    axis=2),
                make_functions(4.5),
                use_gpu=False,
                batch_size=256)
            return m.atpoints(p)[features].to_numpy(float)
        noisy = at.copy()
        noisy.positions += np.random.default_rng(19).normal(0, 0.01, noisy.positions.shape)
        changed = sc.transform(compute(noisy, points))
        shifted = sc.transform(compute(at, points + np.array([0.02, 0.02, 0])))
        full = sc.transform(df[features])
        rows = []
        for k in [2, 3, 4, 5, 6, 8, 10, 12, 14, 18, 24, 30]:
            m = KMeans(n_clusters=k, n_init=5, random_state=42).fit(full)
            labels = m.predict(X)
            tab = pd.crosstab(sites.site, labels)
            w = tab.div(tab.sum(1), axis=0)
            rows.append(dict(k=k,
                    site_purity=float(w.max(0).sum() / 4),
                    site_equivalence=float(w.max(1).mean()),
                    distinct_modal_sites=len(set(tab.idxmax(1))),
                    atomic_noise_assignment_agreement=float(np.mean(m.predict(changed) == labels)),
                    grid_shift_assignment_agreement=float(np.mean(m.predict(shifted) == labels)),
                    atomic_noise_ARI=float(adjusted_rand_score(labels,
                            m.predict(changed)))))
        pd.DataFrame(rows).to_csv(out / 'exact_site_acceptance.csv', index=False)
        xx, yy = np.meshgrid(np.arange(0, L[0], 0.12), np.arange(0, L[1], 0.12), indexing='ij')
        fine = np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, points[0, 2])])
        fineX = sc.transform(compute(at, fine))
        rows = []
        for k in [4, 10, 18, 30]:
            a = KMeans(n_clusters=k, n_init=5, random_state=42).fit(full)
            b = KMeans(n_clusters=k, n_init=5, random_state=42).fit(fineX)
            rows.append(dict(k=k,
                    original_grid_points=len(full),
                    fine_grid_points=len(fineX),
                    shared_site_ARI=float(adjusted_rand_score(a.predict(X),
                            b.predict(X)))))
        pd.DataFrame(rows).to_csv(out / 'grid_refit_stability.csv', index=False)
        translated = at.copy()
        t = np.array([0.37, 0.61, 0])
        translated.positions += t
        delta = compute(at, points) - compute(translated, points + t)
        result.update(translation_max_abs=float(abs(delta).max()),
            atomic_noise_std_A=0.01,
            query_shift_A=[0.02,
                0.02,
                0],
            grid_refit_scope='Fit both grids with frozen original scaler; compare exact-site assignments, not full-plane ARI')
    (out / 'conditions.json').write_text(json.dumps(result, indent=2))
    return result
if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('csv', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--stability', action='store_true')
    a = p.parse_args()
    print(json.dumps(run(a.csv, a.output, a.stability), indent=2))
