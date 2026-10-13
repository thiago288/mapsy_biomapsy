"""Check exact terrace sites and input consistency for decorated_resolution.py.

Run after the resolution study, passing its directory and descriptor CSV.
The test evaluates atop, bridge, fcc and hcp anchors outside both defect cutoffs.
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd
from ase.build import fcc111
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from decorated_resolution import geometry, retention
from site_checks import anchors

EXAMPLES = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLES))
sys.path.insert(0, str(EXAMPLES / 'periodic_surfaces'))
from structures import make_surface
from common import make_functions
from mapsy.data import Grid, System
from mapsy.maps import Maps


def check(csv, output):
    output = Path(output)
    conditions = json.loads((output / 'conditions.json').read_text())
    data = pd.read_csv(csv)
    features = [column for column in data if column.startswith('ACSF_')]
    scaler = StandardScaler().fit(data[features])
    values = scaler.transform(data[features])
    atoms = make_surface('defects')
    model = Maps(System(Grid(cell=atoms.cell), atoms, dimension=2, axis=2),
                 make_functions(6.5, compositional=True), use_gpu=False, batch_size=256)
    subset = np.sort(np.random.default_rng(91).choice(len(data), 256, replace=False))
    fresh = model.atpoints(data.iloc[subset][['x', 'y', 'z']].to_numpy())[features]
    reference = data.iloc[subset][features].to_numpy(dtype=float)
    error = float(np.abs(fresh.to_numpy(dtype=float) - reference).max())
    np.testing.assert_allclose(fresh.to_numpy(dtype=float), reference, rtol=1e-10, atol=1e-12)
    parent = fcc111('Pt', size=(20, 20, 5), a=3.94,
                    orthogonal=True, periodic=True, vacuum=10)
    sites = anchors(parent)
    # Periodic triangulation can repeat anchors with tiny coordinate differences.
    # Deduplicate the anchor list; do not round descriptor inputs or values.
    box = atoms.cell.lengths()[:2]
    coordinates = sites[['x', 'y']].to_numpy() % box
    coordinates[np.isclose(coordinates, box, atol=1e-6, rtol=0)] = 0.
    sites[['x', 'y']] = coordinates
    keys = sites[['site']].copy()
    keys[['x', 'y']] = np.round(coordinates, 6)
    sites = sites.loc[~keys.duplicated()].reset_index(drop=True)
    sites['z'] = data.z.iloc[0]
    regions, _, _ = geometry(sites, atoms, 6.5)
    sites = sites[regions['terrace']].reset_index(drop=True)
    exact = model.atpoints(sites[['x', 'y', 'z']].to_numpy())[features]
    site_values = scaler.transform(exact)
    names = ['atop', 'bridge', 'fcc', 'hcp']
    means = np.vstack([site_values[sites.site == name].mean(axis=0) for name in names])
    distances = np.linalg.norm(means[:, None] - means[None, :], axis=-1)
    pd.DataFrame(distances, index=names, columns=names).to_csv(output / 'terrace_site_distances.csv')
    rows = []
    labels = np.load(output / 'plot_labels.npz')
    for k in sorted({2, 10, conditions['selected_k']}):
        clustering = KMeans(n_clusters=k, n_init=10, random_state=42).fit(values)
        np.testing.assert_array_equal(clustering.labels_, labels[f'k{k}'])
        predicted = clustering.predict(site_values)
        contingency = pd.crosstab(sites.site, predicted).reindex(names, fill_value=0)
        contingency.to_csv(output / f'terrace_sites_k{k}.csv')
        fractions = contingency.div(contingency.sum(axis=1), axis=0)
        rows.append(dict(k=k, distinct_modal_site_labels=len(set(contingency.idxmax(axis=1))),
                         site_discrimination=float(fractions.max(axis=0).sum() / 4),
                         within_family_equivalence=float(fractions.max(axis=1).mean()),
                         modal_labels=json.dumps(contingency.idxmax(axis=1).to_dict())))
    pd.DataFrame(rows).to_csv(output / 'terrace_site_checks.csv', index=False)
    sites.to_csv(output / 'terrace_site_anchors.csv', index=False)
    result = dict(fresh_descriptor_points=256, fresh_descriptor_max_abs=error,
                  fresh_descriptor_allclose=True,
                  exact_site_counts=sites.site.value_counts().to_dict(),
                  within_site_max_standardized_distance={name: float(np.linalg.norm(
                      site_values[sites.site == name] - means[i], axis=1).max())
                      for i, name in enumerate(names)},
                  scope='Exact high-symmetry terrace coordinates, beyond all O and vacancy cutoff spheres; no nearest-grid approximation')
    (output / 'site_check_conditions.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    check(args.csv, args.output)
