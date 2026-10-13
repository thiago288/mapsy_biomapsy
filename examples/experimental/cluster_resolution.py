"""Exploratory coverage/stability sweep on a tutorial descriptor CSV."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score, davies_bouldin_score

def coverage(X, labels):
    fitted = np.zeros_like(X)
    for label in np.unique(labels):
        mask = labels == label
        fitted[mask] = X[mask].mean(axis=0)
    return 1 - ((X - fitted) ** 2).sum(axis=0) / (X ** 2).sum(axis=0)

def run(csv, output, ks=(2, 3, 4, 5, 6, 8, 10, 12, 14, 18, 24, 30), seeds=(42, 371, 829)):
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Use a fresh output directory')
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv)
    features = [c for c in df if c.startswith('ACSF_') or c in ('electrostatic', 'hydrophobicity')]
    raw = df[features].to_numpy(float)
    keep = raw.std(0) > 1e-12
    features = list(np.array(features)[keep])
    X = StandardScaler().fit_transform(raw[:, keep])
    if not features or not np.isfinite(X).all():
        raise ValueError('Need finite nonconstant descriptors')
    if max(ks) >= len(X):
        raise ValueError('Every k must be smaller than the number of points')
    families = [f.split('_')[1] if f.startswith('ACSF_') else 'physicochemical' for f in features]
    W = X.copy()
    for family in set(families):
        W[:, np.array(families) == family] /= np.sqrt(families.count(family))
    sample = np.random.default_rng(42).choice(len(X), min(1500, len(X)), replace=False)
    rows = []
    for strategy, V in [('standardized', X), ('family_balanced', W)]:
        channels = []
        for k in ks:
            solutions = []
            local = []
            for seed in seeds:
                labels = KMeans(n_clusters=k, n_init=5, random_state=seed).fit_predict(V)
                solutions.append(labels)
                cov = coverage(X, labels)
                local.append(dict(strategy=strategy,
                        k=k,
                        seed=seed,
                        silhouette=float(silhouette_score(V[sample],
                                labels[sample])),
                        db=float(davies_bouldin_score(V,
                                labels)),
                        coverage_mean=float(cov.mean()),
                        coverage_worst=float(cov.min()),
                        largest_fraction=float(np.bincount(labels).max() / len(X))))
                if seed == seeds[0]:
                    np.save(out / f'{strategy}_k{k}_labels.npy', labels)
                    channels.append(cov)
            ari = float(np.mean([adjusted_rand_score(solutions[0],
                            s) for s in solutions[1:]])) if len(seeds) > 1 else None
            noise = np.random.default_rng(100 + k).normal(0, 0.01, V.shape)
            changed = KMeans(n_clusters=k, n_init=5, random_state=seeds[0]).fit_predict(V + noise)
            for row in local:
                row.update(seed_ari=ari,
                    noise_ari=float(adjusted_rand_score(solutions[0],
                            changed)))
                rows.append(row)
        pd.DataFrame(channels,
            index=ks,
            columns=features).to_csv(out / f'{strategy}_descriptor_coverage.csv')
    pd.DataFrame(rows).to_csv(out / 'scores.csv', index=False)
    (out / 'conditions.json').write_text(json.dumps(dict(input=str(csv),
                points=len(X),
                features=features,
                ks=list(ks),
                seeds=list(seeds),
                silhouette_points=len(sample),
                n_init=5,
                noise_std=0.01,
                scope='Exploratory diagnostics; no recommended physical k'),
            indent=2))
    return rows
if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('csv', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    run(a.csv, a.output)
