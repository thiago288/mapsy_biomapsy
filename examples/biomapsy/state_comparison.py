"""Compare apo/holo-derived descriptor distributions with a shared PCA basis."""
from pathlib import Path
import sys
import json
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import new_output, feature_columns, plot_protein_fields, make_functions
import numpy as np
import pandas as pd

def compare_states(apo, holo, output=None):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    out = new_output('apo_holo_comparison', output)
    frames = [pd.read_csv(Path(s['output']) / 'descriptors.csv') for s in [apo, holo]]
    keys = feature_columns(frames[0])
    assert keys == feature_columns(frames[1])
    X = np.concatenate([d[keys].to_numpy(float) for d in frames])
    scaler = StandardScaler().fit(X)
    pca = PCA(n_components=4, svd_solver='full').fit(scaler.transform(X))
    scores = pca.transform(scaler.transform(X))
    split = len(frames[0])
    limits = {f'PC{j + 1}': np.quantile(scores[:, j], [0.02, 0.98]).tolist() for j in range(4)}
    for i, (d, label) in enumerate(zip(frames, ['1CFD apo', '1CLL holo-derived'])):
        Z = scores[:split] if i == 0 else scores[split:]
        for j in range(4):
            d[f'PC{j + 1}'] = Z[:, j]
        plot_protein_fields(d,
            ['PC1',
                'PC2'],
            out,
            ['apo_shared_pca',
                'holo_shared_pca'][i],
            label + ' — shared pooled PCA',
            limits=limits)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout='constrained')
    for i, (d,
        label,
        color) in enumerate(zip(frames,
            ['1CFD apo',
                '1CLL holo-derived'],
            ['steelblue',
                'tomato'])):
        idx = np.sort(np.random.default_rng(42).choice(len(d), min(6000, len(d)), replace=False))
        for ax, (a, b) in zip(axes, [(1, 2), (1, 3), (2, 3)]):
            ax.scatter(d[f'PC{a}'].to_numpy()[idx],
                d[f'PC{b}'].to_numpy()[idx],
                s=2,
                alpha=0.3,
                c=color,
                label=label)
            ax.set(xlabel=f'Shared PC{a}', ylabel=f'Shared PC{b}')
            ax.legend(markerscale=3)
    fig.suptitle('Same pooled scaler and PCA basis; protein-only conformations')
    fig.savefig(out / 'shared_pca_scatter.png', dpi=150)
    fig.savefig(out / 'shared_pca_scatter.pdf')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
    rows = []
    for d, label, color in zip(frames, ['1CFD apo', '1CLL holo-derived'], ['steelblue', 'tomato']):
        for ax, key in zip(axes, ['electrostatic', 'hydrophobicity']):
            values = d[key].to_numpy()
            ax.hist(values, bins=50, density=True, alpha=0.5, label=label, color=color)
            ax.set(xlabel=key, ylabel='Probability density')
            ax.legend()
            rows.append(dict(state=label,
                    feature=key,
                    mean=float(values.mean()),
                    std=float(values.std()),
                    minimum=float(values.min()),
                    maximum=float(values.max())))
    fig.suptitle('Notebook-derived fields: distributions, not independent biological replicates')
    fig.savefig(out / 'physicochemical_distributions.png', dpi=150)
    fig.savefig(out / 'physicochemical_distributions.pdf')
    plt.close(fig)
    pd.DataFrame(rows).to_csv(out / 'physicochemical_statistics.csv', index=False)
    pd.DataFrame(pca.components_.T,
        index=keys,
        columns=['PC1',
            'PC2',
            'PC3',
            'PC4']).to_csv(out / 'shared_pca_loadings.csv')
    report = dict(output=str(out),
        features=keys,
        points=[len(d) for d in frames],
        states=['1CFD apo',
            '1CLL holo-derived, calcium omitted'],
        pca='One pooled StandardScaler and PCA, all rows',
        variance=pca.explained_variance_ratio_.tolist(),
        interpretation='Descriptive comparison of different protein-only structures and sampling surfaces; not a causal calcium-binding experiment')
    (out / 'comparison.json').write_text(json.dumps(report, indent=2))
    return report

def assess_descriptor_sets(summary, points=512, output=None):
    """Same-point radial/mixed and with/without chemistry ablation."""
    from protein_workflow import build
    from physicochemical import add_fields
    from mapsy.biomapsy import BioMaps
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
    out = new_output(summary['pdb_accession'] + '_descriptor_sets', output)
    data = pd.read_csv(Path(summary['output']) / 'descriptors.csv')
    idx = np.sort(np.random.default_rng(42).choice(len(data),
            min(points,
                len(data)),
            replace=False))
    query = data[['x', 'y', 'z']].to_numpy(float)[idx]
    system, cs, meta = build(summary['pdb_accession'], mode=summary['contact_mode'])
    mixed = BioMaps(system,
        make_functions(summary['radius_angstrom'],
            len(summary['orders']),
            True,
            'mixed'),
        cs,
        use_gpu=False,
        batch_size=128)
    values = mixed.atpoints(query)
    values, chem = add_fields(values, system.atoms, meta['pdb_path'])
    radial = [c for c in values if c.startswith(('ACSF_RS', 'ACSF_RC'))]
    all_ac = [c for c in values if c.startswith('ACSF_')]
    cached = data[radial].to_numpy(float)[idx]
    np.testing.assert_allclose(cached, values[radial].to_numpy(float), rtol=1e-10, atol=1e-10)
    np.savez_compressed(out / 'same_points_all_channels.npz',
        positions=query,
        values=values[feature_columns(values)].to_numpy(float),
        features=np.array(feature_columns(values)))
    rows = []
    baseline = None
    k = summary['cluster_choices']['best_davies_bouldin']['nclusters']
    for name, keys in [('radial',
            radial),
        ('radial+physicochemical',
            radial + ['electrostatic',
                'hydrophobicity']),
        ('radial+angular',
            all_ac),
        ('radial+angular+physicochemical',
            all_ac + ['electrostatic',
                'hydrophobicity'])]:
        X = StandardScaler().fit_transform(values[keys])
        pca = PCA(n_components=4, svd_solver='full').fit(X)
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
        if baseline is None:
            baseline = labels
        rows.append(dict(configuration=name,
                points=len(query),
                features=len(keys),
                comparison_k=k,
                variance_PC1=float(pca.explained_variance_ratio_[0]),
                variance_first_four=float(pca.explained_variance_ratio_.sum()),
                silhouette=float(silhouette_score(X,
                        labels)),
                db=float(davies_bouldin_score(X,
                        labels)),
                ARI_to_radial=float(adjusted_rand_score(baseline,
                        labels))))
    pd.DataFrame(rows).to_csv(out / 'descriptor_set_comparison.csv', index=False)
    (out / 'conditions.json').write_text(json.dumps(dict(points=len(query),
                seed=42,
                comparison_k=k,
                k_source='Full radial+physicochemical DB sweep; held fixed here to isolate descriptor choice',
                interpretation='Different standardized feature spaces; scores are not a direct ranking of physical accuracy',
                physicochemical=chem),
            indent=2))
    return dict(output=str(out), cases=rows)
