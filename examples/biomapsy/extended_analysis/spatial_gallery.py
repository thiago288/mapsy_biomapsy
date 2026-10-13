from pathlib import Path
import os
import sys
import json
for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[k] = '1'
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root / 'examples'))
from common import plot_protein_fields

def generate(output):
    output = Path(output)
    data = [pd.read_parquet(output / name) for name in ['maps1_full.parquet', 'maps2_full.parquet']]
    features = list(np.load(output / 'acsf_feature_names.npy',
            allow_pickle=True)) + ['electrostatic',
        'hydrophobicity']
    X = np.vstack([df[features].to_numpy(float) for df in data])
    scaler = StandardScaler().fit(X)
    pca = PCA(n_components=len(features)).fit(scaler.transform(X))
    scores = pca.transform(scaler.transform(X))
    assert np.isfinite(scores).all()
    limits = {key: np.quantile(X[:, i], [0.02, 0.98]).tolist() for i, key in enumerate(features)}
    pc_limits = {f'PC{i + 1}': np.quantile(scores[:, i], [0.02, 0.98]).tolist() for i in range(4)}
    pd.DataFrame({'component': np.arange(1,
                len(features) + 1),
            'explained_variance': pca.explained_variance_ratio_}).to_csv(output / 'extended_shared_pca_variance.csv',
        index=False)
    pd.DataFrame(pca.components_[:4].T,
        index=features,
        columns=[f'PC{i + 1}' for i in range(4)]).to_csv(output / 'extended_shared_pca_loadings.csv')
    start = 0
    for df, state, title in zip(data,
        ['apo',
            'holo'],
        ['Apo 1CFD',
            'Holo-derived 1CLL (calcium omitted)']):
        for i in range(4):
            df[f'PC{i + 1}'] = scores[start:start + len(df), i]
        start += len(df)
        plot_protein_fields(df,
            ['PC1',
                'PC2',
                'PC3',
                'PC4'],
            output,
            state + '_extended_shared_pca',
            title + ' — 54-channel shared PCA',
            limits=pc_limits)
        plot_protein_fields(df,
            ['electrostatic',
                'hydrophobicity'],
            output,
            state + '_extended_fields',
            title + ' — exact atom/property alignment',
            limits={k: limits[k] for k in ['electrostatic',
                    'hydrophobicity']})
        for group in range(0, 52, 4):
            keys = features[group:group + 4]
            plot_protein_fields(df,
                keys,
                output,
                state + f'_extended_descriptors_{group // 4 + 1:02d}',
                title + ' — extended radial AC',
                limits={k: limits[k] for k in keys})
        print('STATIC GALLERY COMPLETE', state, flush=True)
    return {'channels': 54,
        'pca_basis': 'Pooled 1CFD/1CLL standardized 54-channel matrix',
        'ac_panels_per_protein': 13,
        'ac_channels_plotted': 52,
        'colors': 'Shared pooled 2–98% limits, opposite views, display subset maximum 12000 with seed42',
        'pca_90_percent_components': int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_),
                0.9) + 1)}
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Render every extended AC channel and pooled PCA from an executed notebook output directory.')
    parser.add_argument('output', type=Path)
    output = parser.parse_args().output
    result = generate(output)
    (output / 'spatial_gallery_validation.json').write_text(json.dumps(result, indent=2))
    print(result)
