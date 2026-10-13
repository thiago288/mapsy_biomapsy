"""Reproducible same-input CPU/GPU checks; CuPy is optional."""
from pathlib import Path
import sys
import argparse
import json
import time
import warnings
sys.path[:0] = [str(Path(__file__).resolve().parents[1]),
    str(Path(__file__).resolve().parents[1] / 'biomapsy'),
    str(Path(__file__).resolve().parents[1] / 'periodic_surfaces')]
from common import new_output, make_functions, environment
from protein_workflow import build as build_protein
from corrected_surfaces import build as build_surface
from mapsy.maps import Maps
from mapsy.biomapsy import BioMaps
import numpy as np
import pandas as pd

def gpu_state():
    try:
        import cupy as cp
        n = cp.cuda.runtime.getDeviceCount()
        if n < 1:
            return (None, dict(available=False, reason='No CUDA device'))
        props = cp.cuda.runtime.getDeviceProperties(0)
        name = props['name']
        name = name.decode() if isinstance(name, bytes) else str(name)
        return (cp, dict(available=True, device=name, cupy=cp.__version__))
    except Exception as e:
        return (None, dict(available=False, reason=str(e)))

def run(*,
    points=2048,
    repeats=3,
    warmups=1,
    output=None,
    descriptor_set='radial',
    include_periodic=True):
    if points < 16 or repeats < 1 or warmups < 1:
        raise ValueError('points >=16, repeats >=1, warmups >=1 required')
    out = new_output('cpu_gpu', output)
    cp, gpu = gpu_state()
    cases = []
    definitions = [('protein',
            protein,
            mode) for protein in ['1CLL',
            '1CFD'] for mode in ['tan',
            'ionic-ass']]
    if include_periodic:
        definitions += [('surface', face, 'surface') for face in ['111', 'defects']]

    def sync():
        if cp is not None:
            cp.cuda.get_current_stream().synchronize()
    for kind, name, mode in definitions:
        start = time.perf_counter()
        if kind == 'protein':
            system, cs, meta = build_protein(name, mode=mode)
            radius = 10
            order = 12
            compositional = True
            factory = lambda gpu: BioMaps(system,
                make_functions(radius,
                    order,
                    compositional,
                    descriptor_set),
                cs,
                use_gpu=gpu,
                use_kdtree=not gpu,
                batch_size=256)
            xyz = cs.positions
        else:
            atoms, system, cs, radius = build_surface(name)
            order = 10
            compositional = name == 'defects'
            factory = lambda gpu: Maps(system,
                make_functions(radius,
                    order,
                    compositional,
                    descriptor_set),
                cs,
                use_gpu=gpu,
                batch_size=256)
            xyz = cs.data[['x', 'y', 'z']].to_numpy()
            meta = dict(atoms=len(atoms), reference='surface', radius=radius)
        construction = time.perf_counter() - start
        indices = np.sort(np.random.default_rng(42).choice(len(xyz),
                min(points,
                    len(xyz)),
                replace=False))
        positions = np.asarray(xyz[indices], dtype=float).copy()
        results = {}
        warnings_seen = []
        for backend in ['cpu', 'gpu'] if gpu['available'] else ['cpu']:
            model = factory(backend == 'gpu')
            executed_backends = []

            def compute():
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    df = model.atpoints(positions)
                warnings_seen.extend((str(w.message) for w in caught))
                executed_backends.append(model.last_backend)
                return df.filter(regex='^ACSF_').to_numpy(dtype=float)
            for _ in range(warmups):
                compute()
                sync()
            times = []
            for _ in range(repeats):
                sync()
                t = time.perf_counter()
                values = compute()
                sync()
                times.append(time.perf_counter() - t)
            if not np.isfinite(values).all():
                raise AssertionError('Non-finite descriptors')
            results[backend] = dict(values=values,
                times=times,
                actual_backend=model.last_backend,
                executed_backends=executed_backends)
        a = results['cpu']['values']
        row = dict(case=name + '_' + mode,
            kind=kind,
            descriptor_set=descriptor_set,
            points=len(positions),
            total_contact_points=len(xyz),
            features=a.shape[1],
            radius_angstrom=radius,
            orders=list(range(order)),
            structural=True,
            compositional=compositional,
            batch_size=256,
            contact_and_setup_seconds=construction,
            cpu_seconds=results['cpu']['times'],
            cpu_median=float(np.median(results['cpu']['times'])),
            warnings=warnings_seen,
            metadata=meta,
            gpu_status='unavailable')
        arrays = dict(positions=positions, selected_contact_indices=indices, cpu=a)
        if 'gpu' in results:
            b = results['gpu']['values']
            arrays['gpu'] = b
            actual = results['gpu']['actual_backend'] if all((b == 'gpu' for b in results['gpu']['executed_backends'])) else 'cpu_fallback'
            equal = bool(np.allclose(a, b, rtol=1e-10, atol=1e-12))
            row.update(actual_gpu_backend=actual,
                gpu_seconds=results['gpu']['times'],
                gpu_median=float(np.median(results['gpu']['times'])),
                max_abs_difference=float(np.max(abs(a - b))),
                numerically_equal=equal,
                gpu_status='verified' if actual == 'gpu' else 'cpu_fallback')
            if not equal:
                raise AssertionError(f'CPU/GPU mismatch: {row}')
            if actual == 'gpu':
                row['cpu_over_gpu'] = row['cpu_median'] / row['gpu_median']
            from sklearn.preprocessing import StandardScaler
            from sklearn.decomposition import PCA
            from sklearn.cluster import KMeans
            from sklearn.metrics import adjusted_rand_score
            scaler = StandardScaler().fit(a)
            A = scaler.transform(a)
            B = scaler.transform(b)
            pca = PCA(n_components=min(4, a.shape[1]), svd_solver='full').fit(A)
            row['common_pca_max_abs_difference'] = float(np.max(abs(pca.transform(A) - pca.transform(B))))
            la = KMeans(n_clusters=8, n_init=10, random_state=42).fit_predict(A)
            lb = KMeans(n_clusters=8, n_init=10, random_state=42).fit_predict(B)
            row['cpu_gpu_kmeans_ARI'] = float(adjusted_rand_score(la, lb))
        np.savez_compressed(out / (row['case'] + '.npz'), **arrays)
        cases.append(row)
        print(row['case'], row.get('cpu_over_gpu'), row['gpu_status'], flush=True)
    report = dict(environment=environment(),
        gpu=gpu,
        conditions=dict(repeats=repeats,
            warmups=warmups,
            seed=42,
            rtol=1e-10,
            atol=1e-12,
            timing='CUDA synchronized wall time; descriptor computation, transfers and DataFrame included; contact construction, model setup and plotting excluded',
            cpu='BioMapSy uses KD-tree; Maps uses vectorized dense batches',
            points='Explicit uniformly sampled contact positions, identical for both backends',
            descriptor_set=descriptor_set,
            angular='Mixed descriptors may execute angular work on CPU; radial-only performance does not predict mixed performance'),
        cases=cases)
    (out / 'benchmark.json').write_text(json.dumps(report, indent=2))
    pd.DataFrame([{k: v for k,
                v in row.items() if k not in ['warnings',
                    'metadata',
                    'cpu_seconds',
                    'gpu_seconds']} for row in cases]).to_csv(out / 'benchmark.csv',
        index=False)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 5), layout='constrained')
    x = np.arange(len(cases))
    width = 0.38

    def errors(key):
        values = [np.asarray(c[key], dtype=float) for c in cases]
        med = np.array([np.median(v) for v in values])
        return np.array([med - np.array([v.min() for v in values]),
                np.array([v.max() for v in values]) - med])
    ax.bar(x - width / 2,
        [c['cpu_median'] for c in cases],
        width,
        label='CPU',
        color='steelblue',
        yerr=errors('cpu_seconds'),
        capsize=3)
    verified = [(i, c) for i, c in enumerate(cases) if c.get('gpu_status') == 'verified']
    if verified:
        ax.bar([i + width / 2 for i,
                c in verified],
            [c['gpu_median'] for i,
                c in verified],
            width,
            label='GPU',
            color='darkorange',
            yerr=np.array([[c['gpu_median'] - min(c['gpu_seconds']) for i,
                        c in verified],
                    [max(c['gpu_seconds']) - c['gpu_median'] for i,
                        c in verified]]),
            capsize=3)
    ax.set(xticks=x,
        xticklabels=[c['case'] for c in cases],
        ylabel='Descriptor wall time (seconds)',
        title=f'Identical input positions; {descriptor_set} descriptors; median and observed min–max')
    ax.tick_params(axis='x', labelrotation=20)
    ax.legend()
    fig.savefig(out / 'cpu_gpu_times.png', dpi=160)
    fig.savefig(out / 'cpu_gpu_times.pdf')
    plt.close(fig)
    return dict(output=str(out), **report)
if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--points', type=int, default=2048)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--warmups', type=int, default=1)
    p.add_argument('--descriptor-set', choices=['radial', 'mixed'], default='radial')
    p.add_argument('--proteins-only', action='store_true')
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    run(points=a.points,
        repeats=a.repeats,
        warmups=a.warmups,
        descriptor_set=a.descriptor_set,
        include_periodic=not a.proteins_only,
        output=a.output)
