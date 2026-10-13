"""External planar probe examples, distinct from legacy notebook reproduction."""
from pathlib import Path
import sys
import argparse
import json
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import new_output, make_functions, analyze, environment
from structures import make_surface
import numpy as np
from mapsy.data import Grid, System
from mapsy.io.parser import ContactSpaceGenerator, ContactSpaceModel
from mapsy.maps import Maps
from mapsy.utils import setscalars

def build(face='111', *, cutoff=80, clearance=1.5, radius=None, size=None):
    if 'reference' not in ContactSpaceModel.__fields__:
        raise RuntimeError('This example requires the fixed MapSy surface-reference API. Integrate the fixed production source before running it.')
    atoms = make_surface(face, size=size)
    radius = float(radius or (6.5 if face == 'defects' else 4.5))
    if radius >= min(atoms.cell.lengths()[:2]) / 2:
        raise ValueError('Cutoff reaches half the shortest lateral repeat; increase the slab size to avoid missing repeated images')
    scalars = setscalars(np.asarray(atoms.cell), cutoff)
    if np.prod(scalars) > 15000000:
        raise ValueError('Contact grid exceeds the example 15-million-point guard')
    system = System(Grid(cell=atoms.cell), atoms, dimension=2, axis=2)
    cs = ContactSpaceGenerator(ContactSpaceModel(mode='system',
            reference='surface',
            distance=clearance,
            spread=1,
            cutoff=cutoff,
            threshold=-1,
            side=1)).generate(system)
    if len(cs.data) == 0 or cs.data.z.min() <= atoms.positions[:, 2].max():
        raise AssertionError('External contact plane is empty or buried')
    return (atoms, system, cs, radius)

def run(face='111',
    *,
    cutoff=80,
    clearance=1.5,
    radius=None,
    size=None,
    use_gpu=False,
    output=None,
    ntries=2,
    comparison_k=None):
    out = new_output('Pt' + face, output)
    atoms, system, cs, radius = build(face,
        cutoff=cutoff,
        clearance=clearance,
        radius=radius,
        size=size)
    m = Maps(system,
        make_functions(radius,
            compositional=face == 'defects'),
        cs,
        use_gpu=use_gpu,
        batch_size=256)
    t = time.perf_counter()
    m.atcontactspace()
    seconds = time.perf_counter() - t
    method = 'kmeans' if face == 'defects' else 'spectral'
    summary = analyze(m.data,
        atoms,
        out,
        periodic=True,
        method=method,
        reference_k=comparison_k,
        ntries=ntries,
        title='Pt(111) with vacancies and O' if face == 'defects' else 'Pt(' + face + ')')
    summary.update(face=face,
        atoms=len(atoms),
        pbc=atoms.pbc.tolist(),
        grid_cutoff_ry=cutoff,
        radius_angstrom=radius,
        reference='surface',
        requested_clearance_angstrom=clearance,
        top_atom_z=float(atoms.positions[:,
                2].max()),
        actual_probe_z=[float(cs.data.z.min()),
            float(cs.data.z.max())],
        descriptor_seconds=seconds,
        backend=m.last_backend,
        environment=environment(),
        output=str(out))
    cs.data.drop(columns=['nn']).to_csv(out / 'contact_space.csv', index=False)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    return summary
if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--surface', choices=['111', '100', '110', 'defects'], default='111')
    p.add_argument('--cutoff', type=float, default=80)
    p.add_argument('--clearance', type=float, default=1.5)
    p.add_argument('--radius', type=float)
    p.add_argument('--size', nargs=3, type=int)
    p.add_argument('--gpu', action='store_true')
    p.add_argument('--ntries', type=int, default=2)
    p.add_argument('--comparison-k', type=int)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    print(json.dumps(run(a.surface,
                cutoff=a.cutoff,
                clearance=a.clearance,
                radius=a.radius,
                size=a.size,
                use_gpu=a.gpu,
                output=a.output,
                ntries=a.ntries,
                comparison_k=a.comparison_k),
            indent=2))
