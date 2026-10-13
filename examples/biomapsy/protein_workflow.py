"""Aperiodic calmodulin examples using mapsy.biomapsy (biomapsy.py)."""
from pathlib import Path
import sys
import argparse
import json
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import EXAMPLES, new_output, make_functions, analyze, environment
from physicochemical import add_fields
import numpy as np
from ase.io import read
from mapsy.data import Grid
from mapsy.biomapsy import BioSystem, BioMaps, create_protein_contact_space
STATES = {'1CFD': 'apo (calcium-free source)',
    '1CLL': 'holo-derived (calcium-bound source; ions omitted)'}

def build(protein='1CLL', *, mode='tan', pdb=None, grid_resolution=0.8):
    path = Path(pdb) if pdb else EXAMPLES / 'data' / 'proteins' / (protein + '_protein.pdb')
    atoms = read(path, index=0)
    atoms.pbc = False
    # Preserve PDB coordinates. This box sizes work arrays; it is not periodic.
    box = np.ptp(atoms.positions, axis=0) + 16
    atoms.set_cell(np.diag(box))
    system = BioSystem(Grid(cell=atoms.cell, pbc=False), atoms)
    kwargs = dict(grid_resolution=grid_resolution,
        padding=10,
        distance=3.5,
        spread=0.5,
        threshold=0.05,
        threshold_is_absolute=False) if mode == 'tan' else dict(cutoff=7,
        alpha=0.8,
        spread=0.5,
        threshold=0.7,
        threshold_is_absolute=True)
    if mode not in ('tan', 'ionic-ass'):
        raise ValueError('Protein examples use only the aperiodic tan and ionic-ass modes')
    t = time.perf_counter()
    cs = create_protein_contact_space(system, mode=mode, **kwargs)
    before = len(cs.data)
    regions = cs.data.region.nunique() if len(cs.data) else 0
    cs.keep_main_region()
    seconds = time.perf_counter() - t
    if not len(cs.positions):
        raise AssertionError('Empty contact space')
    if np.any(system.atoms.pbc):
        raise AssertionError('Protein must be aperiodic')
    return (system,
        cs,
        dict(source_state=STATES.get(protein,
                'user supplied') if pdb is None else 'user supplied; apo/holo not inferred',
            pdb_accession=protein if pdb is None else None,
            pdb_path=str(path),
            input_name=path.name,
            atoms=len(atoms),
            pbc=system.atoms.pbc.tolist(),
            contact_mode=mode,
            contact_parameters=kwargs,
            points_before_main_region=before,
            regions_before_main_region=int(regions),
            points_after_main_region=len(cs.positions),
            contact_seconds=seconds,
            coordinates='Original PDB coordinates; no periodic wrapping'))

def run(protein='1CLL',
    *,
    mode='tan',
    pdb=None,
    use_gpu=False,
    output=None,
    radius=10,
    order=12,
    descriptor_set='radial',
    grid_resolution=0.8,
    ntries=2,
    physicochemical=True,
    comparison_k=None):
    label = protein if pdb is None else Path(pdb).stem
    out = new_output(label + '_' + mode, output)
    system, cs, summary = build(protein, mode=mode, pdb=pdb, grid_resolution=grid_resolution)
    m = BioMaps(system,
        make_functions(radius,
            order,
            True,
            descriptor_set),
        cs,
        use_gpu=use_gpu,
        use_kdtree=not use_gpu,
        batch_size=256)
    t = time.perf_counter()
    m.atcontactspace()
    seconds = time.perf_counter() - t
    data = m.data
    if physicochemical:
        tfields = time.perf_counter()
        data, fields_meta = add_fields(data, system.atoms, summary['pdb_path'])
        summary['physicochemical_fields'] = fields_meta
        summary['physicochemical_seconds'] = time.perf_counter() - tfields
        summary['physicochemical_backend'] = 'numpy_cpu'
    summary.update(analyze(data,
            system.atoms,
            out,
            periodic=False,
            method='kmeans',
            reference_k=comparison_k,
            ntries=ntries,
            title=label + ' — ' + summary['source_state'] + ' — ' + mode))
    summary.update(radius_angstrom=radius,
        orders=list(range(order)),
        descriptor_set=descriptor_set,
        descriptor_seconds=seconds + summary.get('physicochemical_seconds',
            0),
        ac_descriptor_seconds=seconds,
        backend=m.last_backend,
        environment=environment(),
        output=str(out))
    cs.data.drop(columns=['nn'], errors='ignore').to_csv(out / 'contact_space.csv', index=False)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    return summary
if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protein', choices=['1CLL', '1CFD'], default='1CLL')
    p.add_argument('--pdb', type=Path)
    p.add_argument('--mode', choices=['tan', 'ionic-ass'], default='tan')
    p.add_argument('--gpu', action='store_true')
    p.add_argument('--radius', type=float, default=10)
    p.add_argument('--order', type=int, default=12)
    p.add_argument('--descriptor-set', choices=['radial', 'mixed'], default='radial')
    p.add_argument('--grid-resolution', type=float, default=0.8)
    p.add_argument('--output', type=Path)
    p.add_argument('--ntries', type=int, default=2)
    p.add_argument('--no-physicochemical', action='store_true')
    p.add_argument('--comparison-k', type=int)
    a = p.parse_args()
    print(json.dumps(run(a.protein,
                mode=a.mode,
                pdb=a.pdb,
                use_gpu=a.gpu,
                output=a.output,
                radius=a.radius,
                order=a.order,
                descriptor_set=a.descriptor_set,
                grid_resolution=a.grid_resolution,
                ntries=a.ntries,
                physicochemical=not a.no_physicochemical,
                comparison_k=a.comparison_k),
            indent=2))
