"""Classical periodic notebook inputs, with an explicit comparison resolution."""
from pathlib import Path
import sys
import json
import argparse
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import new_output, make_functions, analyze, environment
from structures import make_surface
from mapsy.data import Grid, System
from mapsy.io.parser import ContactSpaceGenerator, ContactSpaceModel
from mapsy.maps import Maps

def run(face='111', *, use_gpu=False, output=None, ntries=2, comparison_k=None):
    if face not in ('111', '100', '110'):
        raise ValueError('Classical ideal faces: 111,100,110')
    atoms = make_surface(face)
    system = System(Grid(cell=atoms.cell), atoms, dimension=2, axis=2)
    settings = dict(mode='system', distance=3.5, spread=1.0, cutoff=80, threshold=-1, side=1)
    cs = ContactSpaceGenerator(ContactSpaceModel.parse_obj(settings)).generate(system)
    out = new_output('classical_Pt' + face, output)
    m = Maps(system, make_functions(4.5), cs, use_gpu=use_gpu, batch_size=256)
    t = time.perf_counter()
    m.atcontactspace()
    seconds = time.perf_counter() - t
    k = comparison_k if comparison_k is not None else 4 if face == '110' else 10
    summary = analyze(m.data,
        atoms,
        out,
        periodic=True,
        method='spectral',
        reference_k=k,
        ntries=ntries,
        title='Classical Pt(' + face + ')')
    summary.update(face=face,
        atoms=len(atoms),
        reference='center_of_mass',
        contact_parameters=settings,
        radius_angstrom=4.5,
        orders=list(range(10)),
        actual_probe_z=sorted(m.data.z.unique().tolist()),
        top_atom_z=float(atoms.positions[:,
                2].max()),
        comparison_reason='Resolution used in the original planar_maps_ideal notebook; not an automatically selected optimum',
        descriptor_seconds=seconds,
        backend=m.last_backend,
        environment=environment(),
        output=str(out))
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    return summary
if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--surface', choices=['111', '100', '110'], default='111')
    p.add_argument('--gpu', action='store_true')
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    print(json.dumps(run(a.surface, use_gpu=a.gpu, output=a.output), indent=2))
