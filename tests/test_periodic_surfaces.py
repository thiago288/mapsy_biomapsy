"""Small physical periodic-surface regressions; CPU by default, GPU optional.

Run: python -m unittest discover -s tests -p test_periodic_surfaces.py -v
Uses installed mapsy and repository-local example fixtures; never installs deps.
"""
import os
import pathlib
import importlib.util
import unittest
import warnings
os.environ.setdefault('MPLBACKEND', 'Agg')
import numpy as np
from ase.geometry import get_distances
from ase.build import fcc111
from mapsy.data import Grid, System
from mapsy.maps import Maps
from mapsy.utils.gpu_utils import batch_distances_periodic, GPU_AVAILABLE
ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('periodic_example',
    ROOT / 'examples/periodic_surfaces/surface_workflow.py')
ex = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ex)
TOL = {'rtol': 1e-10, 'atol': 1e-12}

class PeriodicSurfaceTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.fixtures = {}
        for face in ['111', '100', '110', 'defects']:
            atoms = ex.make_surface(face, size=(8, 8, 3) if face == 'defects' else None)
            system, cs = ex.make_contact_space(atoms, cutoff=8)
            allp = cs.data[['x', 'y', 'z']].to_numpy()
            idx = np.linspace(0, len(allp) - 1, 24, dtype=int)
            cls.fixtures[face] = (atoms, system, cs, allp[idx])

    def model(self, face, gpu=False):
        atoms, s, cs, p = self.fixtures[face]
        return Maps(s, ex.make_functions(face == 'defects'), cs, use_gpu=gpu, batch_size=16)

    def test_nonempty_finite_actual_surfaces(self):
        for face, (atoms, s, cs, p) in self.fixtures.items():
            with self.subTest(face=face):
                self.assertGreater(len(cs.data), 24)
                self.assertTrue(atoms.pbc.all())
                self.assertTrue(np.all(cs.data.z > atoms.get_center_of_mass()[2]))
                vals = self.model(face).atpoints(p).drop(columns=['x',
                        'y',
                        'z']).to_numpy(dtype=float)
                self.assertTrue(np.isfinite(vals).all())
                self.assertEqual(vals.shape[1], 40 if face == 'defects' else 20)

    def test_scalar_ase_reference_radial_and_angular(self):
        for face, (atoms, s, cs, p) in self.fixtures.items():
            with self.subTest(face=face):
                m = self.model(face)
                a = m.atpoints(p).drop(columns=['x', 'y', 'z']).to_numpy(dtype=float)
                v, d = get_distances(p, atoms.positions, cell=atoms.cell, pbc=atoms.pbc)
                ref = np.concatenate([np.array([sf.values(d[i],
                                    v[i]) if sf.angular else sf.values(d[i]) for i in range(len(p))]) for sf in m.symmetryfunctions],
                    axis=1)
                np.testing.assert_allclose(a, ref, **TOL)

    def test_descriptor_invariance_under_lattice_translation(self):
        for face, (atoms, s, cs, p) in self.fixtures.items():
            with self.subTest(face=face):
                m = self.model(face)
                a = m.atpoints(p).drop(columns=['x', 'y', 'z']).to_numpy(dtype=float)
                for shift in [atoms.cell[0], atoms.cell[1], atoms.cell[0] - 2 * atoms.cell[1]]:
                    b = m.atpoints(p + shift).drop(columns=['x', 'y', 'z']).to_numpy(dtype=float)
                    np.testing.assert_allclose(a, b, **TOL)

    def test_periodic_neighbors_wrap_surface_edges(self):
        for face, (_, s, cs, p) in self.fixtures.items():
            with self.subTest(face=face):
                neighbors = np.asarray(cs.neighbors)
                n = len(cs.data)
                self.assertTrue(np.all((neighbors >= -1) & (neighbors < n)))
                wrapped = False
                for i, row in enumerate(neighbors):
                    for j in row[row >= 0]:
                        self.assertIn(i, neighbors[j])
                        delta = np.abs(cs.i2m[i] - cs.i2m[j])
                        wrapped |= bool(np.any(delta[:2] > 1))
                self.assertTrue(wrapped, 'Expected in-plane periodic seam edges')

    def test_original_decorated_geometry(self):
        atoms = ex.make_surface('defects')
        z = atoms.get_atomic_numbers()
        self.assertEqual(int(np.sum(z == 78)), 1992)
        self.assertEqual(int(np.sum(z == 8)), 29)

    def test_defect_fixture_has_oxygen_and_removed_platinum(self):
        atoms = self.fixtures['defects'][0]
        z = atoms.get_atomic_numbers()
        self.assertGreater(np.sum(z == 8), 0)
        self.assertLess(np.sum(z == 78), 8 * 8 * 3)

    def test_skew_primitive_surface_descriptor_ase_reference(self):
        atoms = fcc111('Pt', size=(4, 4, 3), a=3.94, orthogonal=False, periodic=True, vacuum=10)
        p = np.array([[0.001, 0.15, 0], [0.999, 0.35, 0], [0.41, 0.69, 0]]) @ np.array(atoms.cell)
        p[:, 2] = atoms.positions[:, 2].max() + 1.5
        system = System(Grid(cell=atoms.cell), atoms, dimension=0, axis=2)
        m = Maps(system, ex.make_functions(), use_gpu=False, batch_size=2)
        a = m.atpoints(p).drop(columns=['x', 'y', 'z']).to_numpy(dtype=float)
        v, d = get_distances(p, atoms.positions, cell=atoms.cell, pbc=atoms.pbc)
        ref = np.concatenate([np.array([sf.values(d[i],
                            v[i]) if sf.angular else sf.values(d[i]) for i in range(len(p))]) for sf in m.symmetryfunctions],
            axis=1)
        np.testing.assert_allclose(a, ref, **TOL)

    def test_small_skew_slab_cutoff_sensitive_images(self):
        atoms = fcc111('Pt', size=(2, 2, 3), a=3.94, orthogonal=False, periodic=True, vacuum=10)
        center = atoms.positions[np.argmax(atoms.positions[:, 2])]
        p = np.array([center + 0.51 * atoms.cell[0] - 0.255 * atoms.cell[1] + [0,
                    0,
                    0.5],
                center - 0.255 * atoms.cell[0] + 0.51 * atoms.cell[1] + [0,
                    0,
                    0.5]])
        system = System(Grid(cell=atoms.cell), atoms)
        m = Maps(system, ex.make_functions(radius=2.7), use_gpu=False, batch_size=2)
        vals = m.atpoints(p).drop(columns=['x', 'y', 'z']).to_numpy(dtype=float)
        v, d = get_distances(p, atoms.positions, cell=atoms.cell, pbc=atoms.pbc)
        ref = np.concatenate([np.array([sf.values(d[i],
                            v[i]) if sf.angular else sf.values(d[i]) for i in range(len(p))]) for sf in m.symmetryfunctions],
            axis=1)
        np.testing.assert_allclose(vals, ref, **TOL)
        # Radius 2.7Å stays below half the shortest in-plane lattice vector.
        self.assertLess(2.7, np.linalg.norm(atoms.cell[0]) / 2)

    def test_skew_planar_boundary_explicitly_unsupported(self):
        atoms = fcc111('Pt', size=(4, 4, 3), a=3.94, orthogonal=False, periodic=True, vacuum=10)
        with self.assertRaises(NotImplementedError):
            ex.make_contact_space(atoms, cutoff=8)

    def test_partial_periodicity_distances(self):
        atoms, s, cs, p = self.fixtures['111']
        flags = [True, True, False]
        d, v = batch_distances_periodic(p, atoms.positions, atoms.cell, flags)
        av, ad = get_distances(p, atoms.positions, cell=atoms.cell, pbc=flags)
        np.testing.assert_allclose(d, ad, **TOL)
        np.testing.assert_allclose(np.linalg.norm(v, axis=-1), ad, **TOL)
        # At exact half-cell ties ASE may select another equally short image.
        delta = (v - av) @ np.linalg.inv(np.asarray(atoms.cell))
        np.testing.assert_allclose(delta[:, :, :2], np.round(delta[:, :, :2]), atol=1e-12)
        np.testing.assert_allclose(delta[:, :, 2], 0, atol=1e-12)

    @unittest.skipUnless(GPU_AVAILABLE, 'CuPy/CUDA device unavailable; CPU-only CI is supported')
    def test_cpu_gpu_actual_surface_descriptors(self):
        for face, (_, s, cs, p) in self.fixtures.items():
            with self.subTest(face=face), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                cpu = self.model(face).atpoints(p).drop(columns=['x',
                        'y',
                        'z']).to_numpy(dtype=float)
                gm = self.model(face, True)
                gpu = gm.atpoints(p).drop(columns=['x', 'y', 'z']).to_numpy(dtype=float)
                self.assertEqual(gm.last_backend, 'gpu')
                self.assertFalse(any(('falling back' in str(x.message).lower() or 'fallback' in str(x.message).lower() for x in caught)))
                np.testing.assert_allclose(cpu, gpu, **TOL)
if __name__ == '__main__':
    unittest.main()
