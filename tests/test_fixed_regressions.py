"""Native CPU regression suite. No source patching, installs, or GPU emulation."""
import os
import sys
import pathlib
import unittest
import json
import time
import warnings
ROOT = pathlib.Path(__file__).resolve().parents[1]
import tempfile
_TEST_RUNTIME = tempfile.TemporaryDirectory(prefix='mapsy_regression_')
TEST_RUNTIME = pathlib.Path(_TEST_RUNTIME.name)
os.environ['MAPSY_DISABLE_GPU'] = '1'
os.environ['MPLBACKEND'] = 'Agg'
os.environ['MPLCONFIGDIR'] = str(TEST_RUNTIME / 'mplconfig')
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import pandas as pd
from ase import Atoms
from ase.geometry import get_distances
from types import SimpleNamespace
from mapsy.data import Grid, System
from mapsy.maps import Maps
from mapsy.biomapsy import BioSystem, BioMaps, BioContactSpace, AperiodicERFC, create_protein_contact_space, _create_aperiodic_ionic_boundary, _compute_neighbors_and_regions
from mapsy.symfunc.atomic.acsf import ACSymmetryFunction
from mapsy.symfunc.atomic.bpsf import BPSymmetryFunction
from mapsy.utils.gpu_utils import batch_distances_periodic
from mapsy.boundaries.ionic import IonicBoundary
from mapsy.boundaries.contactspace import ContactSpace
from mapsy.utils import get_vdw_radii
from mapsy.utils.multiproc import full2chunk
from mapsy.utils.graph import cluster_graph
TOL = dict(rtol=1e-10, atol=1e-12)

class FixedTests(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(713)
        self.atoms = Atoms('CHO' * 8,
            positions=self.rng.uniform(-2,
                2,
                (24,
                    3)),
            cell=np.eye(3) * 15,
            pbc=True)
        self.points = self.rng.uniform(-3, 3, (32, 3))

    def ac(self, radial=True):
        return ACSymmetryFunction(np.arange(6), 5.0, 'cos', True, True, radial)

    def test_native_pydantic(self):
        from mapsy.symfunc.input import SymFuncModel
        self.assertEqual(SymFuncModel(type='ac', order=4, structural=True).order, [0, 1, 2, 3])

    def test_mic_all_pbc_and_skews(self):
        cells = [np.diag([4.0, 5.0, 6.0]), np.array([[4.0, 0, 0], [3.8, 1, 0], [1.2, 0.7, 6.0]])]
        cells += [self.rng.normal(size=(3, 3)) + np.eye(3) * 3 for _ in range(10)]
        points = self.rng.uniform(-20, 20, (13, 3))
        atoms = self.rng.uniform(-20, 20, (19, 3))
        from itertools import product
        for cell in cells:
            for flags in product([False, True], repeat=3):
                with self.subTest(cell=cell.tolist(), pbc=flags):
                    d, v = batch_distances_periodic(points, atoms, cell, flags)
                    av, ad = get_distances(points, atoms, cell=cell, pbc=flags)
                    np.testing.assert_allclose(d, ad, **TOL)
                    np.testing.assert_allclose(np.linalg.norm(v, axis=-1), d, **TOL)
                    np.testing.assert_allclose(v, av, **TOL)

    def test_mic_singular_partial_and_invalid(self):
        for cell, pbc in [(np.diag([4.0,
                        5.0,
                        0.0]),
                [True,
                    True,
                    False]),
            (np.diag([4.0,
                        0,
                        0]),
                [True,
                    False,
                    False]),
            (np.zeros((3,
                        3)),
                [False] * 3)]:
            d, v = batch_distances_periodic(self.points, self.atoms.positions, cell, pbc)
            av, ad = get_distances(self.points, self.atoms.positions, cell=cell, pbc=pbc)
            np.testing.assert_allclose(d, ad, **TOL)
        with self.assertRaises(ValueError):
            batch_distances_periodic(self.points,
                self.atoms.positions,
                np.zeros((3,
                        3)),
                [True] * 3)

    def test_mic_empty_and_ties(self):
        for p, a in [(np.empty((0, 3)), np.ones((2, 3))), (np.ones((2, 3)), np.empty((0, 3)))]:
            d, v = batch_distances_periodic(p, a, np.eye(3) * 4, [True] * 3)
            self.assertEqual(d.shape, (len(p), len(a)))
        d, v = batch_distances_periodic([[0, 0, 0]], [[2, 0, 0]], np.eye(3) * 4, [True] * 3)
        self.assertEqual(d[0, 0], 2)

    def test_grid_skew_coordinates_and_distances(self):
        cell = np.array([[4.0, 0, 0], [3.8, 1, 0], [0, 0, 6.0]])
        g = Grid(scalars=(3, 4, 5), cell=cell, origin=[2.0, 3.0, 4.0], pbc=[True, True, False])
        np.testing.assert_allclose(g.coordinates[:,
                1,
                2,
                3],
            np.array([1 / 3,
                    2 / 4,
                    3 / 5]) @ cell + g.origin)
        r, r2 = g.get_min_distance(np.array([3.0, 2.0, 1.0]))
        av, ad = get_distances([[3, 2, 1]], g.coordinates.reshape(3, -1).T, cell=cell, pbc=g.pbc)
        np.testing.assert_allclose(r.reshape(3, -1).T, av[0], **TOL)
        np.testing.assert_allclose(r2.reshape(-1), ad[0] ** 2, **TOL)

    def test_maps_vectorized_matches_serial_all_descriptors(self):
        for radial in [True, False]:
            for cell in [np.eye(3) * 15, np.array([[15, 0, 0], [12, 8, 0], [0, 0, 15.0]])]:
                atoms = self.atoms.copy()
                atoms.cell = cell
                s = System(Grid(cell=cell), atoms)
                m = Maps(s, [self.ac(radial)], batch_size=7, use_gpu=False)
                serial = Maps(s, [self.ac(radial)], vectorized=False, use_gpu=False)
                np.testing.assert_allclose(m.atpoints(self.points).to_numpy(dtype=float),
                    serial.atpoints(self.points).to_numpy(dtype=float),
                    **TOL)
        for order in range(1, 6):
            sf = BPSymmetryFunction(order,
                5.0,
                'cos',
                etas=np.array([0.1]),
                lambdas=np.array([-1.0,
                        1.0]),
                zetas=np.array([1,
                        2]))
            s = System(Grid(cell=self.atoms.cell), self.atoms)
            a = Maps(s, [sf], use_gpu=False, batch_size=7)
            b = Maps(s, [sf], use_gpu=False, vectorized=False)
            np.testing.assert_allclose(a.atpoints(self.points).to_numpy(dtype=float),
                b.atpoints(self.points).to_numpy(dtype=float),
                **TOL)

    def test_boundary_idempotent_and_radius_models(self):
        g = Grid(scalars=(12, 12, 12), cell=np.eye(3) * 8)
        s = System(g, Atoms('CH', positions=[[3, 4, 4], [5, 4, 4]], cell=g.cell, pbc=True))
        for mode in ['uff', 'muff', 'bondi', 'pauling', 'ionic']:
            b = IonicBoundary(mode, g, 1.0, 0.5, s)
            b.update()
            gradient = np.asarray(b.gradient).copy()
            switch = np.asarray(b.switch).copy()
            b.update()
            np.testing.assert_allclose(gradient, b.gradient, **TOL)
            np.testing.assert_allclose(switch, b.switch, **TOL)
            self.assertEqual(b.soft_spheres[0].width,
                get_vdw_radii(6,
                    'uff' if mode == 'ionic' else mode))
        before = switch.copy()
        s.atoms.positions[0] += [0.3, 0.2, 0.1]
        b.update()
        self.assertFalse(np.allclose(before, b.switch))

    def test_aperiodic_ionic_gradient_finite_difference(self):

        def field(origin):
            g = Grid(scalars=(2, 2, 2), cell=np.eye(3) * 0.4, origin=origin, pbc=False)
            s = BioSystem(g, Atoms('CH', positions=[[0.0, 0, 0], [3.0, 0.1, 0.2]]))
            b = _create_aperiodic_ionic_boundary(s, g, 0.8, 0.5)
            b.update()
            return (np.asarray(b.switch).copy(), np.asarray(b.gradient).copy())
        o = np.array([1.4, 0.2, 0.3])
        sw, grad = field(o)
        eps = 1e-05
        for d in range(3):
            shift = np.eye(3)[d] * eps
            a, _ = field(o + shift)
            b, _ = field(o - shift)
            np.testing.assert_allclose((a - b) / (2 * eps), grad[d], rtol=2e-07, atol=1e-09)

    def test_aperiodic_erfc_normalization_reset_and_gradient(self):
        g = Grid(scalars=(9, 9, 9), cell=np.eye(3) * 9, pbc=False)
        f = AperiodicERFC(g, 4, 0, 0, 1.0, 0.5, 1.0, np.array([4.0, 4.0, 4.0]))
        self.assertTrue(np.all((f.density >= 0) & (f.density <= 1)))
        old = f.gradient.copy()
        f.reset_derivatives()
        np.testing.assert_allclose(f.gradient, old, **TOL)
        plus = AperiodicERFC(Grid(scalars=(9,
                    9,
                    9),
                cell=g.cell,
                origin=[1e-05,
                    0,
                    0],
                pbc=False),
            4,
            0,
            0,
            1.0,
            0.5,
            1.0,
            np.array([4.0,
                    4.0,
                    4.0]))
        minus = AperiodicERFC(Grid(scalars=(9,
                    9,
                    9),
                cell=g.cell,
                origin=[-1e-05,
                    0,
                    0],
                pbc=False),
            4,
            0,
            0,
            1.0,
            0.5,
            1.0,
            np.array([4.0,
                    4.0,
                    4.0]))
        mask = np.linalg.norm(g.coordinates - np.array([4,
                    4,
                    4])[:,
                None,
                None,
                None],
            axis=0) > 1e-06
        np.testing.assert_allclose(((plus.density - minus.density) / 2e-05)[mask],
            f.gradient[0][mask],
            rtol=1e-06,
            atol=1e-08)

    def test_bio_translation_invariance_contact_and_descriptors(self):
        shift = np.array([20.0, 30.0, 40.0])
        data = []
        for offset in [np.zeros(3), shift]:
            a = Atoms('CH', positions=np.array([[0.0, 0, 0], [2.0, 1.0, 0]]) + offset)
            s = BioSystem(Grid(cell=np.zeros((3, 3)), pbc=False), a)
            for mode in ['tan', 'ionic-ass']:
                cs = create_protein_contact_space(s,
                    mode=mode,
                    cutoff=0.6,
                    grid_resolution=1.0,
                    padding=6.0,
                    distance=2.0,
                    spread=0.5,
                    threshold=0.05)
                self.assertGreater(cs.nm, 0)
                bm = BioMaps(s, [self.ac()], cs, batch_size=11, use_gpu=False)
                df = bm.atcontactspace()
                self.assertTrue(np.isfinite(df.to_numpy()).all())
                data.append((offset, mode, cs, df))
        for i in [0, 1]:
            x, y = (data[i], data[i + 2])
            self.assertEqual(x[2].nm, y[2].nm)
            np.testing.assert_allclose(x[2].positions + shift, y[2].positions, **TOL)
            np.testing.assert_allclose(x[2].probabilities, y[2].probabilities, **TOL)
            np.testing.assert_allclose(x[3].iloc[:, 3:], y[3].iloc[:, 3:], **TOL)

    def test_bio_tree_dense_equivalence_species_alignment(self):
        a = self.atoms.copy()
        a.pbc = False
        s = BioSystem(Grid(cell=a.cell, pbc=False), a)
        sf = [self.ac(),
            self.ac(False)] + [BPSymmetryFunction(o,
                5.0,
                'cos',
                etas=np.array([0.1])) for o in range(1,
                6)]
        cs = BioContactSpace(np.empty((0, 3)), np.empty(0), pd.DataFrame())
        tree = BioMaps(s, sf, cs, batch_size=5, use_gpu=False, use_kdtree=True)
        dense = BioMaps(s, sf, cs, batch_size=5, use_gpu=False, use_kdtree=False)
        pts = np.concatenate([self.points, [[100, 100, 100]]])
        np.testing.assert_allclose(tree.atpoints(pts), dense.atpoints(pts), **TOL)

    def test_bio_tree_refresh_and_caller_ownership(self):
        a = Atoms('C', positions=[[0, 0, 0]], pbc=True)
        s = BioSystem(Grid(cell=np.eye(3)), a)
        self.assertTrue(a.pbc.all())
        self.assertFalse(s.atoms.pbc.any())
        self.assertEqual(s.nearest_distance([[0, 0, 0]])[0], 0)
        s.atoms.positions[0] = [3, 0, 0]
        self.assertEqual(s.nearest_distance([[0, 0, 0]])[0], 3)

    def test_bio_connectivity_and_remapping(self):
        pts = np.array([[0, 0, 0], [1, 0, 0], [100, 0, 0], [101, 0, 0]], float)
        nn, reg = _compute_neighbors_and_regions(pts)
        self.assertEqual(len(np.unique(reg)), 2)
        df = pd.DataFrame(dict(x=[0.0,
                    100,
                    101],
                y=[0.0] * 3,
                z=[0.0] * 3,
                probability=[1.0] * 3,
                region=[0,
                    1,
                    1],
                nn=[(-1,
                        -1),
                    (2,
                        -1),
                    (1,
                        -1)]))
        cs = BioContactSpace(df[['x', 'y', 'z']].to_numpy(), np.ones(3), df, neighbors=list(df.nn))
        cs.keep_main_region()
        self.assertEqual(cs.neighbors, [(1, -1), (0, -1)])
        np.testing.assert_array_equal(cluster_graph([0, 1], cs.neighbors), [[0, 1], [1, 0]])
        np.testing.assert_array_equal(cluster_graph([0, 1], [(1,), ()]), [[0, 1], [1, 0]])

    def test_contact_grid_partial_pbc_and_empty(self):
        for flags, expected in [([True] * 3, 1), ([False] * 3, 2), ([False, True, True], 2)]:
            cs = ContactSpace.__new__(ContactSpace)
            cs.grid = Grid(scalars=(4, 4, 4), cell=np.eye(3) * 4, pbc=flags)
            cs.mask = np.zeros((4, 4, 4), bool)
            cs.mask[0, 1, 1] = cs.mask[3, 1, 1] = True
            cs._get_indexes()
            cs._get_neighbors()
            cs._get_regions()
            self.assertEqual(cs.nregions, expected)
            cs.mask[:] = False
            cs._get_indexes()
            cs._get_neighbors()
            cs._get_regions()
            self.assertEqual(cs.nregions, 0)

    def test_empty_and_invalid_public_inputs(self):
        s = System(Grid(cell=self.atoms.cell), self.atoms)
        m = Maps(s, [self.ac()], use_gpu=False)
        self.assertEqual(m.atpoints(np.empty((0, 3))).shape, (0, 15))
        b = BioMaps(BioSystem(s.grid,
                self.atoms),
            [self.ac()],
            SimpleNamespace(positions=[]),
            use_gpu=False)
        self.assertEqual(b.atpoints(np.empty((0, 3))).shape, (0, 15))
        for x in [np.zeros(3), [[np.nan, 0, 0]]]:
            with self.assertRaises(ValueError):
                m.atpoints(x)
        self.assertEqual(full2chunk(np.empty((0, 3))), [])
        self.assertEqual(len(full2chunk(np.ones((5, 3)), n_jobs=2)), 2)
        with self.assertRaises(ValueError):
            Maps(s, [], batch_size=0)
        with self.assertRaises(NotImplementedError):
            Maps(s, [SimpleNamespace(atomic=False)])

    def test_acsf_origin_policy_angular_exclusion(self):
        sf = self.ac()
        sf.setup(Atoms('CC'))
        with self.assertRaisesRegex(ValueError, 'singular'):
            sf.values(np.array([0.0, 1.0]))
        angular = self.ac(False)
        angular.setup(Atoms('CC'))
        v = angular.values(np.array([0.0, 1.0]), np.array([[0, 0, 0], [1, 0, 0]]))
        self.assertTrue(np.isfinite(v).all())
        np.testing.assert_array_equal(v, np.zeros(12))

    def test_unit_origin_parsers(self):
        from ase.units import Bohr
        from mapsy.io.xyz import XYZParser
        from mapsy.io.cube import CubeParser
        for label, header in [('cubic', '4'), ('ortho', '4 4 4')]:
            f = TEST_RUNTIME / f'test_{label}.xyz'
            f.write_text('1\n' + header + '\nC 1 1 1\n')
            s = XYZParser(str(f), units='bohr').systemparse()
            np.testing.assert_allclose(s.grid.cell, np.eye(3) * 4 * Bohr)
        f = TEST_RUNTIME / 'test.cube'
        f.write_text('test\norigin\n1 2 3 4\n2 1 0 0\n2 0 1 0\n2 0 0 1\n6 0 1 1 1\n0 1 2 3 4 5 6 7\n')
        s = CubeParser(str(f), units='bohr', hasdata=True).systemparse()
        np.testing.assert_allclose(s.grid.origin, np.array([2, 3, 4]) * Bohr)
        np.testing.assert_allclose(s.grid.basis, np.eye(3) * Bohr)
        self.assertFalse(s.grid.pbc.any())

    def test_benchmark_crossovers_and_nan(self):
        from mapsy.benchmark import recommend_settings
        for cpu, gpu, above in [([2.0, 1.0], [1.0, 2.0], False), ([1.0, 2.0], [2.0, 1.0], True)]:
            r = recommend_settings(pd.DataFrame(dict(n_points=[100,
                            1000],
                        cpu_time=cpu,
                        gpu_best_time=gpu,
                        gpu_bs100_time=gpu)))
            self.assertEqual(r['gpu_faster_above'], above)
        r = recommend_settings(pd.DataFrame(dict(n_points=[100,
                        1000],
                    cpu_time=[1.0,
                        2.0],
                    gpu_best_time=[np.nan] * 2,
                    gpu_bs100_time=[np.nan] * 2)))
        self.assertIsNone(r['best_gpu_batch_size'])

    def test_analysis_and_plotting(self):
        import matplotlib.pyplot as plt
        for kind in ['maps', 'bio']:
            cs = SimpleNamespace(data=pd.DataFrame(dict(region=np.zeros(20,
                            int))),
                neighbors=[[(i + 1) % 20,
                        (i - 1) % 20] for i in range(20)],
                nm=20)
            s = System(Grid(cell=self.atoms.cell), self.atoms)
            m = Maps(s,
                [],
                contactspace=cs,
                use_gpu=False) if kind == 'maps' else BioMaps(BioSystem(s.grid,
                    self.atoms),
                [],
                cs,
                use_gpu=False)
            X = self.rng.normal(size=(20, 4))
            X[10:] += 3
            m.data = pd.DataFrame(X, columns=list('abcd'))
            m.features = list('abcd')
            m.data['x'] = self.rng.random(20)
            m.data['y'] = self.rng.random(20)
            m.data['z'] = 0.0
            m.reduce(npca=3, scale=True)
            m.cluster(nclusters=2, random_state=42, scale=True)
            m.sites()
            m.scatter(feature='a', splitby='z')
            m.scatter_pca_grid(feature='a')
            plt.close('all')

    def test_all_active_module_imports_and_compile(self):
        import importlib
        import ast
        historical = {'biomapsy_original',
            'biomapsy_v3',
            'biomapsy_v4',
            'maps_old',
            'maps_original',
            'maps_og'}
        for path in (ROOT / 'src/mapsy').rglob('*.py'):
            ast.parse(path.read_text(), filename=str(path))
            parts = path.relative_to(ROOT / 'src').with_suffix('').parts
            if path.stem in historical or path.name == '__main__.py':
                continue
            name = '.'.join(parts[:-1] if parts[-1] == '__init__' else parts)
            importlib.import_module(name)

    def test_maps_from_file_and_parser_isolation(self):
        import yaml
        from mapsy.maps import MapsFromFile
        from mapsy.io.parser import SystemParser
        from mapsy.io.input.base import SystemModel
        xyz = TEST_RUNTIME / 'pipeline.xyz'
        xyz.write_text('2\n12 12 12\nC 4 6 6\nH 8 6 6\n')
        cfg = {'system': {'systemtype': 'ions',
                'file': {'name': str(xyz),
                    'fileformat': 'xyz+',
                    'units': 'angstrom'}},
            'contactspace': {'mode': 'ionic',
                'radiusmode': 'uff',
                'cutoff': 1.0,
                'alpha': 0.8,
                'spread': 0.5,
                'threshold': 0.05},
            'symmetryfunctions': {'functions': [{'type': 'ac',
                        'radius': 4.0,
                        'order': 4,
                        'radial': True,
                        'structural': True}]}}
        path = TEST_RUNTIME / 'pipeline.yaml'
        path.write_text(yaml.safe_dump(cfg))
        m = MapsFromFile(str(path), use_gpu=False, batch_size=13)
        df = m.atcontactspace()
        self.assertGreater(len(df), 0)
        self.assertTrue(np.isfinite(df.to_numpy(dtype=float)).all())
        m.reduce(npca=2)
        m.cluster(nclusters=2, random_state=42)
        m.sites()
        m.tovolumetric(feature=m.features[0])
        model = SystemModel(**cfg['system'])
        a = SystemParser(model)
        b = SystemParser(model)
        self.assertIsNot(a.propfiles, b.propfiles)

    def test_sparse_large_bio_tree_equivalence(self):
        a = Atoms(['C', 'H', 'O', 'N'] * 500, positions=self.rng.uniform(-100, 100, (2000, 3)))
        s = BioSystem(Grid(cell=np.eye(3) * 210, pbc=False), a)
        cs = SimpleNamespace(positions=[])
        sf = [self.ac(), self.ac(False), BPSymmetryFunction(5, 5.0, 'cos', etas=[0.1])]
        points = a.positions[:16] + [0.5, 0.4, 0.3]
        tree = BioMaps(s, sf, cs, batch_size=8, use_gpu=False, use_kdtree=True)
        dense = BioMaps(s, sf, cs, batch_size=8, use_gpu=False, use_kdtree=False)
        np.testing.assert_allclose(tree.atpoints(points).to_numpy(dtype=float),
            dense.atpoints(points).to_numpy(dtype=float),
            **TOL)

    def test_ionic_ass_translation_with_stored_cell(self):
        results = []
        for shift in [np.zeros(3), np.array([20.0, 30.0, 40.0])]:
            a = Atoms('C', positions=[[0, 0, 0]] + shift, cell=np.eye(3) * 10, pbc=True)
            s = BioSystem(Grid(cell=a.cell), a)
            cs = create_protein_contact_space(s, mode='ionic-ass', cutoff=1.0, threshold=0.05)
            results.append(cs)
        self.assertGreater(results[0].nm, 0)
        np.testing.assert_allclose(results[0].positions + [20, 30, 40], results[1].positions, **TOL)
        np.testing.assert_allclose(results[0].probabilities, results[1].probabilities, **TOL)

    def test_skew_extended_fields_rejected_and_image_constraint_warned(self):
        cell = np.array([[4, 0, 0], [3.8, 1, 0], [0, 0, 6.0]])
        g = Grid(scalars=(3, 4, 5), cell=cell)
        with self.assertRaises(NotImplementedError):
            g.get_min_distance(np.zeros(3), dim=1)
        with self.assertWarnsRegex(UserWarning, 'one minimum image'):
            Maps(System(g,
                    Atoms('C',
                        positions=[[1,
                                1,
                                1]],
                        cell=cell,
                        pbc=True)),
                [self.ac()],
                use_gpu=False)

    def test_cli_and_lazy_exports(self):
        import subprocess
        from mapsy import Maps as exported, BioMaps as exported_bio
        self.assertIs(exported, Maps)
        self.assertIs(exported_bio, BioMaps)
        env = os.environ.copy()
        env['PYTHONPATH'] = str(ROOT / 'src')
        completed = subprocess.run([sys.executable,
                '-m',
                'mapsy',
                'version'],
            env=env,
            text=True,
            capture_output=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_aperiodic_erfc_complement(self):
        g = Grid(scalars=(8, 8, 8), cell=np.eye(3) * 8, pbc=False)
        positive = AperiodicERFC(g, 3, 0, 0, 1.0, 0.5, 1.0, np.array([4, 4, 4]))
        negative = AperiodicERFC(g, 4, 0, 0, 1.0, 0.5, 1.0, np.array([4, 4, 4]))
        np.testing.assert_allclose(positive.density + negative.density, 1.0, **TOL)
        np.testing.assert_allclose(positive.gradient + negative.gradient, 0.0, **TOL)

    def test_bp_degenerate_center_excluded(self):
        for order in [4, 5]:
            sf = BPSymmetryFunction(order, 5.0, 'cos')
            sf.setup(Atoms('CC'))
            np.testing.assert_array_equal(sf.values(np.array([0.0,
                            1.0]),
                    np.array([[0.0,
                                0,
                                0],
                            [1.0,
                                0,
                                0]])),
                np.zeros(len(sf.keys)))
if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(FixedTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
