import unittest
import sys
import json
import hashlib
from pathlib import Path
import numpy as np
EXAMPLES = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXAMPLES), str(EXAMPLES / 'periodic_surfaces'), str(EXAMPLES / 'biomapsy')]
from common import make_functions
from corrected_surfaces import build as build_surface
from protein_workflow import build as build_protein
from mapsy.maps import Maps
from mapsy.biomapsy import BioMaps

class ExampleTests(unittest.TestCase):

    def test_bundled_provenance_hashes(self):
        provenance = json.loads((EXAMPLES / 'data/proteins/provenance.json').read_text())
        for name, meta in provenance.items():
            self.assertEqual(hashlib.sha256((EXAMPLES / 'data/proteins' / (name + '_protein.pdb')).read_bytes()).hexdigest(),
                meta['sha256'])

    def test_periodic_translation_and_external_plane(self):
        atoms, system, cs, radius = build_surface('111', cutoff=20)
        self.assertTrue(np.all(atoms.pbc))
        self.assertGreater(cs.data.z.min(), atoms.positions[:, 2].max())
        points = cs.data[['x', 'y', 'z']].to_numpy()[::max(1, len(cs.data) // 32)][:32]
        model = Maps(system, make_functions(radius), cs, use_gpu=False, batch_size=16)
        a = model.atpoints(points).filter(regex='^ACSF_').to_numpy(float)
        b = model.atpoints(points + atoms.cell[0]).filter(regex='^ACSF_').to_numpy(float)
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-10)

    def test_protein_aperiodic_distances_and_dense_tree_agreement(self):
        system, cs, meta = build_protein('1CLL', mode='tan', grid_resolution=1.6)
        self.assertFalse(np.any(system.atoms.pbc))
        points = cs.positions[::max(1, len(cs.positions) // 32)][:32]
        distances = np.linalg.norm(system.atoms.positions[None,
                :,
                :] - points[:,
                None,
                :],
            axis=-1).min(axis=1)
        np.testing.assert_allclose(system.nearest_distance(points),
            distances,
            rtol=1e-12,
            atol=1e-12)
        models = [BioMaps(system,
                make_functions(8,
                    6,
                    True,
                    'mixed'),
                cs,
                use_gpu=False,
                use_kdtree=tree,
                batch_size=16) for tree in [True,
                False]]
        a, b = [m.atpoints(points).filter(regex='^ACSF_').to_numpy(float) for m in models]
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-10)

    def test_physicochemical_fields_match_scalar_notebook_formulas(self):
        from physicochemical import fields
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [9.0, 0.0, 0.0]])
        charge = np.array([0.5, -0.8, 0.2])
        hydro = np.array([1.8, -3.5, 4.2])
        points = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
        e, h = fields(positions, charge, hydro, points, batch_size=2)
        for i, p in enumerate(points):
            r = np.linalg.norm(positions - p, axis=1)
            w = 1 / np.maximum(r, 0.1)
            me = r <= 8
            mh = r <= 5
            self.assertAlmostEqual(e[i], float(np.sum(charge[me] * w[me])), places=12)
            self.assertAlmostEqual(h[i],
                float(np.sum(hydro[mh] * w[mh]) / w[mh].sum()) if mh.any() else 0.0,
                places=12)

    def test_physicochemical_pdb_alignment_and_variation(self):
        from physicochemical import atom_properties, fields
        for accession in ['1CFD', '1CLL']:
            system, cs, meta = build_protein(accession, mode='tan', grid_resolution=1.6)
            q, h, audit = atom_properties(system.atoms,
                EXAMPLES / 'data/proteins' / (accession + '_protein.pdb'))
            e, hyd = fields(system.atoms.positions,
                q,
                h,
                cs.positions[::max(1,
                        len(cs.positions) // 128)])
            self.assertEqual(len(q), len(system.atoms))
            self.assertGreater(np.std(e), 1e-06)
            self.assertGreater(np.std(hyd), 1e-06)
