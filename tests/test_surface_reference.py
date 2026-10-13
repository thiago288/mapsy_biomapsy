import unittest
import numpy as np
from ase.build import fcc111
from mapsy.data import Grid, System
from mapsy.boundaries.system import SystemBoundary
from mapsy.io.parser import ContactSpaceGenerator, ContactSpaceModel

class SurfaceReferenceTests(unittest.TestCase):

    def test_surface_clearance_independent_of_slab_thickness(self):
        for layers in (3, 5, 7):
            atoms = fcc111('Pt', size=(4, 4, layers), orthogonal=True, vacuum=10, periodic=True)
            system = System(Grid(cell=atoms.cell), atoms, dimension=2, axis=2)
            for side in (1, -1):
                cs = ContactSpaceGenerator(ContactSpaceModel(reference='surface',
                        distance=1.5,
                        spread=1,
                        cutoff=20,
                        threshold=-1,
                        side=side)).generate(system)
                target = atoms.positions[:,
                    2].max() + 1.5 if side == 1 else atoms.positions[:,
                    2].min() - 1.5
                self.assertGreater(len(cs.data), 0)
                self.assertLessEqual(float(np.max(abs(cs.data.z - target))),
                    float(cs.grid.cell[2,
                            2] / cs.grid.scalars[2] / 2) + 1e-10)

    def test_legacy_default_and_invalid_reference(self):
        atoms = fcc111('Pt', size=(4, 4, 3), orthogonal=True, vacuum=10, periodic=True)
        grid = Grid(cell=atoms.cell, scalars=np.array([4, 4, 32]))
        system = System(grid, atoms, dimension=2, axis=2)
        b = SystemBoundary('system', grid, 3.5, 1, system)
        np.testing.assert_array_equal(b.reference_position, atoms.get_center_of_mass())
        self.assertEqual(b.reference_width, 3.5)
        with self.assertRaises(ValueError):
            SystemBoundary('system', grid, 1.5, 1, system, reference='invalid')
        with self.assertRaises(ValueError):
            SystemBoundary('system', grid, 20, 1, system, reference='surface')
        with self.assertRaises(ValueError):
            SystemBoundary('system',
                grid,
                1.5,
                1,
                System(grid,
                    atoms,
                    dimension=0),
                reference='surface')
