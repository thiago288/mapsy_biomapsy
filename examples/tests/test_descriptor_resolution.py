"""Check the variance accounting used by the resolution example."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experimental'))
from decorated_resolution import retention


class DescriptorRetentionTests(unittest.TestCase):
    def test_total_variance_decomposition(self):
        values = np.array([[0., 1.], [2., 3.], [8., 4.], [10., 8.]])
        labels = np.array([0, 0, 1, 1])
        centers = np.vstack([values[labels == label].mean(axis=0) for label in [0, 1]])
        total = ((values - values.mean(axis=0)) ** 2).sum(axis=0)
        within = ((values - centers[labels]) ** 2).sum(axis=0)
        between = (2 * (centers - values.mean(axis=0)) ** 2).sum(axis=0)
        np.testing.assert_allclose(total, within + between)
        np.testing.assert_allclose(retention(values, labels), between / total)

    def test_label_names_do_not_change_result(self):
        values = np.random.default_rng(13).normal(size=(80, 5))
        labels = np.repeat(np.arange(4), 20)
        permutation = np.array([3, 1, 0, 2])
        np.testing.assert_allclose(retention(values, labels), retention(values, permutation[labels]))

    def test_nested_refinement_preserves_or_improves_fidelity(self):
        values = np.random.default_rng(29).normal(size=(40, 3))
        coarse = np.repeat(np.arange(2), 20)
        fine = np.repeat(np.arange(4), 10)
        self.assertTrue(np.all(retention(values, fine) >= retention(values, coarse) - 1e-14))
        np.testing.assert_allclose(retention(values, np.arange(40)), 1.)

    def test_constant_channels_and_bad_heldout_centers(self):
        values = np.array([[0., 5.], [2., 5.]])
        result = retention(values, np.array([0, 0]))
        self.assertEqual(result[0], 0.)
        self.assertTrue(np.isnan(result[1]))
        heldout = retention(values, np.array([0, 0]), np.array([[100., 5.]]))
        self.assertLess(heldout[0], 0.)


if __name__ == '__main__':
    unittest.main()
