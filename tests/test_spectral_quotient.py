"""Check population-weighted reduction against the independent dense reference."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.metrics.pairwise import euclidean_distances
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('decorated_clustering',
    ROOT / 'examples/periodic_surfaces/decorated_clustering.py')
dc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dc)

class SpectralQuotientTests(unittest.TestCase):

    def test_population_weighted_clusters_match_dense_reference(self):
        X = np.repeat(np.random.default_rng(917).normal(size=(16,
                    5)),
            np.arange(1,
                17) % 5 + 1,
            axis=0)
        embedding, inverse, counts, U, evidence = dc.quotient_embedding(X, maxclusters=4)
        distances = euclidean_distances(U)
        np.fill_diagonal(distances, 0)
        for k in [2, 3, 4]:
            with self.subTest(k=k):
                labels = KMeans(n_clusters=k,
                    n_init=10,
                    random_state=42).fit_predict(embedding[inverse,
                        :k])
                reference = SpectralClustering(n_clusters=k,
                    random_state=42,
                    eigen_tol=1e-10).fit_predict(X)
                self.assertEqual(adjusted_rand_score(labels, reference), 1)
                group = labels[np.unique(inverse, return_index=True)[1]]
                self.assertAlmostEqual(dc.weighted_silhouette(distances,
                        group,
                        counts),
                    silhouette_score(X,
                        labels),
                    places=8)
        self.assertEqual(evidence['all_points'], len(X))
        self.assertLess(evidence['groups'], len(X))
        self.assertLess(evidence['maximum_eigenvector_residual'], 1e-08)
if __name__ == '__main__':
    unittest.main()
