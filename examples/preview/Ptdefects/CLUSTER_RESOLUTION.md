# Why show a finer partition for decorated Pt(111)?

Both original metric curves select **k=2**. That result remains in the gallery. It separates a small, strongly different environment from a much larger group; it does not provide enough resolution for this descriptor map.

On the same 81,000 points and 40 standardized descriptors, k=2 retains only **41.7%** of descriptor variance. Its largest cluster contains **91.7%** of points. All terrace points share one label, as do all vacancy-affected points. Consequently, the partition retains none of the descriptor variation within either of those regions. These findings come from the descriptors and known geometry, without requiring energies.

## A stated error budget

For channel j, measure retained variance as

`R²_j = 1 − Σ_i (x_ij − mean(cluster_i, j))² / Σ_i (x_ij − mean(j))²`.

This is the usual decomposition of variance into within-group and between-group parts. A cluster label represents a point's descriptors more faithfully when its within-cluster residual is smaller. It answers a different question from silhouette, which rewards compact, separated groups.

We tested k=2,4,…,40,50,64, with three seeds and ten K-means initializations per seed. For this comparison, require at least 90% average retained variance globally, 80% average within each descriptor family in every geometry-defined region, and minimum pairwise seed ARI of 0.90. **k=32 is the first tested value to meet all three conditions.** The 90%/80%/0.90 budgets are explicit working choices, not physical constants. Threshold sensitivity is reported in the tables.

| Diagnostic | k=2 | k=10 | k=32 |
| --- | ---: | ---: | ---: |
| Minimum global retention across seeds | 41.7% | 79.9% | 90.9% |
| Minimum region/family retention across seeds | 0.0% | 61.6% | 80.3% |
| Minimum pairwise seed ARI | 0.999 | 0.998 | 0.980 |
| Distinct labels for four exact terrace-site families | 1 | 2 | 3 |

## What remains unresolved

The finer map resolves more variation around vacancies and oxygen, but **fcc and hcp terrace anchors still share a label**. Their descriptors are distinguishable: the standardized distance between their means is 0.876, against within-family differences below 1.3×10⁻¹³. This is a clustering-resolution limitation in this test, not indistinguishable descriptors.

Three 20%-holdout refits retain 90.8–91.0% overall variance at k=32, but assignment ARI ranges from 0.766 to 0.966. The worst regional retention is 79.92–81.06%, narrowly missing the 80% budget in two refits. These are interpolation checks on one correlated spatial grid, not independent structures. The map is a justified higher-resolution comparison, not a unique or completely validated site classification.

For other complex surfaces or proteins, specify the descriptor error you can tolerate, check individual channels and relevant regions, and test seed/refit stability. Where known distinct sites matter, make their separation an additional acceptance requirement. Increasing k alone does not guarantee it. No energy labels are needed for these checks.

[Full-surface comparison](resolution_comparison.png) · [Defect close-ups](defect_zoom.png) · [Metric and fidelity curves](resolution_curves.png) · [Regional curves](regional_fidelity.png) · [Tables, definitions and reproduction](../../measurements/Ptdefects/resolution/README.md).
