# Decorated-surface resolution measurements

These results supplement the original k=2 silhouette/Davies–Bouldin partition. They use the same saved 81,000 contact points, 40 AC channels, standardization and K-means method. The descriptor radius is 6.5 Å. The plane is z=22.3698871864 Å, above the oxygen layer; no probe-plane or descriptor change was made for this comparison.

[The short explanation and figures](../../../preview/Ptdefects/CLUSTER_RESOLUTION.md) describe the main findings.

## Why compactness can prefer a coarse partition

Silhouette averages `s_i=(b_i−a_i)/max(a_i,b_i)` over sampled points: a_i is mean distance to the same cluster and b_i is the smallest mean distance to another cluster. It rewards large separation relative to cluster width. A finer split of a continuous environment map can lower b_i and therefore lower silhouette even while its descriptors are represented more accurately. In an average over spatial points, each region's contribution is proportional to its sampled area. The present k=2 fit assigns 91.7% of points to one label and leaves every vacancy-region point in that label.

Davies–Bouldin averages, over clusters, the worst ratio of within-cluster scatter to centroid separation: `DB=(1/k) Σ_c max_(d≠c) (S_c+S_d)/||μ_c−μ_d||`. It weights clusters equally, not grid points directly. Its objection to finer nearby groups comes from increased scatter/separation ratios. Neither score minimizes descriptor reconstruction error. Both original medians favor k=2; the new three-seed sweep still does so among the tested values.

## Variance accounting

For channel j and labels c_i, `SST_j=Σ_i(x_ij−mean_j)²` and `SSE_j=Σ_i(x_ij−mean_(c_i,j))²`. Then `R²_j=1−SSE_j/SST_j`. Conditional means minimize squared residuals for fixed labels, and `SST_j=SSE_j+SSB_j` follows because each cluster's residuals sum to zero. With full-plane standardization, each nonconstant channel has SST=N, so average retained variance is exactly `1−KMeans.inertia_/(N×40)` for these fitted centroids.

This provides a statistical reason to allow more clusters when the purpose is preserving descriptor diversity. It is an information-fidelity criterion, not a new physical metric. Refinement of a fixed nested partition cannot increase its SSE. Independently refitted K-means partitions are not nested, however, so individual channel/region retention and site separation need not improve monotonically. We report the actual measurements rather than assume monotonicity.

A region is marked oxygen-affected or vacancy-affected when the 6.5 Å cutoff sphere at a query intersects an O atom or a missing top-layer Pt position from the ideal parent. Periodic x/y distances and the actual vertical separation are included. The disjoint masks contain 49,079 terrace, 10,108 vacancy and 21,813 oxygen points; none lie in both masks. They are geometric proximity controls, not claims that every point's chemistry is fully described by its mask. They do not enter K-means fitting.

Within each mask we compute conditional channel means for the labels present there and normalize residuals by that region's own variance. This asks whether labels resolve variation inside the region. These conditional means are diagnostics; they are not the global K-means centroids or a modified clustering model. The regional budget applies to each of RS, RC, AS and AC separately, averaging its ten channels. Correlated channels are not independent votes. Individual-channel results are also retained: even at k=32, the worst channel retains about 77.2%, so the budgets do not mean that every channel retains 90%.

## Selection and checks

The tested k values are 2,4,…,40,50,64; seeds are 42,371,829, n_init=10. Every fit and DB calculation uses the full plane. Silhouette uses the same 2,000-row uniform sample with seed 42 as the tutorial. The first two seeds reproduce the original k=2 scores. Odd k values and k>64 were not examined.

The first tested k meeting global mean retention ≥0.90 and worst region/family retention ≥0.80 in all three seeds is k=30. Its minimum pairwise seed ARI is only 0.765. Adding a seed-agreement screen ≥0.90 gives k=32, whose minimum ARI is 0.980. The fidelity budgets and agreement screen are analyst-chosen requirements. The stability screen was added after inspecting the fidelity-only sweep; this is exploratory selection, not a preregistered model-selection result. `budget_sensitivity.csv` shows the effect of changing these choices. Some stricter budgets find no passing candidate.

Three refits each withhold 16,200 randomly chosen grid points. The original scaler is frozen. Held-out global errors use fitted training centroids; regional errors use training-only region/cluster conditional means. Two refits narrowly miss the regional budget, and held-out assignment ARI ranges 0.766–0.966. Spatially correlated rows mean this assesses interpolation/refit sensitivity, not transfer to another geometry or a thermally perturbed surface. The random splits were diagnostic checks after the sweep and did not select k.

Exact terrace anchors are recomputed for atop, bridge, fcc and hcp coordinates outside both affected masks. Periodic duplicate anchors are removed at a 10⁻⁶ Å coordinate-key precision; descriptor values and query coordinates are not rounded. k=2 merges all four families; k=10 separates two; k=32 separates three and merges fcc/hcp. All four descriptor means are distinct, while equivalent anchors agree to roughly 10⁻¹³ in standardized coordinates. Thus increasing resolution improves coverage here but does not satisfy a complete known-site separation requirement.

A fresh current-source calculation at 256 seeded grid points agrees with the archived input to rtol=10⁻¹⁰, atol=10⁻¹²; max absolute difference is 8.17×10⁻¹⁴. The input SHA-256 is in `input_integrity.json`. Four unit tests verify variance decomposition, label-permutation invariance, nested-refinement behavior, and constant-channel/held-out edge cases.

## Files

- `resolution_trials.csv` contains every fit, compactness score, size diagnostic and seed agreement.
- `resolution_summary.csv` contains the k-level summaries used for selection.
- `regional_retention.csv` and `channel_retention.csv` show where descriptor information is lost.
- `holdout_checks.csv` and `budget_sensitivity.csv` record refit and threshold sensitivity.
- `terrace_site_checks.csv`, `terrace_site_distances.csv` and `site_check_conditions.json` record the independent geometry-based check.
- `conditions.json` and `input_integrity.json` identify the workload and input.

Labels and bulky descriptor arrays stay under Git-ignored `examples/results/`; useful tables and figures are published. A cluster color is a nominal label and has no implied numerical or physical order.

## Reproduce

From the repository root, with the package installed:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python examples/experimental/decorated_resolution.py --output examples/results/decorated_resolution
python examples/experimental/decorated_site_control.py examples/results/YOUR_DECORATED_TUTORIAL/descriptors.csv examples/results/decorated_resolution
python -m unittest discover -s examples/tests -v
```

The first command recomputes the tutorial descriptors. Alternatively supply `--descriptors examples/results/YOUR_DECORATED_TUTORIAL/descriptors.csv` to both analysis commands so they use the identical saved dataset. Use a new output directory. No dependencies or data are downloaded by these scripts.

For other systems, reuse the variance calculation but adapt the geometry masks, input checks and budgets. Proteins may need residue, domain, contact-component or independently justified descriptor strata instead of Pt/O proximity masks. Check scaling and channel redundancy, preserve rare regions, test refits and use known-site constraints where available. No energies or new physics-based clustering algorithm are assumed here. This example is not evidence that k=32 is suitable for another system.
