# Measured results

These tables and figures accompany executed examples. They are recorded workloads, not universal rankings. Main measurements were freshly rerun for package integration; the experimental cluster-resolution pilot and original-source regression are retained as separately identified measurements.

## Main example conditions

Classical surfaces use the original center-of-mass3.5Å probe and comparison resolutions10/10/4. Revised surfaces use surface clearance1.5Å. Ideal faces: radius4.5Å, orders0–9,20 structural radial/angular channels. Decorated surface: radius6.5Å,40 structural/compositional radial/angular channels. Protein datasets: radius10Å, orders0–11,24 radial AC plus2 physicochemical fields; both states and both aperiodic boundaries. Every fit/DB uses all points; silhouette above6000 rows uses2000 seeded points; seeds42/371, median selection. Main K-means n_init10; spectral raw RBF gamma1.

Directory-specific summary.json stores exact settings, software versions, query heights/region counts, choices and field provenance. clustering_metrics.csv stores every trial; clustering_medians.csv stores both median curves. pca_variance.csv, pca_loadings.csv and feature_summary.csv make figures auditable. Protein opposite views display up to12000 seeded points with2–98% field color limits.

## Original-source descriptor regression

surface,points,features,baseline_vs_fixed_max_abs,optimized_vs_fixed_max_abs,baseline_allclose,optimized_allclose
Pt111,3240,20,1.557087792036782e-14,8.881784197001252e-16,True,True
Pt100,3600,20,4.440892098500626e-15,6.661338147750939e-16,True,True
Pt110,5400,20,1.8423605555262235e-05,2.220446049250313e-16,False,True

Optimized and fixed implementations agree at identical notebook coordinates (max8.88e-16 across three surfaces). The older baseline agrees for Pt111/Pt100 but differs in Pt110 radial channels by1.84e-5: its fc>1e-6 mask discards near-cutoff terms retained by the vectorized implementation. Direct reconstruction explains the discrepancy to1.40e-15; matched K-means/spectral controls at k2/4/10, seed42 have ARI1.0. See descriptor_regression/cutoff_diagnosis.json and per-surface paired plots.

The historical notebook has no stored outputs. Older source imports require the bundled Pydantic.v1 namespace, and its contact generator fails (missing Grid.local_blocks). Its descriptor engine was tested on identical externally supplied coordinates; this is not a successful historical end-to-end notebook run. The tested baseline is the user's older local snapshot, not a verified official GitHub commit.

## CPU/GPU

benchmark.csv/json contain fresh2048-point-per-case AC measurements on identical inputs, one warm-up, three synchronized repetitions; contact construction and the CPU physicochemical fields are excluded. Reported speedups apply only where the measured backend is GPU. The benchmark notebook shows CPU and GPU timings and tests descriptor, shared-PCA and label-permutation-invariant clustering agreement. [Figures](../preview/cpu_gpu/README.md).

## Experimental resolution study

[Experimental tables, plots and portable scripts](../experimental/README.md) report coverage and stability separately from geometric metric choices. The prototype is not a new validated metric. It uses n_init5 and different silhouette sample size from the main tutorials; its scores must not be silently substituted for main results.

## Classical Pt111 known-site control

The original center-of-mass probe model was additionally checked with the portable site script. The k10 comparison separates all four checked anchor families at seed42, whereas both k2 metric choices merge them. Site-discrimination scores are1.0 versus0.5; within-family equivalence is1.0 for both. See classical_Pt111/site_assignments.csv and recorded nearest-grid offsets. This is a site-anchor acceptance control, not a validation of adsorption energies or a universal recommended k.

## Decorated-surface resolution comparison

A new [three-seed sweep](Ptdefects/resolution/README.md) compares k=2 through k=64 on the original 81,000-point decorated-surface descriptors. It records variance retained per channel and region, seed agreement, held-out refits and exact terrace-site checks. The [k=32 comparison figures](../preview/Ptdefects/CLUSTER_RESOLUTION.md) supplement the original metric-selected k=2 map. Both results remain visible.
