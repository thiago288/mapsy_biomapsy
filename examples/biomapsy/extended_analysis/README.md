# Verified extended calmodulin analysis

[Run the notebook](02_extended_calmodulin.ipynb). It completed from a fresh kernel in the existing `mapsy_opt` environment: **53/53 executable code cells (three empty cells skipped), zero execution errors, 840.8 seconds**. [Execution record](execution.json), [numerical comparison](numerical_comparison.json), [scientific checks](scientific_checks.json), [both clustering metrics for k=2–25](clustering_metrics.csv).

Inputs: **apo 1CFD**, **holo-derived 1CLL**; protein heavy atoms, calcium and water omitted. This reproduces the legacy **aperiodic `Maps`/`ionic`** workflow. The main BioMapSy tutorial separately uses active `biomapsy.py` with `tan` and `ionic-ass`.

Descriptors: structural/compositional radial AC at 10 Å/order 12 and 12 Å/order 14, **52 AC coefficients**, plus electrostatic and hydrophobicity proxies. Contact construction and AC formulas are preserved. Coordinates are shifted to the bounding-box minimum plus 8 Å, as in the original extended notebook; this translation does not impose periodicity. Alternate-location ATOM records are retained to match the original descriptor inputs. All data are bundled; no downloads or dependency installation. New runs write to fresh `examples/results/extended_*` folders. Launch from this notebook's folder or the repository root.

## Dependencies and settings

Optional dependencies are listed in [requirements-extended.txt](requirements-extended.txt); [verified environment versions](verified_environment.json) record the successful run. This run used four analysis workers and one BLAS thread. Spectral clustering uses the 52 AC channels, apo-fitted scaling applied to both states, 5,000 sampled points per protein, 15 nearest neighbors, n_init=5, seed=42, and k=2–25. Silhouette uses a 2,000-point seeded subset; DB uses all 5,000. Full pooled PCA and classification use 54 channels including both fields.

The run emitted warnings about seeded UMAP using one thread and an upcoming scikit-learn quantile-default change; these are recorded in [execution warnings](execution_warnings.json). The verified package versions above identify the tested behavior.

## Verification and corrections

- Both proteins are explicitly aperiodic; every one of the 54 channels is finite.
- Full AC matrices agree with the previously saved extended results at rtol=1e-10, atol=1e-10 on matching coordinates; the measured maximum AC difference is exactly zero for both proteins.
- Physicochemical assignment now checks exact PDB-record/ASE coordinate order. Bio.PDB collapses alternate locations in 1CLL (1,130 records versus ASE's 1,133), so the historical index-based lookup was unsafe. The holo-derived electrostatic and hydrophobicity fields change (maximum differences 1.54518 and 6.44399 in their respective proxy units); apo fields are unchanged. Independent evaluation of 512 points per protein agrees with the tested batched formulas within 2.23e-15; see [field validation](independent_field_validation.json).
- Corrected PCA dimensionality, feature-name parsing, file references, accession labels and feature-count captions. Charge and hydrophobicity lookup constants are unchanged.
- Classification scaling is fitted inside each training fold. Classification, in-sample permutation importance, LDA separation and KS comparisons are exploratory for these two structures: neighboring samples are correlated, and they do not validate generalization to independent proteins.
- A nearest-neighbor graph describes geometric connectivity, not energetic transitions. Independent UMAP fits have separate coordinate bases.

## Extended results

![Full standardized PCA explained variance](figures/figure_002.png)

This notebook runs feature distributions/correlations, full standardized PCA, pooled and separate UMAP analyses, silhouette and Davies–Bouldin sweeps, spatial partitions, geometric connectivity, cluster/physicochemical comparisons, LDA and classification diagnostics, descriptor-to-field regression, and KS distribution comparisons.

### Apo shared-basis PCA maps

![Apo shared-basis PCA maps](figures/apo_extended_shared_pca.png)

### Holo-derived shared-basis PCA maps

![Holo-derived shared-basis PCA maps](figures/holo_extended_shared_pca.png)

### Apo corrected physicochemical fields

![Apo corrected physicochemical fields](figures/apo_extended_fields.png)

### Holo-derived corrected physicochemical fields

![Holo-derived corrected physicochemical fields](figures/holo_extended_fields.png)

### Both clustering metrics

![Both clustering metrics](figures/optimal_k_sweep.png)

### Silhouette-selected spatial partitions

![Silhouette-selected spatial partitions](figures/clusters_best_silhouette.png)

### Davies–Bouldin-selected spatial partitions

![Davies–Bouldin-selected spatial partitions](figures/clusters_best_db.png)

### Connectivity, silhouette choice

![Connectivity, silhouette choice](figures/connectivity_sil.png)

### Connectivity, DB choice

![Connectivity, DB choice](figures/connectivity_db.png)

### Physicochemical variation by cluster

![Physicochemical variation by cluster](figures/physicochemical_per_cluster_boxplot.png)

### Descriptor variation by cluster

![Descriptor variation by cluster](figures/descriptor_per_cluster_boxplot.png)

### UMAP and partitions

![UMAP and partitions](figures/umap_cluster_comparison.png)

### Shared UMAP analysis

![Shared UMAP analysis](figures/umap_full_features.png)

### Separate UMAP analyses

![Separate UMAP analyses](figures/umap_full_features_corrected.png)

### Exploratory classification

![Exploratory classification](figures/roc_feature_importance.png)

### LDA distribution

![LDA distribution](figures/lda_distribution.png)

### LDA coefficients

![LDA coefficients](figures/lda_coefficients.png)

### Descriptor distribution comparisons

![Descriptor distribution comparisons](figures/ks_all_features_bar.png)

### KS empirical distribution comparisons

![KS empirical distribution comparisons](figures/ks_cdf_top_bottom.png)

## Reproduce the static spatial gallery

```bash
python examples/biomapsy/extended_analysis/spatial_gallery.py examples/results/extended_YOUR_RUN
```

This plots all 52 AC channels, both fields and four shared-basis PCs for each protein. Rerunning the gallery records display sampling and pooled color limits alongside the newly generated figures. These regenerable rendering records are omitted from the bundled examples.

## All generated static figures

- [3d_clusters_db](figures/3d_clusters_db.png)
- [3d_clusters_sil](figures/3d_clusters_sil.png)
- [apo_extended_descriptors_01](figures/apo_extended_descriptors_01.png)
- [apo_extended_descriptors_02](figures/apo_extended_descriptors_02.png)
- [apo_extended_descriptors_03](figures/apo_extended_descriptors_03.png)
- [apo_extended_descriptors_04](figures/apo_extended_descriptors_04.png)
- [apo_extended_descriptors_05](figures/apo_extended_descriptors_05.png)
- [apo_extended_descriptors_06](figures/apo_extended_descriptors_06.png)
- [apo_extended_descriptors_07](figures/apo_extended_descriptors_07.png)
- [apo_extended_descriptors_08](figures/apo_extended_descriptors_08.png)
- [apo_extended_descriptors_09](figures/apo_extended_descriptors_09.png)
- [apo_extended_descriptors_10](figures/apo_extended_descriptors_10.png)
- [apo_extended_descriptors_11](figures/apo_extended_descriptors_11.png)
- [apo_extended_descriptors_12](figures/apo_extended_descriptors_12.png)
- [apo_extended_descriptors_13](figures/apo_extended_descriptors_13.png)
- [apo_extended_fields](figures/apo_extended_fields.png)
- [apo_extended_shared_pca](figures/apo_extended_shared_pca.png)
- [calmodulin_teaser](figures/calmodulin_teaser.png)
- [cluster_vs_physicochemical_db](figures/cluster_vs_physicochemical_db.png)
- [cluster_vs_physicochemical_sil](figures/cluster_vs_physicochemical_sil.png)
- [clusters_best_db](figures/clusters_best_db.png)
- [clusters_best_silhouette](figures/clusters_best_silhouette.png)
- [connectivity_db](figures/connectivity_db.png)
- [connectivity_sil](figures/connectivity_sil.png)
- [descriptor_per_cluster_boxplot](figures/descriptor_per_cluster_boxplot.png)
- [figure_001](figures/figure_001.png)
- [figure_002](figures/figure_002.png)
- [figure_003](figures/figure_003.png)
- [figure_004](figures/figure_004.png)
- [figure_005](figures/figure_005.png)
- [figure_006](figures/figure_006.png)
- [figure_007](figures/figure_007.png)
- [figure_008](figures/figure_008.png)
- [figure_009](figures/figure_009.png)
- [figure_010](figures/figure_010.png)
- [figure_011](figures/figure_011.png)
- [figure_012](figures/figure_012.png)
- [figure_013](figures/figure_013.png)
- [figure_014](figures/figure_014.png)
- [figure_015](figures/figure_015.png)
- [figure_016](figures/figure_016.png)
- [figure_017](figures/figure_017.png)
- [figure_018](figures/figure_018.png)
- [figure_019](figures/figure_019.png)
- [figure_020](figures/figure_020.png)
- [figure_021](figures/figure_021.png)
- [figure_022](figures/figure_022.png)
- [figure_023](figures/figure_023.png)
- [figure_024](figures/figure_024.png)
- [figure_025](figures/figure_025.png)
- [figure_026](figures/figure_026.png)
- [figure_027](figures/figure_027.png)
- [figure_028](figures/figure_028.png)
- [figure_029](figures/figure_029.png)
- [figure_030](figures/figure_030.png)
- [holo_extended_descriptors_01](figures/holo_extended_descriptors_01.png)
- [holo_extended_descriptors_02](figures/holo_extended_descriptors_02.png)
- [holo_extended_descriptors_03](figures/holo_extended_descriptors_03.png)
- [holo_extended_descriptors_04](figures/holo_extended_descriptors_04.png)
- [holo_extended_descriptors_05](figures/holo_extended_descriptors_05.png)
- [holo_extended_descriptors_06](figures/holo_extended_descriptors_06.png)
- [holo_extended_descriptors_07](figures/holo_extended_descriptors_07.png)
- [holo_extended_descriptors_08](figures/holo_extended_descriptors_08.png)
- [holo_extended_descriptors_09](figures/holo_extended_descriptors_09.png)
- [holo_extended_descriptors_10](figures/holo_extended_descriptors_10.png)
- [holo_extended_descriptors_11](figures/holo_extended_descriptors_11.png)
- [holo_extended_descriptors_12](figures/holo_extended_descriptors_12.png)
- [holo_extended_descriptors_13](figures/holo_extended_descriptors_13.png)
- [holo_extended_fields](figures/holo_extended_fields.png)
- [holo_extended_shared_pca](figures/holo_extended_shared_pca.png)
- [ks_all_features_bar](figures/ks_all_features_bar.png)
- [ks_cdf_comparison](figures/ks_cdf_comparison.png)
- [ks_cdf_top_bottom](figures/ks_cdf_top_bottom.png)
- [lda_coefficients](figures/lda_coefficients.png)
- [lda_distribution](figures/lda_distribution.png)
- [optimal_k_sweep](figures/optimal_k_sweep.png)
- [physicochemical_per_cluster_boxplot](figures/physicochemical_per_cluster_boxplot.png)
- [roc_feature_importance](figures/roc_feature_importance.png)
- [umap_cluster_comparison](figures/umap_cluster_comparison.png)
- [umap_full_features](figures/umap_full_features.png)
- [umap_full_features_corrected](figures/umap_full_features_corrected.png)
