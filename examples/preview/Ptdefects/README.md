# Pt(111) with vacancies and O

81,000 points; 40 features; kmeans in standardized descriptor space.

## Descriptor maps

![Several descriptor channels](descriptors.png)

Every coefficient: [descriptor_panel_01](descriptor_panel_01.png), [descriptor_panel_02](descriptor_panel_02.png), [descriptor_panel_03](descriptor_panel_03.png), [descriptor_panel_04](descriptor_panel_04.png)

## PCA

![Explained variance](pca_variance.png)

![Loadings](pca_loadings.png)

![Physical PCA maps](pca_physical.png)

![PCA pairs](pca_scatter.png)

## Cluster-count curves

![Silhouette and Davies–Bouldin](k_selection.png)

Median-over-two-seed choices: silhouette k=2, DB k=2. Values and all trials: [measured tables](../../measurements/Ptdefects/clustering_metrics.csv).

## Partition selected by both metrics

![Both metric choices](best_silhouette.png)


## Higher-resolution descriptor comparison

The k=2 metric optimum merges substantial descriptor variation. A separate fidelity and seed-agreement study supports showing **k=32** alongside k=2 and k=10. It does not resolve every known site family. Read the [short explanation](CLUSTER_RESOLUTION.md) for the mathematical criterion, held-out checks and limitations.

![Full-plane comparison](resolution_comparison.png)

![Vacancy and oxygen close-ups](defect_zoom.png)

![Compactness, fidelity and seed agreement](resolution_curves.png)

![Within-region descriptor fidelity](regional_fidelity.png)
