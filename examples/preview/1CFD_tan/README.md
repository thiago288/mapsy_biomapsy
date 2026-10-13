# 1CFD — apo (calcium-free source) — tan

53,986 points; 26 features; kmeans in standardized descriptor space.

## Descriptor maps

![Several descriptor channels](descriptors.png)

Every coefficient: [descriptor_panel_01](descriptor_panel_01.png), [descriptor_panel_02](descriptor_panel_02.png), [descriptor_panel_03](descriptor_panel_03.png)

![Physicochemical fields](physicochemical.png)

## PCA

![Explained variance](pca_variance.png)

![Loadings](pca_loadings.png)

![Physical PCA maps](pca_physical.png)

![PCA pairs](pca_scatter.png)

## Cluster-count curves

![Silhouette and Davies–Bouldin](k_selection.png)

Median-over-two-seed choices: silhouette k=2, DB k=4. Values and all trials: [measured tables](../../measurements/1CFD_tan/clustering_metrics.csv).

## Silhouette-selected partition

![Silhouette choice](best_silhouette.png)

## Davies–Bouldin-selected partition

![DB choice](best_davies_bouldin.png)

