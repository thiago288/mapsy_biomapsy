# Classical Pt(110)

5,400 points; 20 features; spectral in raw descriptor space.

## Descriptor maps

![Several descriptor channels](descriptors.png)

Every coefficient: [descriptor_panel_01](descriptor_panel_01.png), [descriptor_panel_02](descriptor_panel_02.png)

## PCA

![Explained variance](pca_variance.png)

![Loadings](pca_loadings.png)

![Physical PCA maps](pca_physical.png)

![PCA pairs](pca_scatter.png)

## Cluster-count curves

![Silhouette and Davies–Bouldin](k_selection.png)

Median-over-two-seed choices: silhouette k=3, DB k=4. Values and all trials: [measured tables](../../measurements/classical_Pt110/clustering_metrics.csv).

## Silhouette-selected partition

![Silhouette choice](best_silhouette.png)

## Davies–Bouldin-selected partition

![DB choice](best_davies_bouldin.png)

## Classical comparison resolution

Resolution used in the original planar_maps_ideal notebook; not an automatically selected optimum

![Comparison resolution](comparison.png)

