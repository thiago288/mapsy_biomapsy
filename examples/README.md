# MapSy & BioMapSy: Periodic and Aperiodic Surfaces

MapSy generates local-environment descriptor maps and supports exploratory dimensionality reduction and clustering. Cluster counts and assignments require validation for the intended scientific application.

These examples live at the repository root, alongside `src/`. Use the installed fixed MapSy implementation and the same environment as your notebook kernel. No examples download data or install dependencies automatically.

## Tutorials

| Example | Content |
| --- | --- |
| [Classical periodic surfaces](periodic_surfaces/00_classical_surfaces.ipynb) | Original Pt111/100/110 model inputs; radial/angular descriptors, PCA, both metrics and original comparison resolutions |
| [Surface-referenced ideal surfaces](periodic_surfaces/01_ideal_surfaces.ipynb) | The three ideal faces with an explicit surface clearance |
| [Vacancies and oxygen](periodic_surfaces/02_decorated_surface.ipynb) | Full decorated Pt111 grid; structural/compositional radial/angular descriptors |
| [Calmodulin](biomapsy/01_calmodulin.ipynb) | Apo1CFD and holo-derived1CLL; aperiodic tan and ionic-ass; physicochemical fields, shared PCA and angular controls |
| [Extended calmodulin analysis](biomapsy/extended_analysis/02_extended_calmodulin.ipynb) | 54 channels, PCA/UMAP, both metrics, spatial partitions, connectivity, physicochemical comparisons and classification diagnostics |
| [CPU/GPU comparison](cpu_gpu/01_cpu_gpu.ipynb) | Identical-input synchronized timings and descriptor/PCA/cluster agreement |

Each scientific fixture shows multiple descriptor channels, PCA variance/loadings and spatial plots, silhouette and Davies–Bouldin curves, and labeled partitions. Where both metrics choose the same partition, the plot says so. Classical fixed resolutions reproduce the original example choices; they are not metric optima. [Preview gallery](preview/README.md), [measured results](measurements/README.md), [experimental resolution study](experimental/README.md).

## Run

```bash
python examples/periodic_surfaces/classical_surfaces.py --surface 111
python examples/periodic_surfaces/corrected_surfaces.py --surface defects
python examples/biomapsy/protein_workflow.py --protein 1CFD --mode tan
python examples/biomapsy/protein_workflow.py --protein 1CLL --mode ionic-ass
python examples/cpu_gpu/benchmark.py --points 2048 --repeats 3
python -m unittest discover -s examples/tests -v
```

Outputs go to fresh timestamped `examples/results/` directories, ignored by Git. `MAPSY_EXAMPLES_OUTPUT` redirects notebook results. Optional notebook tooling is listed in `requirements-notebooks.txt`. Notebooks ship clean; measured figures and tables ship separately.

## Interpretation

The classical contact model uses center-of-mass distance3.5Å; surface-referenced tutorials use clearance1.5Å. These can sample different heights. Ideal surfaces use raw RBF spectral clustering; decorated/protein examples use standardized K-means. PCA uses standardized descriptors. Fits and DB use all points; silhouette above6000 rows uses2000 fixed seeded points. Display sampling and color limits are recorded, separately from calculations.

The calmodulin fixtures retain protein heavy atoms only. 1CFD is apo;1CLL is holo-derived with calcium omitted. Physicochemical fields retain the original notebook lookup formulas and are approximate descriptor proxies. [Protein details](biomapsy/README.md), [data provenance](data/proteins/README.md), [validation](VALIDATION.md), [scientific scope](SCIENTIFIC_SCOPE.md).

## Protein results at a glance

The protein examples include both conformations and both contact models, with complete descriptor panels, physicochemical maps, PCA, and both clustering metrics. See the [protein gallery](biomapsy/README.md#protein-figure-gallery) and [verified extended analysis](biomapsy/extended_analysis/README.md).

![Apo calmodulin descriptor maps](preview/1CFD_tan/descriptors.png)

![Holo-derived calmodulin physicochemical maps](preview/1CLL_tan/physicochemical.png)
