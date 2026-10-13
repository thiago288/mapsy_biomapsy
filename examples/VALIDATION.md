# Example validation

Five notebooks were executed from clean kernels using the existing mapsy_opt environment and fixed MapSy source. No dependencies were installed. Original user notebooks and production source were not modified. Notebooks ship with cleared outputs; the resulting selected figures/tables are included separately.

| notebook | status | seconds | errors |
| --- | --- | --- | --- |
| biomapsy/01_calmodulin.ipynb | passed | 168.78621864318848 | 0 |
| cpu_gpu/01_cpu_gpu.ipynb | passed | 14.687781810760498 | 0 |
| periodic_surfaces/00_classical_surfaces.ipynb | passed | 138.93697905540466 | 0 |
| periodic_surfaces/01_ideal_surfaces.ipynb | passed | 398.80403232574463 | 0 |
| periodic_surfaces/02_decorated_surface.ipynb | passed | 74.69394564628601 | 0 |

14 existing package regression tests and5 example tests passed. Portable experimental scripts were checked on recorded data; coverage formula checks and site counts are recorded in the integration audit. Descriptor regression records the small original-source Pt110 cutoff difference rather than claiming blanket equivalence. CPU/GPU timing/equivalence outcomes are in measurements/benchmark.json.

The validation demonstrates these workloads and numerical checks. It does not validate biological mechanisms, universal physical cluster counts, a new clustering metric, or complete force-field assignments.

## Extended notebook verification

The [extended calmodulin notebook](biomapsy/extended_analysis/README.md) passed all code cells from a fresh kernel. Its execution, full-matrix numerical comparisons and scientific checks are included with the example. Historical unexecuted notebook copies were retained under the local audit directory and removed from the public examples by moving them, without deleting files.
