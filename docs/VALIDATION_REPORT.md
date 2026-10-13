# Validation and performance

## Release 0.1.0 installation checks

On 12 October 2026, the package wheel was built and installed in a new Python 3.12 environment with freshly resolved dependencies. No packages from the original environment were reused for this CPU check. All 40 CPU package tests and all 9 example tests passed. The installed module came from the wheel, the CLI reported 0.1.0, and `pip check` reported no broken requirements.

Resolved versions included NumPy 2.5.4, SciPy 1.18.1, pandas 3.0.6, ASE 3.29.0, scikit-learn 1.9.1, Pydantic 2.14.1 and Matplotlib 3.11.2. These verify installation compatibility; the published example measurements retain their original recorded software versions.

The installed wheel also passed the periodic-surface GPU regression on an NVIDIA RTX 4070 Laptop GPU with CuPy 14.0.1, using the existing CUDA libraries. An initial isolated run lacked the NVRTC library search path; explicitly exposing the existing runtime libraries to that process resolved it. No MapSy algorithm or existing environment was changed. CPU/GPU descriptor tolerance remains rtol=1e-10, atol=1e-12.

The [CPU workflow](../.github/workflows/cpu-tests.yml) is configured for Python 3.10, 3.11 and 3.12 on pushes and pull requests. Its syntax and local commands were checked, but the workflow has not yet run on GitHub. Only Python 3.12 was exercised locally during this installation check. Detailed build, installation and GPU logs remain in the private release archive.


The public package suite contains 41 tests. On 12 October 2026, 40 passed on the CPU and the GPU test was skipped in that run. A separate run on an NVIDIA RTX 4070 Laptop GPU with CuPy 14.0.1 passed the remaining test on Pt(111), Pt(100), Pt(110) and decorated Pt(111), without CPU fallback. The five example tests also passed. CPU/GPU descriptor comparisons use rtol=1e-10 and atol=1e-12.

Run `python -m unittest discover -s tests -v` and `python -m unittest discover -s examples/tests -v` from the repository root. The reports below summarize earlier, broader comparisons; their raw artifacts are preserved outside the public repository.

GPU follow-up: 26 actual-device tests pass. See [GPU_VALIDATION_REPORT.md](GPU_VALIDATION_REPORT.md) and [FULL_CHANGE_REPORT.md](FULL_CHANGE_REPORT.md).

27 assertion-based native tests pass using `python`. The fixed package imports Pydantic v1 compatibility directly; only unchanged baseline packages use the comparison-harness aliases. No dependencies were installed.

The comparison reference is the supplied historical backup, not a verified official GitHub commit. Official GitHub provenance remains unavailable. Original source integrity checks pass for both directories; see `source_integrity.json` (private development archive).

## Numerical results

Ordinary ACSF radial/angular and BPSF G1–G3 fixtures agree with both references at rtol=1e-10, atol=1e-12. BPSF G4/G5 agree with optimized for ordinary inputs; the backup has incorrect self-pairs/channel allocation. Fixed-versus-serial tests cover all five BP orders and both AC families. Fixed-input PCA and clustering remain comparable, including label-permutation-invariant ARI.

Random minimum-image tests cover 12 full-rank cells and all eight PBC combinations, plus singular partial-periodic cells. Saved skew fixture maximum distance disagreement with ASE: 1.7763568394002505e-15 Angstrom.

Bio tan and ionic-ass pass translation invariance, including structures with stored cells. Ionic-ass gradients pass central finite differences (step 1e-5 Angstrom, rtol=2e-7, atol=1e-9). KD-tree and dense built-in descriptors agree, including global species vocabulary alignment and empty local neighborhoods.

## Intentional output changes

- Skew MIC and corrected grid geometry can change periodic features, boundaries and cutoff membership.
- Tan double-origin and ionic-ass frame corrections change wrongly translated surface samples.
- Correcting ionic-ass gradient sign changes vector orientation; its magnitude alone is unchanged.
- Spatial grid adjacency changes regions/graphs previously joined by unrestricted kNN links.
- Non-UFF requests now use their requested tables; default UFF stays UFF.
- Repeated boundary updates now reproduce their first correct field.
- At exact periodic image ties, boundary vector direction can differ: tested gradient discrepancies are <=5.97e-8, with agreement away from ties. Such bisectors have an ambiguous MIC branch and a nondifferentiable scalar boundary; see audit/boundary_tie_results.json.
- Radial ACSF at distance <=1e-10 Angstrom raises instead of returning Inf or an arbitrary huge finite value; angular coincident centers are excluded.
- Cube/XYZ unit corrections change previously inconsistent coordinates.
- Scaled-clustering representative sites now use the same metric as clustering.

## Measured CPU performance

One warmup, five repetitions, medians, one numerical thread. Synthetic fixtures; imports/setup excluded. Memory is tracemalloc allocated peak, not process RSS or GPU memory.

| Fixture | Optimized time | Fixed time | Optimized peak | Fixed peak | Optimized/fixed time |
|---|---:|---:|---:|---:|---:|
| radial_scale_256_128, 8 orders, two AC channels | 1.087 ms | 0.826 ms | 10.05 MB | 2.49 MB | 1.32x |
| radial_scale_1024_512, 8 orders, two AC channels | 35.623 ms | 23.687 ms | 159.52 MB | 38.47 MB | 1.50x |

The radial scale inputs are uniform synthetic distances in [0.5,4.5] Angstrom (seed 51), not a consistent full atom/query geometry. Outputs match both baselines; polynomial-order streaming preserves vectorization over pairs while removing the order-sized temporary stack.

| Correct distance fixture, 128 points × 256 atoms | ASE time | Fixed time | ASE peak | Fixed peak |
|---|---:|---:|---:|---:|
| distance_orthogonal | 21.832 ms | 0.712 ms | 62.13 MB | 2.62 MB |
| distance_skew | 18.408 ms | 16.835 ms | 62.13 MB | 4.75 MB |

Sparse Bio radial fixture: 10,000 CHON atoms uniform in a 200 Angstrom cube, 64 queries, batch_size=8; KD-tree 1.769 ms versus dense 26.740 ms (15.12x). This sparse geometry is favorable to neighborhood pruning and is not a real-protein benchmark.

Correct skew handling costs more than the old incorrect single-image selection. The fixed path is not promised to retain the old skew-case runtime. Small-query radial kernels can also be slower from order streaming; see raw per-case timing samples rather than assuming universal speedup.

## Remaining limits

- Initial CPU-only run did not exercise GPU. Subsequent actual-device validation is recorded in GPU_VALIDATION_REPORT.md and audit/gpu_test_results.json; its scope and limits supersede the earlier unavailable-GPU assessment.
- One-MIC-image-per-atom descriptors are retained. A warning exposes cutoffs beyond half the shortest lattice vector; image-complete periodic environments are not implemented.
- Skew-cell line/plane boundary fields are explicitly unsupported. Spherical point fields and periodic descriptors support skew cells.
- Dense RBF spectral clustering and species-set-relative composition remain existing methods. Large clustering workloads still have quadratic affinity memory.
- No production protein/crystal dataset was supplied. Synthetic coverage cannot establish correctness of every future input or compatibility of previously trained models.
- Build/install and other Python/dependency versions were not tested. Original environment imports still resolve to the original project unless PYTHONPATH selects this copy.
- Historical original/v3/v4 modules were retained as reference material. They are not the active Bio implementation and their algorithms are not certified.

## Reproduce

```bash
PYTHONDONTWRITEBYTECODE=1 python /home/thiago/mapsy_sandbox/mapsy_opt_fixed/audit/test_fixed.py
PYTHONDONTWRITEBYTECODE=1 python /home/thiago/mapsy_sandbox/mapsy_opt_fixed/audit/validate_snapshots.py
PYTHONDONTWRITEBYTECODE=1 python /home/thiago/mapsy_sandbox/mapsy_opt_fixed/audit/additional_validation.py
PYTHONDONTWRITEBYTECODE=1 python /home/thiago/mapsy_sandbox/mapsy_opt_fixed/audit/benchmark_fixed.py
```
