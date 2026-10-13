# Actual GPU validation

This report describes the earlier implementation audit. Detailed run artifacts are stored in the private development archive. Current package and example checks are listed in [VALIDATION_REPORT.md](VALIDATION_REPORT.md).

Updated 2026-10-09T15:51:01.818958+00:00. This report supersedes the earlier unavailable-GPU assessment. CuPy was already present when this task started; no dependencies were installed during validation.

**26 GPU tests pass, with 192 recorded comparisons and checks. The CPU suite also passes 27 tests after the GPU fallback repairs.**

## Hardware and execution

- Device: NVIDIA GeForce RTX 4070 Laptop GPU, approximately 8 GB VRAM.
- CuPy 14.0.1; CUDA runtime 12.9 (`12090`), driver API 13.0 (`13000`). The CUDA version shown by nvidia-smi describes driver support, not a separate installation performed here.
- Python 3.12.3 in `python`.
- Scientific arrays and kernels use float64.
- NumPy/BLAS thread settings: one numerical thread.
- Real device arrays and synchronized CUDA operations were used. Successful CPU fallback was not accepted as successful GPU execution.

## Correctness coverage

- Six full-rank cells, including severe skew, with all eight PBC combinations; singular partially periodic cells and aperiodic zero cells.
- CPU/GPU/ASE distance and vector agreement, rotations, permutations and translated aperiodic coordinates.
- Radial ACSF: both cutoff types, structural/compositional combinations, high/noncontiguous/repeated orders and cutoff endpoints.
- Radial BPSF G1–G3 and direct angular BP G4/G5 device reduction.
- Complete periodic orthogonal, skewed and partial-PBC workflows; mixed radial/angular descriptors.
- BioMapSy radial/mixed workflows and tan/ionic-ass contact-to-descriptor pipelines.
- Translation to coordinates of order 1e8 Angstrom to test cancellation resistance. This is an edge-case test, not a typical protein coordinate range.
- PCA and clustering on CPU/GPU-generated features: PCA maximum discrepancy 4.885e-15; adjusted Rand index 1.0.
- Repeated calls and live-pool allocation stability.
- Controlled Maps/BioMaps CUDA failure recovery; controlled descriptor-kernel failures; public benchmark rejection of CPU fallback and correct recording of effective sample counts.

The largest absolute numerical discrepancy recorded by the GPU regression suite was 5.151e-14. MIC distances/vectors against ASE had maximum discrepancy 2.842e-14 Angstrom. General comparison tolerance: rtol=1e-10, atol=1e-12. Each comparison is saved in `gpu_test_results.json` (private development archive).

These tests do not establish scientific equivalence for intentionally changed methods, such as old angular self-pairs, corrected surface origins or different graph topology.

## Additional bugs exposed by GPU testing

The initial 20 successful-operation tests passed. Three controlled-failure tests then demonstrated a fallback conversion bug: radial ACSF and BPSF could convert a NumPy input to CuPy, encounter a kernel error, then try `np.asarray` on the device array because the fallback checked the original input type. Direct BP angular fallback had a related conversion problem.

The fixed copy now checks the array’s current type and explicitly transfers device data to host before CPU recovery. Direct angular ACSF inputs are also transferred explicitly for its intentional CPU calculation. Evidence: `gpu_test_results_before_fallback_fix.json` (private development archive) retains the three real failure traces; the final suite proves recovery.

Implementation: [ACSymmetryFunction.values_batch](../src/mapsy/symfunc/atomic/acsf.py); [BPSymmetryFunction.values_batch](../src/mapsy/symfunc/atomic/bpsf.py). No original source was changed.

Controlled exceptions are not actual VRAM exhaustion. We did not intentionally exhaust the laptop GPU or certify every possible driver/OOM failure mode.

## What actually runs on GPU

| Operation | Normal Maps/BioMaps GPU path |
|---|---|
| Distances and periodic image candidates | GPU array operations; lattice reduction/cache preparation on CPU |
| Radial ACSF and BP G1–G3 | Device-native descriptor arithmetic; final features return to host |
| Angular ACSF and BP G4/G5 | CPU descriptor evaluation in the normal mixed workflow, after GPU distances |
| Direct BP G4/G5 GPU helper | Tested separately; CPU pair discovery plus device padded tensor reduction |
| Contact-space construction, PCA, clustering, graph/sites, plotting | CPU |

Calling `use_gpu=True` therefore does not mean the entire scientific pipeline executes on the device. Angular workloads retain their current CPU bottleneck.

## Synchronized wall-clock benchmarks

Two warmups and five timed repetitions; medians below. Workflow timings include transfers, descriptor calculation, results returned to host and DataFrame creation, but exclude fixture/model construction. CPU and GPU receive identical inputs. No fallback warning is allowed. These are synthetic CHON fixtures, not production proteins or crystals.

| Fixed workflow | Atoms × points | Batch size | Dense CPU | GPU | Dense CPU / GPU |
|---|---:|---:|---:|---:|---:|
| periodic_radial_small | 512 × 256 | 256 | 10.65 ms | 6.43 ms | 1.66x |
| periodic_radial_medium | 2048 × 1024 | 256 | 136.88 ms | 49.73 ms | 2.75x |
| periodic_radial_large | 8192 × 2048 | 64 | 1027.34 ms | 372.16 ms | 2.76x |
| bio_radial_medium | 2048 × 1024 | 256 | 128.47 ms | 6.94 ms | 18.50x |
| bio_radial_large | 8192 × 2048 | 64 | 889.66 ms | 44.17 ms | 20.14x |
| periodic_mixed | 512 × 128 | 256 | 53.28 ms | 53.04 ms | 1.00x |
| bio_mixed | 512 × 128 | 256 | 38.06 ms | 37.16 ms | 1.02x |
| periodic_radial_skew_medium | 2048 × 1024 | 256 | 1302.89 ms | 695.86 ms | 1.87x |

The main radial fixtures use eight orders and both structural/compositional channels (16 features). Mixed fixtures add four angular AC orders and angular G5 parameters. The main periodic table uses orthogonal cells; the separate skew fixture uses the exact reduced-basis image search.

Mixed workloads at 1.00–1.02x do not demonstrate a material GPU benefit. No statistical confidence interval is claimed from five repetitions.

## Compare GPU against CPU KD-tree, not only dense CPU

On the same 8,192-atom/2,048-query biological fixture, the default-capable CPU KD-tree path took 172.53 ms versus GPU 56.38 ms: 3.06x, substantially less than the approximately 20x dense-CPU comparison.

For a very sparse 10,000-atom/64-query fixture: dense CPU 26.11 ms, KD-tree CPU 1.80 ms, GPU 8.16 ms. CPU KD-tree is 4.55x faster than GPU. The distribution is uniform in a 200 Angstrom cube and is especially favorable to pruning.

GPU selection should consider descriptor family, neighborhood sparsity and batch size, not only atom count × query count. The current automatic heuristic is not universally optimal.

The isolated orthogonal distance kernel was slower on GPU for 256×512 resident inputs (4.24 ms versus CPU 2.87 ms); the full radial workflow can still benefit because it also accelerates descriptor arithmetic. The isolated corrected skew kernel benefited (42.29 ms GPU versus 69.29 ms CPU).

## Fixed versus unchanged optimized GPU performance and memory

| Validated ordinary fixture | Optimized GPU | Fixed GPU | Optimized tracked device peak | Fixed tracked device peak |
|---|---:|---:|---:|---:|
| periodic_radial_small | 8.75 ms | 6.43 ms | 44.10 MB | 14.87 MB |
| periodic_radial_medium | 56.23 ms | 49.73 ms | 176.27 MB | 59.35 MB |
| periodic_radial_large | 411.12 ms | 372.16 ms | 176.43 MB | 59.52 MB |
| bio_radial_medium | 13.86 ms | 6.94 ms | 167.90 MB | 50.96 MB |
| periodic_mixed | 53.57 ms | 53.04 ms | 22.06 MB | 7.44 MB |
| bio_mixed | 37.60 ms | 37.16 ms | 22.06 MB | 7.44 MB |

The large periodic radial case reduced tracked device allocations from 176.43 MB to 59.52 MB (about 66% less) while improving GPU time from 411.12 ms to 372.16 ms. The memory improvement is stronger than the GPU speed improvement on this fixture.

Memory is a separate untimed CuPy default-memory-pool allocation high-water measurement, with starting live allocations subtracted. It excludes CUDA context, driver allocations, library workspaces outside that pool and total process/device RSS. Host tracemalloc peaks and every timing sample are retained in JSON.

The query-memory limit is a sizing heuristic, not a hard 64 MiB VRAM ceiling. The skew end-to-end benchmark had a tracked device peak near 92.86 MB because extra candidate/reduction arrays coexist.

## The optimized baseline strict-tolerance miss

On the unchanged optimized BioMapSy large fixture, 13 of 32768 descriptor values missed rtol=1e-10, atol=1e-12. Maximum absolute difference: 6.897e-11. All values pass rtol=1e-8, atol=1e-10.

This is a small floating-point difference at ordinary coordinates, not evidence of physically meaningful incorrect protein results. The squared-absolute-coordinate distance identity differed from stable displacement distances by up to 8.583e-12 Angstrom on the affected query rows. The fixed componentwise kernel passed the original strict gate.

No GPU timing or speedup was accepted for that failed strict baseline case. Details, including the worst tolerance-normalized feature, are in `baseline_gpu_difference_results.json` (private development archive).

At artificially translated coordinates near 1e8 Angstrom, the old squared-distance identity demonstrably returned zero for distances of 0.572822 and 0.910014 Angstrom. The fixed kernel avoids that cancellation. This edge case motivates numerical stability without asserting that ordinary working datasets experienced it.

## Scope and remaining limits

- Results apply to this RTX 4070 Laptop GPU, CuPy 14.0.1, CUDA 12.9 runtime and float64 fixtures.
- No production dataset was supplied; no universal speedup, all-input correctness or GPU-clustering implementation is claimed.
- One-image periodic descriptor convention and unsupported skew line/plane fields remain as documented in VALIDATION_REPORT.md.
- Actual resource-exhaustion stress, other GPU architectures, float32/mixed precision and alternative CUDA/CuPy versions were not tested.
- Original package/backup integrity is verified separately; no dependencies were installed and no environment was redirected.

## Reproduce

```bash
PYTHONDONTWRITEBYTECODE=1 python /home/thiago/mapsy_sandbox/mapsy_opt_fixed/audit/test_gpu.py
PYTHONDONTWRITEBYTECODE=1 python /home/thiago/mapsy_sandbox/mapsy_opt_fixed/audit/benchmark_gpu.py
PYTHONDONTWRITEBYTECODE=1 python /home/thiago/mapsy_sandbox/mapsy_opt_fixed/audit/benchmark_gpu_supplemental.py
PYTHONDONTWRITEBYTECODE=1 python /home/thiago/mapsy_sandbox/mapsy_opt_fixed/audit/baseline_gpu_difference.py
```
