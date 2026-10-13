# CPU/GPU benchmark

Run `01_cpu_gpu.ipynb` or `python examples/cpu_gpu/benchmark.py --points 2048 --repeats 3`. Mixed-descriptor control: `--points 512 --descriptor-set mixed`. `--proteins-only` excludes periodic fixtures.

Actual contact spaces are generated once per case; a seeded uniform subset of those positions is shared by both backends. Timings include descriptors, transfers and DataFrames, exclude contact/model construction, and synchronize CUDA. One warm-up and three repetitions are defaults. Every measured duration is retained, together with medians and observed ranges. GPU fallback is labeled and excluded from speedup claims.

Numerical checks use float64, rtol=1e-10 and atol=1e-12. PCA compares both outputs with the same CPU-fitted scaler/basis. Independent K-means fits use the same seed; ARI handles arbitrary label permutations. A near-degenerate partition can change despite numerically close descriptors, so inspect the recorded ARI rather than treating it as guaranteed.

NPZ files preserve exact coordinates, selected contact indices and CPU/GPU arrays. `benchmark.json` records versions, GPU model, backend, thread environment, parameter metadata and timing scope. There is no theoretical break-even line, and contact construction is not mixed into the descriptor speedup plot.

The benchmark times AC descriptors only. The protein tutorial additionally computes notebook-derived physicochemical fields with batched CPU NumPy; their time is recorded separately and is not included in AC-only speedup claims.
