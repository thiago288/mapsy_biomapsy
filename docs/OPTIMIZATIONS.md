# Calculation and memory use

Descriptor calculations process query points in batches. NumPy or CuPy evaluates the query/atom distance arrays together, so the code avoids a Python loop over every pair.

Periodic distances use a fast wrapping calculation for orthogonal cells. Skew cells use a cached Minkowski-reduced lattice and evaluate nearby image shifts one at a time. This avoids storing an array with a separate axis for every candidate image.

Radial Chebyshev descriptors stream polynomial orders through a forward recurrence. They retain two polynomial states rather than the full order-by-query-by-atom tensor. The loop over polynomial orders is intentional; query/atom pairs remain vectorized.

BioMapSy can use a KD-tree on the CPU to select atoms within the descriptor cutoff. Species channels remain aligned with the complete structure. GPU batches use dense arrays.

Batching limits temporary array size but does not impose a hard memory ceiling. Large atom counts and angular neighbor-pair calculations can still require substantial memory. Measure your own workload with the [CPU/GPU example](../examples/cpu_gpu/README.md). Published measurements are listed in [the example results](../examples/measurements/README.md).
