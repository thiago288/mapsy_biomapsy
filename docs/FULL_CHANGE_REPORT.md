# Complete explanation of the fixed implementation

This report describes the earlier implementation audit. Detailed run artifacts are stored in the private development archive. Current package and example checks are listed in [VALIDATION_REPORT.md](VALIDATION_REPORT.md).

The purpose of this copy is to retain the successful vectorized approach while repairing specific geometry, state, API and numerical edge cases. The audit’s 40 findings were not 40 demonstrated failures in the active vectorized workflow. Some were inherited defects, already-corrected behavior, inactive experiments, scientific conventions or unavailable validation. The disposition of each finding is summarized below.

Final validation: 27 CPU tests and 26 actual-GPU tests pass. Ordinary outputs remain comparable to mapsy-optimized and the supplied backup where their mathematical definitions agree. Neither original directory was changed.

The historical backup is the comparison baseline; its identity as an official GitHub branch commit was not established. No claim here attributes every inherited defect to the official repository.

## 1. Periodic distances and geometric consistency

The previous descriptor kernel rounded each periodic fractional coordinate independently. That gives correct nearest images for orthogonal cells, including rotated orthogonal bases, but is not generally correct when the metric couples skewed directions. The fixed kernel retains that fast path and, for skew cells, caches a Minkowski-reduced basis then evaluates neighboring image candidates in vectorized point/atom batches. The basis change preserves the lattice; it does not strain the structure, force right angles or increase atom count.

The image search is a small fixed loop over shifts, not a Python loop over every point/atom. Candidate tensors are evaluated one at a time rather than allocated with a 27-image axis. That protects the original vectorization/memory intent. Singular partially periodic cells are completed in nonperiodic directions; invalid dependent periodic vectors are rejected.

UFF determines soft-sphere radii and is independent of the minimum-image problem. UFF remains the protein ionic-ass/default legacy radius choice. Requested Bondi/Pauling/MUFF modes are honored rather than silently replaced by UFF.

Grid basis scaling now follows ASE’s row-vector convention and divides each lattice row by its own grid count. Spherical grid fields use the corrected MIC and explicit periodic flags; contact-grid adjacency wraps only permitted axes.

Evidence: [batch_distances_periodic](../src/mapsy/utils/gpu_utils.py); [_prepare_mic](../src/mapsy/utils/gpu_utils.py); [Grid.__init__](../src/mapsy/data/grid.py); [Grid._apply_minimum_image_convension](../src/mapsy/data/grid.py); `test_mic_all_pbc_and_skews`, GPU MIC evidence.

## 2. Boundary state and radius configuration

IonicBoundary rebuilds start from a zero gradient. Previously, accumulating into the old field and then negating could alternate between the correct gradient and zero on unchanged-system updates; this behavior was shared with the backup. Sphere positions/settings are rebuilt for moved atoms, and per-sphere cached fields are released after each pass instead of all being retained.

Repeated updates now reproduce the same switch/gradient. Non-UFF radius requests intentionally produce different boundaries. Scalar distances at exact periodic image ties agree, but vector branch direction can differ; tested gradient differences are <=5.97e-8 and disappear away from ties. A scalar MIC field has a nondifferentiable branch at such bisectors.

Evidence: [IonicBoundary._build](../src/mapsy/boundaries/ionic.py); [IonicBoundary._set_soft_spheres](../src/mapsy/boundaries/ionic.py); `test_boundary_idempotent_and_radius_models`, `boundary_tie_results.json` (private development archive).

## 3. BioMapSy remains aperiodic

The active module is biomapsy.py. No periodic distance wrapping was introduced into BioMapSy’s descriptor or ionic-ass/tan calculations. The retained ionic branch stays available as the historical comparison mode; retaining it was intentional and is not classified as a defect.

Tan mode now reads lab-frame grid coordinates directly, applying the bounding-box origin once. The old second addition changed where the protein was sampled and broke translation invariance. Ionic-ass also constructs an aperiodic bounding box even when the input structure stores a cell, so a stored cell cannot force its surface back to an inappropriate origin.

The ionic-ass switch is s=1-product(1-h_a). Its gradient is the positive sum of grad(h_a) times the product of all other complements. The old implementation had the opposite sign. Correcting this changes vector orientation, while a pure sign correction leaves its magnitude-based surface selection unchanged.

Standalone AperiodicERFC kind 3 is volume*erfc/2 and kind 4 is its complement. Density/gradient now agree with those definitions; reset recomputes rather than leaving a zero gradient. BioSystem owns an Atoms copy and refreshes its KD-tree after position changes, avoiding caller PBC mutation and stale contact geometry.

Evidence: [_create_tan_contact_space](../src/mapsy/biomapsy.py); [_create_aperiodic_ionic_boundary](../src/mapsy/biomapsy.py); [AperiodicERFC._compute](../src/mapsy/biomapsy.py); [BioSystem._ensure_kdtree](../src/mapsy/biomapsy.py); translation and finite-difference regression tests.

## 4. Descriptor definitions, stability and memory

Radial ACSF still evaluates the same Chebyshev polynomials for ordinary inputs. It now streams polynomial orders with two recurrence states, retaining vectorization across all point/atom pairs instead of stacking every order and contribution simultaneously. This reduces temporary memory on CPU and GPU. Requested order sequence, repeated orders and output feature ordering are preserved.

At atom centers the radial ACSF mathematical weight is singular. The fixed implementation raises clearly at distance <=1e-10 Angstrom rather than return infinity or an arbitrary huge finite regularization. Angular AC/BP exclude coincident centers, where the angle is undefined. These are explicit scientific edge policies, not a claim that surface points normally coincide with atoms.

The optimized angular support threshold fc>1e-8 and radial near-cutoff support were retained. They still differ from the backup’s tolerance rules. Distinct-neighbor BP j<k pairs and correct species-pair channel counts were already improvements in mapsy-optimized; the fixed copy retains them. Consequently, backup angular outputs cannot be declared equivalent to optimized/fixed outputs.

BPSF parameter defaults now match the input model; scalar parameter inputs become valid one-dimensional arrays, and invalid/mismatched parameters are rejected. Optimized feature naming is preserved; rounded-name collisions raise explicitly, with a tested-config key map for older naming. Existing trained models are not automatically migrated.

Dense batches use a memory-sizing heuristic while retaining batch_size as an upper bound. CPU BioMapSy can prune each batch to the union of KD-tree neighbors and slice global species IDs/weights. It does not rebuild the species vocabulary for each local environment, which would change compositional channels.

GPU radial biological distances now sum squared Cartesian differences componentwise. This retains an (N,M) temporary rather than the full vector tensor, while avoiding subtraction of huge squared absolute coordinates.

Evidence: [ACSymmetryFunction._calculate_radial_batch_gpu](../src/mapsy/symfunc/atomic/acsf.py); [BPSymmetryFunction.__init__](../src/mapsy/symfunc/atomic/bpsf.py); [BioMaps._process_vectorized](../src/mapsy/biomapsy.py); [BioMaps._process_gpu](../src/mapsy/biomapsy.py); CPU/GPU comparisons and allocation reports.

## 5. Surface connectivity, graphs and representative sites

Generated biological surfaces now use six-face spatial grid adjacency without periodic wrapping, rather than unrestricted kNN links that could bridge remote patches. Connected components are treated as undirected. This intentionally changes topology when the old neighbor heuristic joined disconnected surfaces.

Keeping the main region remaps retained node indices and removes references to discarded points. Cluster graphs count each physical undirected edge once, so reciprocal or asymmetric neighbor lists do not erase real edges through integer division.

When clustering is scaled, representative-site distance calculations use the same scaling. Public raw cluster centers remain available. Dense RBF spectral clustering itself was not replaced, and species-set-relative compositional weights were not redefined; those remain existing methodological choices.

Evidence: [_compute_neighbors_and_regions](../src/mapsy/biomapsy.py); [BioContactSpace.keep_main_region](../src/mapsy/biomapsy.py); [cluster_graph](../src/mapsy/utils/graph.py); [Maps.sites](../src/mapsy/maps.py); graph/remapping and analysis tests.

## 6. I/O, empty inputs, compatibility and diagnostics

XYZ cubic-cell and header-origin Bohr units are converted consistently. Cube basis vectors and origin are converted, the parsed origin is retained, and the row-cell convention is consistent. Cube remains aperiodic by default because cube input does not encode periodicity; periodicity is not invented from the presence of a cell.

MapsFromFile parses one system once, and SystemParser property-file state is per-instance. Empty query results retain the feature schema; empty contact spaces have zero regions. Chunk splitting handles empty inputs, validates numeric data and accepts the old n_jobs keyword as an alias. Batch size and query shape/finiteness are checked.

Pydantic compatibility uses its supported v1 namespace directly, without patching sys.modules in production. The comparison harness still aliases unchanged older packages only. Metadata declares existing runtime dependencies, Python>=3.10, a CLI entry point and fixed-copy version 0.0.1+fixed. No dependencies were installed. Lazy public exports allow Maps/BioMaps imports without forcing heavy CLI startup.

Plotting checks missing data before filtering; one-panel scatter and feature-only PCA plotting work. Cube symmetry functions, undefined tofile serialization and solvent-aware requests now reject unsupported work explicitly rather than silently appear successful. Existing ERFC derivative diagnostics no longer compare an unpopulated derivative to a density-volume integral, and the density warning handles both error signs.

Historical modules remain archived; only their import provider was repaired where necessary. They are not certified as active replacements. Old optimization documents are marked historical, and the copied smoke entrypoint now executes assertion-based tests.

Evidence: I/O/native tests, [SystemParser.__init__](../src/mapsy/io/parser.py); [MapsFromFile.__init__](../src/mapsy/maps.py); `pyproject.toml`; `test_optimized_mapsy.py`; source patch.

## 7. GPU validation and benchmark reliability

Actual GPU testing passes 26 tests. It exposed and repaired explicit device-to-host conversion in three descriptor failure paths; original failing traces are retained. Normal mixed workflows use GPU distances and radial descriptor arithmetic, with angular descriptors on CPU. Direct BP angular tensor reduction was separately verified on GPU. PCA/clustering remain CPU algorithms.

Benchmark timings include synchronization, warmups, output equivalence and executed-backend checks. Actual sample size is distinguished from requested size. Crossover direction and multiple crossings are recorded; no increasing-size GPU recommendation is invented from a decreasing-speedup crossover. All-NaN GPU rows are handled.

The measured results support GPU use for substantial radial workloads on this hardware. They do not support a universal benefit for mixed angular or sparse tree-pruned workloads. Read GPU_VALIDATION_REPORT.md for complete tables, memory definitions, the small unchanged-baseline tolerance miss and timing limitations.

## 8. What remains a limitation rather than a fixed error

- Maps retains one minimum image per original atom. A warning exposes cutoffs beyond half the shortest lattice vector; image-complete primitive/supercell-invariant environments were not implemented.
- Spherical periodic fields and descriptors support skew cells; skew line/plane boundary fields explicitly reject unsupported projected-lattice geometry.
- Dense spectral clustering still has quadratic affinity-memory scaling.
- Compositional coordinates still depend on the species vocabulary; changing that would require a defined new descriptor/model convention.
- No production protein/crystal dataset, alternate hardware or cross-version dependency matrix was supplied/tested. Resource-exhaustion recovery was tested by controlled exception injection, not intentional real VRAM exhaustion.
- Build/install was not exercised and the environment still selects the original package unless PYTHONPATH points at this copy.

## 9. Scientific and publication interpretation

Vectorization, order streaming, memory-bounded image search, correctly aligned neighbor pruning and reliable device fallback are software-engineering/algorithm-implementation contributions. Minkowski reduction and the symmetry-function definitions are established methods; their use here is not evidence of a new physical descriptor.

Aperiodic surface mapping for biological systems may provide a valuable application/workflow contribution, but methodological novelty requires a literature comparison and demonstrated scientific outcomes. The changes improve correctness and scalability without proving novelty or invalidating ordinary results from the prior working vectorized version.

## Complete finding disposition index

| Finding | Disposition |
|---|---|
| F01: Fractional wrapping is not general triclinic MIC | FIXED |
| F02: Rank-deficient partially periodic cell fails | FIXED |
| F03: Requested radius model silently ignored | FIXED |
| F04: Boundary updates are not idempotent in either snapshot | FIXED |
| F05: Backup active ERFC cannot use its active Grid | ALREADY CORRECTED |
| F06: Angular BPSF removes self-pairs | ALREADY CORRECTED |
| F07: BPSF element-pair channel allocation corrected | ALREADY CORRECTED |
| F08: Radial ACSF tolerance and endpoint semantics changed | FIXED WITH EXPLICIT POLICY |
| F09: Angular support and floating-point handling change | FIXED / CONVENTION RETAINED |
| F10: Angular ACSF returns NaNs at query/atom coincidence | FIXED |
| F11: Descriptor column names are not fully compatible | GUARDED / MIGRATION DOCUMENTED |
| F12: Active protein tan grid applies origin twice | FIXED |
| F13: Retaining main region leaves stale graph indices | FIXED |
| F14: Empty-grid/query edge cases fail | FIXED |
| F15: Parallel helper compatibility/protection lost | FIXED |
| F16: Retained multiprocessing alternative has unresolved import | FIXED HISTORICAL IMPORT |
| F17: Non-atomic descriptor handling is silent zeros | GUARDED |
| F18: Active BioMaps descriptor KD-tree option is unused | FIXED FOR BUILT-IN CPU DESCRIPTORS |
| F19: kNN regions can bridge spatially disconnected surfaces | FIXED |
| F20: BioMaps.graph drops valid edges | FIXED |
| F21: ERFC integration diagnostic has wrong parentheses/state | FIXED |
| F22: MapsFromFile parses system twice and shared parser leaks state | FIXED |
| F23: GPU path coverage and reachability limitations | GPU VERIFIED FOR TESTED PATHS |
| F24: Vectorization increases temporary and cached-field memory | FIXED FOR MEASURED MEMORY BOTTLENECKS |
| F25: Useful backup-only experimental algorithms not promoted | RETAINED AS DEVELOPMENT OPTIONS |
| F26: Shared periodic grid topology ignores system PBC | FIXED FOR POINT FIELDS; EXTENDED SKEW GUARDED |
| F27: AperiodicERFC kind-4 normalization contradicts complement | FIXED |
| F28: Aperiodic ionic vector gradient has opposite sign | FIXED |
| F29: Benchmark recommendation reverses a decreasing-speedup crossover | FIXED |
| F30: GPU benchmark provenance can be misleading | GPU VERIFIED FOR TESTED BENCHMARKS |
| F31: Bundled smoke test is not a rigorous regression suite | FIXED |
| F32: Optimization documentation is stale/unverified | FIXED DOCUMENTATION |
| F33: Native Pydantic 2 compatibility fails in both versions | FIXED |
| F34: Packaging omits runtime dependencies and misstates API support | FIXED METADATA; INSTALL UNTESTED |
| F35: Shared XYZ/cube units and origin defects | FIXED UNITS / PBC POLICY RETAINED |
| F36: Shared plotting and placeholder gaps | FIXED DEFECTS / UNSUPPORTED FEATURES GUARDED |
| F37: Shared scaling and data-model caveats | METRIC FIXED; METHOD CONVENTIONS RETAINED |
| F38: Active BioSystem can retain stale neighbors and mutate caller | FIXED |
| F39: Aperiodic radial squared-distance identity risks cancellation | GPU VERIFIED |
| F40: Finite periodic neighborhoods are not image-complete | DOCUMENTED AND WARNED; METHOD RETAINED |

New repairs beyond the original finding IDs: N01 BPSF defaults/parameter shape; N02 grid row scaling for unequal dimensions; N03 descriptor GPU fallback host conversion and direct angular input handling. Their implementations are described above; detailed development records remain in the private archive.

Evidence files: `changes.patch` (private development archive), `fix_checklist.json` (private development archive), `all_snapshot_comparisons.json` (private development archive), `gpu_test_results.json` (private development archive), `gpu_benchmark_results.json` (private development archive), `gpu_supplemental_benchmark_results.json` (private development archive), `source_integrity.json` (private development archive).
