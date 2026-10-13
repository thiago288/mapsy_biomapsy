# MapSy and BioMapSy

MapSy describes local atomic environments on periodic surfaces. BioMapSy extends the workflow to aperiodic surfaces, including proteins. Both produce descriptor maps that can be explored with PCA and clustering.

This repository brings together my optimized implementation of periodic MapSy and BioMapSy, which I developed for aperiodic surfaces. It builds on the original MapSy project developed within MATERIALab at Boise State University. [AUTHORS.md](AUTHORS.md) records the original contributors and my role in this version.

## My additions to MapSy

I developed the vectorized implementation to make large descriptor calculations practical. Query points are processed in batches with NumPy or, optionally, CuPy on a GPU. Radial Chebyshev calculations stream polynomial orders to reduce temporary memory. BioMapSy can use a KD-tree to select nearby atoms for CPU calculations.

BioMapSy extends the workflow to aperiodic surfaces, including proteins. Its `tan` and `ionic-ass` contact models use Cartesian distances without periodic wrapping. Periodic MapSy remains available for crystalline surfaces.

The descriptor families and the PCA/clustering workflow build on original MapSy. My contributions extend the implementation's computational capacity and range of applications. The worked examples cover ideal and decorated platinum surfaces and calmodulin, including the physicochemical fields used in my earlier protein notebooks.

Version 0.1.0 combines this development with numerical corrections, regression tests and verified examples. See [RELEASE_NOTES.md](RELEASE_NOTES.md) for the changes and validation included in this release.

## Getting started

Use Python 3.10 or later. From the repository root, install the package in your environment:

```bash
python -m pip install -e .
```

The [examples](examples/README.md) provide notebooks, runnable scripts, input structures and figures. Start with the [classical periodic surfaces](examples/periodic_surfaces/00_classical_surfaces.ipynb) or the [calmodulin tutorial](examples/biomapsy/01_calmodulin.ipynb). For notebook tooling, install `python -m pip install -e ".[notebooks]"`. The extended protein analysis has additional dependencies; use `python -m pip install -e ".[extended]"` when running it. The corresponding requirements files remain available under `examples/`.

```bash
python examples/periodic_surfaces/classical_surfaces.py --surface 111
python examples/biomapsy/protein_workflow.py --protein 1CFD --mode tan
```

Each run writes to a new directory under `examples/results/`. Generated results are ignored by Git except for the included, verified k=32 decorated-surface run.

## CPU and GPU calculations

NumPy handles CPU calculations. GPU calculations require CuPy, a compatible CUDA device and the CUDA runtime/compiler libraries that CuPy uses, including NVRTC. A visible GPU alone does not establish that those libraries are available in a new environment. Choose `use_gpu=False` for CPU execution or `use_gpu=True` to request the GPU. Some calculations, including angular descriptor work, still run on the CPU. Check the model's `last_backend` when assessing GPU execution.

The [CPU/GPU example](examples/cpu_gpu/README.md) measures both paths with identical inputs. Speed depends on the structure, descriptors and batch size; the GPU is not faster for every workload.

## Tests

```bash
python -m unittest discover -s tests -v
python -m unittest discover -s examples/tests -v
```

The CPU workflow is configured to run these checks on every push and pull request, on Python 3.10, 3.11 and 3.12. It can also be started manually from GitHub Actions. Hosted CPU checks skip the device-dependent GPU test; the real-device test is recorded separately. Full scientific notebooks are not rerun by this workflow.

GPU tests skip when no usable CUDA device is available. Set `MAPSY_DISABLE_GPU=1` to run on the CPU. The [validation report](docs/VALIDATION_REPORT.md) records the test scope and known limits.

## Scientific conventions

MapSy uses one nearest periodic image per atom. If a descriptor cutoff requires additional images, enlarge the cell or check whether this convention fits your application. Skew cells use a reduced lattice to find nearest images without changing the structure.

BioMapSy's `tan` and `ionic-ass` contact models are aperiodic. The calmodulin examples contain protein heavy atoms only: 1CFD is apo; 1CLL is derived from a calcium-bound structure with the calcium ions omitted.

See the [descriptor and clustering details](examples/SCIENTIFIC_SCOPE.md), [implementation changes](docs/FULL_CHANGE_REPORT.md) and [calculation and memory notes](docs/OPTIMIZATIONS.md).md).

## Release and license

This is version 0.1.0, an initial research release. The package is named `mapsy`; imports remain `mapsy` and `mapsy.biomapsy`. The repository contains the worked examples, while an installed wheel contains the Python package.

MapSy and BioMapSy are distributed under the [MIT license](LICENSE). The original 2021 MATERIALab Group at Boise State University notice is preserved, with a separate notice for my contributions. See [AUTHORS.md](AUTHORS.md) for the credits.
