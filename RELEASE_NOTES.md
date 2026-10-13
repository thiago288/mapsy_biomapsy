# MapSy and BioMapSy 0.1.0

This release brings together my optimized periodic MapSy implementation and BioMapSy for aperiodic surfaces. That development preceded the release preparation. The project description is in [README.md](README.md), and contributor credits are in [AUTHORS.md](AUTHORS.md).

## Corrections included in this release

Periodic distances now use a cached reduced lattice for skew cells while retaining fractional wrapping for orthogonal cells. Other corrections address contact-grid coordinates, repeated boundary updates and species-channel alignment in local neighborhoods. Radial Chebyshev orders are streamed to reduce temporary memory. The [change report](docs/FULL_CHANGE_REPORT.md) describes the affected calculations and intentional numerical differences.

## Examples

The repository includes classical and surface-referenced Pt(111), Pt(100) and Pt(110) tutorials, vacancy/O-decorated Pt(111), apo and holo-derived calmodulin, and CPU/GPU comparisons. The figures show descriptor channels, PCA, both clustering metrics and partitions in physical coordinates.

The decorated-surface example retains the metric-selected k=2 result and adds a k=32 comparison selected using stated descriptor-fidelity and seed-agreement requirements. It still merges fcc/hcp terrace sites and shows sensitivity to some held-out refits. The [resolution note](examples/preview/Ptdefects/CLUSTER_RESOLUTION.md) explains the measurements and limits.

## Validation and automated checks

A clean Python 3.12 installation of the wheel passed 40 CPU package tests and nine example tests. The remaining package test passed separately against the installed wheel on an NVIDIA RTX 4070 Laptop GPU with CuPy 14.0.1. Package metadata, the CLI and declared dependencies were checked. Scientific Python code was unchanged during the subsequent documentation and license updates.

The GitHub CPU workflow is configured for pushes, pull requests and manual runs on Python 3.10, 3.11 and 3.12. It installs the package and runs the package and example tests. Python 3.10/3.11 were not tested locally, and the workflow has not yet run on GitHub. Full scientific notebooks were verified separately and are not rerun on each push.

The [validation report](docs/VALIDATION_REPORT.md) records the test conditions and known limits. The [example scope](examples/SCIENTIFIC_SCOPE.md) explains how to interpret the scientific results.

## License

The MIT license preserves the original notice, `Copyright (c) 2021 MATERIALab Group at Boise State University`, and includes a separate notice for my contributions. The original permission and warranty text is retained in [LICENSE](LICENSE).
