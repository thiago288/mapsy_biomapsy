# Periodic surfaces

Start with `00_classical_surfaces.ipynb` to reproduce the original center-of-mass distance3.5Å setup and comparison k10/k10/k4. Continue with `01_ideal_surfaces.ipynb` for the explicitly surface-referenced clearance1.5Å model, then `02_decorated_surface.ipynb` for vacancies/O. All use full descriptor/PCA/metric plotting with atomic overlays.

Ideal slabs:4×4×3, a3.94Å, vacuum10Å, cutoff80Ry, radial/angular structural AC radius4.5Å/orders0–9. Decorated slab:20×20×5, radius6.5Å, structural/compositional radial/angular. Probe positions are saved; the upper decorated plane lies above oxygen. Keep radius below half the shortest lateral repeat. Main code requires the fixed surface-reference API for the revised models. Package source itself retains the legacy default reference.
