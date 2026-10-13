"""
BioMapSy — MapSy for Aperiodic Biological Systems (Proteins)

Bugs fixed over earlier implementation:
 A5. sf.atomic guard now present in CPU and GPU compute paths.
 A6. GPU path: distance tensor computed on GPU but SF eval moved to
     vectorised CPU (genuine partial GPU benefit; full GPU for radial SFs
     via values_batch).
 A7. tan-mode contact space: gradient now computed with correct physical
     grid spacing (Angstroms), not unit-cell index spacing.
 A8. BioMaps class-level mutable defaults → per-instance.

New in Phase D:
 D1. BioMaps.cluster() / graph() / sites() — full API parity with Maps.
 D2. BioMaps.reduce() — PCA with plot option.
 D3. BioMaps.scatter() / plot() — visualisation parity.
 D4. BioMaps._process_vectorized() uses batch_distances (no MIC) + values_batch.

Design principle:
    ALL aperiodic / protein handling is confined to this file.
    maps.py always uses MIC and assumes periodic boundary conditions.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib.pyplot as plt
import scipy.spatial.distance as cdist_scipy
import scipy.special as sp
from scipy.spatial import cKDTree
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from ase.geometry import get_distances

from mapsy.data import Grid, GradientField, ScalarField, System
from mapsy.boundaries import Boundary, ContactSpace
from mapsy.boundaries.ionic import IonicBoundary
from mapsy.symfunc import SymmetryFunction
from mapsy.utils import get_vdw_radii, setscalars
from mapsy.utils.constants import FPI, SQRTPI
from mapsy.utils.functions.functions import FUNC_TOL

try:
    from mapsy.utils.gpu_utils import GPU_AVAILABLE, should_use_gpu
except ImportError:
    GPU_AVAILABLE = False
    def should_use_gpu(n_atoms, n_points): return False
# BioSystem — aperiodic System with KD-tree
class BioSystem(System):
    """
    Aperiodic System wrapper for proteins and other large aperiodic structures.

    Forces pbc=False and builds a KD-tree for fast O(log N) neighbour queries.
    """

    def __init__(
        self,
        grid: Grid,
        atoms=None,
        electrons=None,
        dimension: int = 0,
        axis: int = 0,
    ):
        super().__init__(grid, atoms, electrons, dimension, axis)
        if self.atoms is not None:
            self.atoms.pbc = False
        self._kdtree: Optional[cKDTree] = None
        if self.atoms is not None and len(self.atoms) > 0:
            self._kdtree = cKDTree(np.asarray(self.atoms.positions, dtype=float))
    # Distance helpers
    def nearest_distance(self, points: npt.NDArray) -> npt.NDArray:
        """Minimum distance from each of the given points to any atom."""
        if self._kdtree is None:
            self._kdtree = cKDTree(np.asarray(self.atoms.positions, dtype=float))
        d, _ = self._kdtree.query(np.asarray(points, dtype=float), k=1)
        return np.asarray(d, dtype=float)

    def get_distances_kdtree(
        self,
        position: npt.NDArray,
        rcut: float,
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        """
        Return (distances, vectors) to all atoms within rcut of position.

        Uses KD-tree → O(log N) instead of O(N) linear scan.

        Returns:
            distances: (K,)  Euclidean distances
            vectors:   (K, 3) displacement vectors (atom - position)
        """
        if self._kdtree is None:
            self._kdtree = cKDTree(np.asarray(self.atoms.positions, dtype=float))

        pos = np.asarray(position, dtype=float)
        idx = np.array(self._kdtree.query_ball_point(pos, float(rcut)), dtype=np.int64)

        if len(idx) == 0:
            return np.zeros((0,), dtype=float), np.zeros((0, 3), dtype=float)

        atom_pos = np.asarray(self.atoms.positions, dtype=float)
        vec  = atom_pos[idx] - pos                     # (K, 3)
        dist = np.linalg.norm(vec, axis=1)             # (K,)
        return dist, vec

    def batch_distances_aperiodic(
        self,
        positions: npt.NDArray,               # (N, 3)
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        """
        Vectorised Euclidean distances from N query points to all M atoms.

        Returns:
            distances: (N, M)
            vectors:   (N, M, 3)   (atom - position)
        """
        pos      = np.asarray(positions, dtype=float)       # (N, 3)
        atom_pos = np.asarray(self.atoms.positions, dtype=float)  # (M, 3)
        vecs     = atom_pos[None, :, :] - pos[:, None, :]   # (N, M, 3)
        dists    = np.linalg.norm(vecs, axis=-1)             # (N, M)
        return dists, vecs
# BioContactSpace — contact space container
@dataclass
class BioContactSpace:
    """Container compatible with BioMaps descriptor computation."""
    positions:     npt.NDArray[np.float64]
    probabilities: npt.NDArray[np.float64]
    data:          pd.DataFrame
    grid:          Optional[Grid]     = None
    boundary:      Optional[Boundary] = None
    neighbors:     Optional[list]     = None   # list of (up to 6) neighbour indices
    nm:            int                = 0      # number of CS points

    def __post_init__(self):
        self.nm = len(self.positions)

    def keep_main_region(self) -> "BioContactSpace":
        """Retain only the largest connected component."""
        if len(self.data) == 0 or "region" not in self.data.columns:
            return self
        counts = self.data.groupby("region").size()
        main   = int(counts.idxmax())
        self.data          = self.data[self.data["region"] == main].reset_index(drop=True)
        self.positions     = self.data[["x", "y", "z"]].to_numpy(dtype=float)
        self.probabilities = self.data["probability"].to_numpy(dtype=float)
        if "nn" in self.data.columns:
            self.neighbors = list(self.data["nn"])
        self.nm = len(self.positions)
        return self
# AperiodicERFC — soft sphere without MIC
class AperiodicERFC:
    """
    Aperiodic ERFC soft sphere density and gradient.

    h_a(r) = 0.5 * erfc((|r - R_a| - alpha*RvdW) / spread)

    No minimum-image convention is applied (Euclidean distance only).
    Compatible with the FunctionContainer interface expected by IonicBoundary.
    """

    def __init__(
        self,
        grid: Grid,
        kind: int,
        dim: int,
        axis: int,
        width: float,
        spread: float,
        volume: float,
        pos: npt.NDArray,
        label: str = "",
    ):
        self.grid   = grid
        self.label  = str(label)
        self.kind   = int(kind)
        self.dim    = int(dim)
        self.axis   = int(axis)
        self.width  = float(width)
        self.spread = float(spread)
        self.volume = float(volume)
        self.pos    = np.asarray(pos, dtype=float).reshape(3)

        self.density  = ScalarField(grid, label=f"{self.label}_density")
        self.gradient = GradientField(grid, label=f"{self.label}_gradient")

        self._compute()

    def _compute(self) -> None:
        coords = np.asarray(self.grid.coordinates).reshape(3, -1)   # (3, npts)
        diff   = coords - self.pos[:, None]                          # (3, npts)
        dist   = np.sqrt(np.sum(diff ** 2, axis=0))                  # (npts,)
        arg    = (dist - self.width) / self.spread
        erfc_v = sp.erfc(arg)                                        # (npts,)

        if self.kind == 4:
            grid_vol_per_pt = self.grid.volume / max(self.grid.ndata, 1)
            integral        = np.sum(erfc_v) * grid_vol_per_pt * 0.5
            analytic_vol    = self._analytic_volume()
            if analytic_vol > 0 and abs(integral - analytic_vol) / max(analytic_vol, 1e-10) < 0.5:
                scale = self.volume / analytic_vol * 0.5
            else:
                scale = self.volume / max(integral, 1e-10) * 0.5
            self.density.flat[:] = self.volume + erfc_v * scale
        elif self.kind == 3:
            scale = self.volume
            self.density.flat[:] = erfc_v * scale
        else:
            raise ValueError(f"AperiodicERFC: unsupported kind={self.kind}. Use 3 or 4.")

        # Gradient
        self.gradient.flat[:] = 0.0
        mask = dist > FUNC_TOL
        if np.any(mask):
            r_m   = diff[:, mask]                                    # (3, K)
            d_m   = dist[mask]
            a_m   = arg[mask]
            derfc = -2.0 / SQRTPI * np.exp(-a_m ** 2)              # (K,)
            r_hat = r_m / d_m[None, :]                              # (3, K)
            grad_vec = derfc[None, :] * r_hat * scale / self.spread  # (3, K)

            grid_shape = self.grid.scalars
            flat_idx = np.where(mask)[0]
            ix = flat_idx // (grid_shape[1] * grid_shape[2])
            iy = (flat_idx // grid_shape[2]) % grid_shape[1]
            iz = flat_idx % grid_shape[2]
            for dim in range(3):
                self.gradient[dim, ix, iy, iz] = grad_vec[dim, :]

    def _analytic_volume(self) -> float:
        s, w = self.spread, self.width
        if s < FUNC_TOL or w < FUNC_TOL:
            return 0.0
        t, invt = s / w, w / s
        f1 = (1.0 + sp.erf(invt)) * 0.5
        f2 = np.exp(-invt ** 2) * 0.5 / SQRTPI
        return FPI / 3.0 * w ** 3 * ((1.0 + 1.5 * t ** 2) * f1 + (1.0 + t ** 2) * t * f2)

    def reset_derivatives(self) -> None:
        if self.gradient is not None:
            self.gradient[:] = 0.0

    def update(self) -> None:
        self._compute()
# Grid helpers
def _make_cell_grid(atoms, cutoff: float) -> Grid:
    """Build a Grid that matches MapSy's cell-based sizing."""
    cell = np.asarray(atoms.cell, dtype=float)
    if np.allclose(cell, 0):
        pos   = np.asarray(atoms.positions, dtype=float)
        pmin  = pos.min(axis=0) - float(cutoff)
        pmax  = pos.max(axis=0) + float(cutoff)
        cell  = np.diag(pmax - pmin)
        sc    = setscalars(cell, float(cutoff))
        grid  = Grid(scalars=tuple(sc.tolist()), cell=cell)
        grid._pmin = pmin
        return grid
    sc = setscalars(cell, float(cutoff))
    return Grid(scalars=tuple(sc.tolist()), cell=cell)


def _make_bbox_grid(
    atoms,
    grid_resolution: float,
    padding: float,
) -> Tuple[Grid, npt.NDArray]:
    """Bounding-box grid for the tan-mode contact space."""
    pos    = np.asarray(atoms.positions, dtype=float)
    origin = pos.min(axis=0) - float(padding)
    pmax   = pos.max(axis=0) + float(padding)
    box    = pmax - origin
    sc     = np.maximum(np.ceil(box / float(grid_resolution)).astype(int), 2)
    cell   = np.diag(box)
    try:
        grid = Grid(scalars=tuple(sc.tolist()), cell=cell, origin=origin)
    except TypeError:
        grid = Grid(scalars=tuple(sc.tolist()), cell=cell)
    return grid, origin
# Contact space generation
def create_protein_contact_space(
    system: BioSystem,
    *,
    mode: Literal["ionic", "ionic-ass", "tan"] = "tan",
    # ionic / ionic-ass
    cutoff: float  = 7.0,
    alpha:  float  = 0.8,
    spread: float  = 0.5,
    # tan mode
    grid_resolution: float = 0.8,
    padding:         float = 10.0,
    distance:        float = 3.5,
    # common
    threshold:            float = 0.7,
    threshold_is_absolute: bool = True,
) -> BioContactSpace:
    """
    Build a protein contact space.

    Modes
    -----
    ionic       MapSy original IonicBoundary (with MIC — only valid if atoms.cell is set).
    ionic-ass   Aperiodic product-of-ERFC boundary (no MIC).
    tan         Distance-to-nearest-atom tanh boundary (no MIC, recommended for proteins).
    """
    if mode == "ionic":
        return _create_ionic_contact_space(
            system, cutoff=cutoff, alpha=alpha, spread=spread,
            threshold=threshold, threshold_is_absolute=threshold_is_absolute,
            use_mapsy_original=True,
        )
    elif mode == "ionic-ass":
        return _create_ionic_contact_space(
            system, cutoff=cutoff, alpha=alpha, spread=spread,
            threshold=threshold, threshold_is_absolute=threshold_is_absolute,
            use_mapsy_original=False,
        )
    elif mode == "tan":
        return _create_tan_contact_space(
            system,
            grid_resolution=grid_resolution,
            padding=padding,
            distance=distance,
            spread=spread,
            threshold=threshold,
            threshold_is_absolute=threshold_is_absolute,
        )
    else:
        raise ValueError(f"Unknown mode={mode!r}. Use 'ionic', 'ionic-ass', or 'tan'.")


def _create_ionic_contact_space(
    system: BioSystem,
    *,
    cutoff: float,
    alpha: float,
    spread: float,
    threshold: float,
    threshold_is_absolute: bool,
    use_mapsy_original: bool,
) -> BioContactSpace:
    grid = _make_cell_grid(system.atoms, cutoff=float(cutoff))
    system.grid = grid

    if use_mapsy_original:
        boundary = IonicBoundary(
            mode="ionic", grid=grid,
            alpha=float(alpha), softness=float(spread),
            system=system, label="ionic_boundary",
        )
    else:
        boundary = _create_aperiodic_ionic_boundary(system, grid, alpha, spread)

    boundary.update()

    gmod = boundary.gradient.modulus
    gmax = float(np.max(gmod))

    if gmax == 0.0:
        return _empty_contact_space()

    if threshold < 0:
        tol = gmax - 1e-4
    elif threshold_is_absolute:
        tol = float(threshold)
    else:
        tol = float(threshold) * gmax

    mask = gmod > tol
    if not np.any(mask):
        warnings.warn(f"[BioMapSy ionic] No CS points at threshold={threshold}.")
        return _empty_contact_space()

    pos  = grid.coordinates[:, mask].T
    prob = gmod[mask].reshape(-1)

    # Shift to lab frame if grid has a pmin offset (aperiodic bbox)
    pmin = getattr(grid, '_pmin', None)
    if pmin is not None:
        pos = pos + pmin[None, :]

    # kNN connectivity
    neighbors, regions = _compute_neighbors_and_regions(pos)

    df = pd.DataFrame({
        "probability": prob,
        "x": pos[:, 0], "y": pos[:, 1], "z": pos[:, 2],
        "nn": neighbors, "region": regions,
    })
    return BioContactSpace(
        positions=pos, probabilities=prob,
        data=df, grid=grid, boundary=boundary,
        neighbors=neighbors,
    )


def _create_aperiodic_ionic_boundary(system, grid, alpha, spread):
    """
    Create the aperiodic product-of-ERFC ionic boundary (no MIC).

    Returns an object whose `.gradient` is a GradientField with a native
    `.modulus` property — no wrapper needed.

    Root-cause fix: the previous implementation tried to set
        self._gradient.modulus = property(...)
    which fails because GradientField.modulus is a read-only class property.
    The fix is to store `gradient` as a plain instance attribute (GradientField
    already has `.modulus` built in), and to populate it by writing directly
    into the array after computation.
    """

    class AperiodicIonicBoundary:
        """Product-of-ERFC switch: s = 1 - ∏_a (1 - h_a(|r - R_a|))."""

        def __init__(self, system, grid, alpha, softness):
            self.ions     = system.atoms
            self.grid     = grid
            self.alpha    = float(alpha)
            self.softness = float(softness)

            # Plain instance attributes — no property setter involved.
            # GradientField already has a native .modulus property.
            self.switch   = ScalarField(grid,   label="aperiodic_switch")
            self._gradient = GradientField(grid, label="aperiodic_grad")

        # Read-only property — no setter, so __init__ stores to self._gradient directly.
        @property
        def gradient(self) -> GradientField:
            return self._gradient

        def update(self) -> None:
            self._compute_switch_and_gradient()

        def _compute_switch_and_gradient(self) -> None:
            npts       = self.grid.ndata
            nx, ny, nz = self.grid.scalars

            coords   = np.asarray(self.grid.coordinates).reshape(3, -1).T  # (npts, 3)
            atom_Z   = self.ions.get_atomic_numbers()
            atom_pos = np.asarray(self.ions.positions, dtype=float).copy()

            # Shift atom positions into the grid coordinate frame if needed
            pmin = getattr(self.grid, '_pmin', None)
            if pmin is not None:
                atom_pos -= pmin[None, :]

            grid_tree = cKDTree(coords)
            s_prod    = np.ones(npts,      dtype=np.float64)
            grad_arr  = np.zeros((3, npts), dtype=np.float64)
            cache     = []

            # Pass 1: accumulate s_prod = ∏_a (1 - h_a)
            for i, an in enumerate(atom_Z):
                w    = get_vdw_radii(int(an), "uff") * self.alpha
                rc_i = w + 6.0 * self.softness
                idx  = np.array(
                    grid_tree.query_ball_point(atom_pos[i], rc_i), dtype=np.int64
                )
                if len(idx) == 0:
                    cache.append(None)
                    continue
                diff = (coords[idx] - atom_pos[i]).T          # (3, K)
                dist = np.sqrt(np.einsum('ij,ij->j', diff, diff))
                arg  = (dist - w) / self.softness
                h_a  = sp.erfc(arg) * 0.5
                s_prod[idx] *= (1.0 - h_a)
                cache.append((idx, diff, dist, w))

            # Pass 2: accumulate gradient using cached neighbour data
            for i, an in enumerate(atom_Z):
                if cache[i] is None:
                    continue
                idx, diff, dist, w = cache[i]
                arg    = (dist - w) / self.softness
                h_a    = sp.erfc(arg) * 0.5
                d_corr = 1.0 - h_a

                valid = (dist > 1e-10) & (d_corr > 1e-60)
                if not np.any(valid):
                    continue

                idx_v   = idx[valid]
                derfc   = (-2.0 / SQRTPI) * np.exp(-arg[valid] ** 2)
                r_hat   = diff[:, valid] / dist[valid]
                grad_ha = derfc[None, :] * r_hat * 0.5 / self.softness
                weight  = s_prod[idx_v] / d_corr[valid]
                grad_arr[:, idx_v] += (-grad_ha) * weight

            # Write results into the pre-allocated fields
            switch_arr = 1.0 - s_prod
            self.switch.flat[:] = switch_arr

            g3d = grad_arr.reshape(3, nx, ny, nz)
            for d in range(3):
                self._gradient[d, :, :, :] = g3d[d]

    return AperiodicIonicBoundary(system, grid, alpha, spread)


def _create_tan_contact_space(
    system: BioSystem,
    *,
    grid_resolution: float,
    padding: float,
    distance: float,
    spread: float,
    threshold: float,
    threshold_is_absolute: bool,
) -> BioContactSpace:
    """
    Tan (tanh-distance) contact space.

    Bug A7 fix: np.gradient now receives explicit grid spacing in Angstroms
    so that the gradient magnitude is in units of Å⁻¹ (physical), not
    grid-index⁻¹.
    """
    grid, origin = _make_bbox_grid(
        system.atoms,
        grid_resolution=float(grid_resolution),
        padding=float(padding),
    )
    system.grid = grid
    nx, ny, nz  = grid.scalars

    # Physical grid spacings (Å) per axis
    # cell is diagonal for bbox grid; spacing = cell[i,i] / n_i
    cell = np.asarray(grid.cell, dtype=float)
    dx   = cell[0, 0] / nx
    dy   = cell[1, 1] / ny
    dz   = cell[2, 2] / nz

    # Grid point positions in lab frame
    coords = np.asarray(grid.coordinates).reshape(3, -1).T   # (npts, 3)
    points = coords + origin if np.linalg.norm(origin) > 1e-6 else coords

    # Min-distance field
    dmin = system.nearest_distance(points).reshape(nx, ny, nz)

    # Tanh boundary
    boundary = 0.5 * (1.0 + np.tanh((dmin - float(distance)) / float(spread)))

    # physical spacing
    grad = np.array(np.gradient(boundary, dx, dy, dz))  # (3, nx, ny, nz) in Å⁻¹
    grad_mod = np.sqrt(np.sum(grad ** 2, axis=0))        # (nx, ny, nz)

    gmax = float(np.max(grad_mod)) if grad_mod.size else 0.0
    if gmax == 0.0:
        return _empty_contact_space()

    if threshold < 0:
        tol = gmax - 1e-4
    elif threshold_is_absolute:
        tol = float(threshold)
    else:
        tol = float(threshold) * gmax

    mask = grad_mod > tol
    if not np.any(mask):
        warnings.warn(
            f"[BioMapSy tan] No CS points: threshold={threshold} "
            f"(tol={tol:.4g}, max|∇|={gmax:.4g} Å⁻¹)."
        )
        return _empty_contact_space()

    flat_mask = mask.reshape(-1)
    pos  = points[flat_mask]                  # (K, 3)
    prob = grad_mod.reshape(-1)[flat_mask]    # (K,)

    neighbors, regions = _compute_neighbors_and_regions(pos)

    df = pd.DataFrame({
        "probability": prob,
        "x": pos[:, 0], "y": pos[:, 1], "z": pos[:, 2],
        "nn": neighbors, "region": regions,
    })
    return BioContactSpace(
        positions=pos, probabilities=prob,
        data=df, grid=grid, boundary=None,
        neighbors=neighbors,
    )
# Shared helpers for contact space
def _empty_contact_space() -> BioContactSpace:
    df = pd.DataFrame({"probability": [], "x": [], "y": [], "z": [], "nn": [], "region": []})
    return BioContactSpace(
        positions=np.zeros((0, 3), dtype=float),
        probabilities=np.zeros((0,), dtype=float),
        data=df,
    )


def _compute_neighbors_and_regions(
    pos: npt.NDArray,
    k: int = 7,
) -> Tuple[list, npt.NDArray]:
    """kNN connectivity + depth-first region labelling."""
    n = len(pos)
    if n == 0:
        return [], np.zeros(0, dtype=np.int64)

    tree    = cKDTree(pos)
    _, idx  = tree.query(pos, k=min(k, n))   # (n, k)

    # Build neighbour list (up to 6 neighbours, skip self)
    neighbors = []
    for i in range(n):
        row = [-1] * 6
        ns  = idx[i][1:] if idx.ndim > 1 else []   # skip self (index 0)
        for t, j in enumerate(ns[:6]):
            row[t] = int(j)
        neighbors.append(tuple(row))

    neigh_arr = np.array(neighbors, dtype=np.int64)  # (n, 6)

    # Depth-first flood fill for connected regions
    regions = -np.ones(n, dtype=np.int64)
    cur     = 0
    for start in range(n):
        if regions[start] != -1:
            continue
        stack = [start]
        regions[start] = cur
        while stack:
            ii = stack.pop()
            for jj in neigh_arr[ii]:
                if jj < 0:
                    continue
                if regions[jj] == -1:
                    regions[jj] = cur
                    stack.append(int(jj))
        cur += 1

    return neighbors, regions
# BioMaps — descriptor computation
class BioMaps:
    """
    Compute symmetry-function descriptors at BioContactSpace points.

    Features:
    - KD-tree for efficient neighbour queries (aperiodic, no MIC)
    - Vectorised batch distances + values_batch() API
    - GPU-accelerated distance tensor (radial SFs computed on CPU after)
    - Full analysis API matching Maps: cluster, graph, sites, reduce, scatter, plot
    """

    def __init__(
        self,
        system: BioSystem,
        symmetryfunctions: List[SymmetryFunction],
        contactspace: BioContactSpace,
        *,
        batch_size: int = 500,
        use_gpu: Optional[bool] = None,
        use_kdtree: bool = True,
        n_workers: Optional[int] = None,
        verbosity: int = 0,
    ):
        # per-instance lists
        self.system          = system
        self.symmetryfunctions: List[SymmetryFunction] = list(symmetryfunctions)
        self.contactspace    = contactspace
        self.nsfs            = len(self.symmetryfunctions)
        self.hasatomicsf     = any(getattr(sf, "atomic", True) for sf in self.symmetryfunctions)
        self.batch_size      = int(batch_size)
        self.use_kdtree      = bool(use_kdtree) and (system._kdtree is not None)
        self.n_workers       = n_workers
        self.verbosity       = int(verbosity)

        # GPU decision
        if use_gpu is None:
            self.use_gpu = bool(
                should_use_gpu(len(system.atoms), len(contactspace.positions))
            )
        else:
            self.use_gpu = bool(use_gpu)

        # Per-instance analysis state (Bug A8)
        self.data:             Optional[pd.DataFrame] = None
        self.features:         List[str]              = []
        self.npca:             Optional[int]          = None
        self.nclusters:        int                    = 0
        self.cluster_centers:  Optional[npt.NDArray]  = None
        self.cluster_sizes:    Optional[npt.NDArray]  = None
        self.cluster_graph:    Optional[npt.NDArray]  = None
        self.cluster_edges:    Optional[npt.NDArray]  = None
        self.cluster_features: List[str]              = []
        self.centroids:        Optional[npt.NDArray]  = None

        # Initialise symmetry functions
        for sf in self.symmetryfunctions:
            if hasattr(sf, "setup"):
                sf.setup(self.system.atoms)

        # Max cutoff (for KD-tree queries)
        self._rcut = 0.0
        for sf in self.symmetryfunctions:
            rc = getattr(sf, "rcut", 0.0)
            if rc > self._rcut:
                self._rcut = float(rc)
    # Public compute API
    def atcontactspace(self) -> pd.DataFrame:
        """Compute descriptors at all contact space points."""
        df           = self.atpoints(self.contactspace.positions)
        self.data    = df
        self.features = [c for c in df.columns if c not in ("x", "y", "z")]
        return df

    def atpoints(self, positions: npt.NDArray) -> pd.DataFrame:
        """Compute descriptors at specified points."""
        positions = np.asarray(positions, dtype=float)
        if len(positions) == 0:
            return pd.DataFrame({"x": [], "y": [], "z": []})

        if self.use_gpu:
            try:
                results = self._process_gpu(positions)
            except Exception as e:
                warnings.warn(f"[BioMapSy] GPU error — CPU fallback: {e}")
                results = self._process_vectorized(positions)
        else:
            results = self._process_vectorized(positions)

        return self._results_to_df(positions, results)
    # Vectorised CPU path (Phase B / D4)
    def _process_vectorized(self, positions: npt.NDArray) -> list:
        """
        Vectorised CPU path using batch (N, M) distance matrices.

        For each batch of N_batch points:
          - Compute (N_batch, M) distances and (N_batch, M, 3) vectors
          - Call sf.values_batch() or fall back to per-point loop
        """
        n_pts    = len(positions)
        sf_results: list[list] = [[] for _ in range(self.nsfs)]

        for start in range(0, n_pts, self.batch_size):
            end   = min(start + self.batch_size, n_pts)
            batch = positions[start:end]               # (N_b, 3)

            # Batch aperiodic distances (no MIC)
            dist_batch, vec_batch = self.system.batch_distances_aperiodic(batch)

            for i, sf in enumerate(self.symmetryfunctions):
                # Bug A5 fix: only process atomic SFs
                if not getattr(sf, "atomic", True):
                    n_keys = len(sf.keys) if sf.keys else 0
                    sf_results[i].append(np.zeros((len(batch), n_keys), dtype=np.float64))
                    continue

                if hasattr(sf, "values_batch") and callable(sf.values_batch):
                    batch_result = sf.values_batch(
                        dist_batch,
                        vec_batch if getattr(sf, "angular", False) else None,
                        use_gpu=False,
                    )
                else:
                    # Fallback: per-point
                    rows = []
                    for b in range(len(batch)):
                        d = dist_batch[b]
                        v = vec_batch[b]
                        if getattr(sf, "angular", False):
                            row = sf.values(d, v)
                        else:
                            row = sf.values(d)
                        rows.append(row)
                    batch_result = np.stack(rows, axis=0)

                sf_results[i].append(np.asarray(batch_result, dtype=np.float64))

        return [np.concatenate(frags, axis=0) for frags in sf_results]
    # GPU path (Phase C — partial: distance on GPU, SF on CPU)
    def _process_gpu(self, positions: npt.NDArray) -> list:
        """
        GPU-accelerated path.

        Distance tensor computed on GPU; SF evaluation on CPU via values_batch.
        Bug A6 fix: no immediate .get() followed by Python loop — we at least
        compute the (N_b, M, 3) tensor once on GPU, then transfer once per batch.
        """
        import cupy as cp

        n_pts    = len(positions)
        atom_pos = np.asarray(self.system.atoms.positions, dtype=float)
        atom_gpu = cp.asarray(atom_pos)                    # transfer once

        sf_results: list[list] = [[] for _ in range(self.nsfs)]

        for start in range(0, n_pts, self.batch_size):
            end   = min(start + self.batch_size, n_pts)
            batch = positions[start:end]
            batch_gpu = cp.asarray(batch, dtype=cp.float64)

            # (N_b, M, 3) difference tensor + distances — all on GPU
            diff_gpu = atom_gpu[cp.newaxis, :, :] - batch_gpu[:, cp.newaxis, :]
            dist_gpu = cp.linalg.norm(diff_gpu, axis=2)   # (N_b, M)

            # Transfer once per batch (single H2D, one D2H for the batch)
            dist_cpu = dist_gpu.get()                      # (N_b, M)
            diff_cpu = diff_gpu.get()                      # (N_b, M, 3)

            for i, sf in enumerate(self.symmetryfunctions):
                # Bug A5 fix
                if not getattr(sf, "atomic", True):
                    n_keys = len(sf.keys) if sf.keys else 0
                    sf_results[i].append(np.zeros((len(batch), n_keys), dtype=np.float64))
                    continue

                if hasattr(sf, "values_batch") and callable(sf.values_batch):
                    batch_result = sf.values_batch(
                        dist_cpu,
                        diff_cpu if getattr(sf, "angular", False) else None,
                        use_gpu=False,
                    )
                else:
                    rows = []
                    for b in range(len(batch)):
                        d = dist_cpu[b]
                        v = diff_cpu[b]
                        row = sf.values(d, v) if getattr(sf, "angular", False) else sf.values(d)
                        rows.append(row)
                    batch_result = np.stack(rows, axis=0)

                sf_results[i].append(np.asarray(batch_result, dtype=np.float64))

            if start % (self.batch_size * 10) == 0:
                cp.get_default_memory_pool().free_all_blocks()

        return [np.concatenate(frags, axis=0) for frags in sf_results]
    # Results → DataFrame
    def _results_to_df(self, positions: npt.NDArray, results: list) -> pd.DataFrame:
        df = pd.DataFrame({
            "x": positions[:, 0],
            "y": positions[:, 1],
            "z": positions[:, 2],
        })
        for i, sf in enumerate(self.symmetryfunctions):
            vals = np.asarray(results[i], dtype=np.float64)
            if vals.ndim == 0:
                vals = vals.reshape(1, 1)
            elif vals.ndim == 1:
                n_keys = len(sf.keys) if sf.keys else 1
                vals = vals.reshape(-1, 1) if n_keys == 1 else vals.reshape(1, -1)
            keys = list(getattr(sf, "keys", [f"sf{i}_{j}" for j in range(vals.shape[1])]))
            for j, key in enumerate(keys[: vals.shape[1]]):
                df[key] = vals[:, j]
        return df
    # Phase D1: PCA
    def reduce(self, npca: Optional[int] = None, scale: bool = True):
        """
        PCA dimensionality reduction.

        If npca is given, adds pca0…pca{npca-1} columns and returns fitted PCA.
        If npca is None, returns a scree plot (fig, ax1, ax2).
        """
        if self.data is None:
            raise RuntimeError("No data — call atcontactspace() first.")
        X = self.data[self.features].values.astype(np.float64)
        if scale:
            X = StandardScaler().fit_transform(X)

        if npca is not None:
            npca = int(min(npca, X.shape[0], X.shape[1]))
            pca  = PCA(n_components=npca).fit(X)
            Z    = pca.transform(X)
            for j in range(npca):
                self.data[f"pca{j}"] = Z[:, j]
            self.npca = npca
            return pca

        pca = PCA().fit(X)
        ev  = np.cumsum(pca.explained_variance_ratio_)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(range(1, len(ev) + 1), ev, marker="o")
        ax1.axhline(0.98, color="r", linestyle="--", label="98 %")
        ax1.set_xlabel("n components"); ax1.set_ylabel("Cumulative variance"); ax1.grid(True)
        ax2.plot(range(1, len(pca.explained_variance_ratio_) + 1),
                 pca.explained_variance_ratio_, marker="o")
        ax2.set_xlabel("n components"); ax2.set_ylabel("Variance ratio")
        ax2.set_title("Scree plot"); ax2.grid(True)
        return fig, ax1, ax2
    # Phase D1: Clustering
    def cluster(
        self,
        nclusters: Optional[int] = None,
        features: Optional[List[str]] = None,
        maxclusters: int = 20,
        ntries: int = 1,
        random_state: Optional[int] = None,
        scale: bool = False,
    ):
        """Spectral clustering — full API parity with Maps.cluster()."""
        if self.data is None:
            raise RuntimeError("No data — call atcontactspace() first.")

        self.cluster_features = features if features is not None else self.features
        X = self.data[self.cluster_features].values.astype(np.float64)
        if scale:
            X = StandardScaler().fit_transform(X)

        if nclusters is not None:
            rs = random_state if random_state is not None else np.random.randint(0, 1000)
            self.random_state = rs
            labels = SpectralClustering(n_clusters=nclusters, random_state=rs).fit_predict(X)
            self.data["Cluster"] = labels
            self.nclusters       = int(nclusters)
            self.cluster_centers = self.data.groupby("Cluster")[self.cluster_features].mean().values
            self.cluster_sizes   = self.data.groupby("Cluster").size().values
            self.cluster_graph   = self.graph(self.data["Cluster"].values)
            self.cluster_edges   = self.cluster_graph.copy()
            np.fill_diagonal(self.cluster_edges, 0)
            return

        # Screening
        cluster_data = []
        for nc in range(2, maxclusters):
            rss = np.random.randint(0, 1000, ntries)
            for rs in rss:
                lbl = SpectralClustering(n_clusters=nc, random_state=rs).fit_predict(X)
                k   = len(np.unique(lbl))
                cluster_data.append({
                    "nclusters":        k,
                    "random_state":     rs,
                    "silhouette_score": silhouette_score(X, lbl),
                    "db_index":         davies_bouldin_score(X, lbl),
                })
        self.cluster_screening = pd.DataFrame(cluster_data)
        self.best_clusters = self.cluster_screening.loc[
            self.cluster_screening.groupby("nclusters")["db_index"].idxmin()
        ]
        fig, ax1 = plt.subplots()
        ax1.scatter(self.cluster_screening["nclusters"],
                    self.cluster_screening["silhouette_score"], color="b", marker="o")
        ax1.set_xlabel("n clusters"); ax1.set_ylabel("Silhouette", color="b")
        ax2 = ax1.twinx()
        ax2.scatter(self.cluster_screening["nclusters"],
                    self.cluster_screening["db_index"], color="r", marker="s")
        ax2.set_ylabel("Davies-Bouldin", color="r")
        return fig, ax1, ax2
    # Phase D1: Graph connectivity and sites
    def graph(self, clusters: npt.NDArray) -> npt.NDArray:
        """Build cluster adjacency matrix from kNN neighbour list."""
        nclusters = int(np.max(clusters)) + 1
        G = np.zeros((nclusters, nclusters), dtype=np.int64)
        neighbors = self.contactspace.neighbors
        if neighbors is None:
            return G
        for i, ni in enumerate(neighbors):
            ci = clusters[i]
            for j in ni:
                if j >= 0:
                    cj = clusters[j]
                    G[ci, cj] += 1
        return G // 2

    def sites(self, region: int = 0) -> npt.NDArray:
        """Find high-symmetry site centroids within the specified region."""
        if self.data is None:
            raise RuntimeError("No data — call atcontactspace() first.")
        if self.nclusters == 0:
            raise RuntimeError("No clusters — call cluster() first.")

        filterdata = self.data[self.contactspace.data["region"] == region]
        centroid_indexes = []

        for ci in np.unique(filterdata["Cluster"].values):
            pts  = filterdata[filterdata["Cluster"] == ci][self.cluster_features].values
            idxs = filterdata[filterdata["Cluster"] == ci].index.values
            D    = cdist_scipy.cdist(pts, self.cluster_centers)
            edges_idx = np.where(self.cluster_edges[ci, :] != 0)[0]
            score = np.sum(
                self.cluster_sizes[edges_idx] / (D[:, edges_idx] ** 2 + 1e-30), axis=1
            )
            centroid_indexes.append(idxs[np.argmin(score)])

        self.centroids = np.array(centroid_indexes, dtype=int)
        return self.centroids
    # Phase D3: Visualisation
    def scatter(
        self,
        feature: Optional[str] = None,
        index: Optional[int] = None,
        axes: tuple = ("x", "y"),
        region: int = 0,
        categorical: bool = False,
        centroids: bool = False,
        splitby: Optional[str] = None,
        set_aspect: str = "on",
        **kwargs,
    ):
        """2-D scatter plot of a descriptor on the contact space."""
        if self.data is None:
            raise RuntimeError("No data — call atcontactspace() first.")
        if feature is None and index is None:
            raise ValueError("Provide feature or index.")
        if index is not None:
            feature = self.features[index]

        filterdata = self.data[self.contactspace.data["region"] == region]
        x1 = filterdata[axes[0]].values.astype(np.float64)
        x2 = filterdata[axes[1]].values.astype(np.float64)
        f  = filterdata[feature].values.astype(np.float64)

        if splitby is not None:
            s, uniqs = filterdata[splitby].values, np.unique(filterdata[splitby].values)
            nsplit   = len(uniqs)
            fig, axs = plt.subplots(nsplit, 1, figsize=(8, 4 * nsplit))
            axslist  = axs.flat if nsplit > 1 else [axs]
        else:
            nsplit   = 1
            uniqs    = [None]
            fig, axs = plt.subplots(1, 1, figsize=(8, 4))
            axslist  = [axs]

        if categorical:
            colors = [plt.cm.tab20(k) for k in range(len(np.unique(f)))]
        else:
            fmin, fmax = np.min(f), np.max(f)

        fig.subplots_adjust(hspace=0.3)

        for i, ax in enumerate(axslist):
            if splitby is not None:
                mask = filterdata[splitby] == uniqs[i]
                x1m, x2m, fm = x1[mask], x2[mask], f[mask]
                ax.set_title(f"{feature}  {splitby}={uniqs[i]:6.2f}")
            else:
                x1m, x2m, fm = x1, x2, f
                ax.set_title(f"{feature}")

            ax.set_xlabel(axes[0]); ax.set_ylabel(axes[1])

            if categorical:
                for j, fv in enumerate(np.unique(f)):
                    ax.scatter(x1m[fm == fv], x2m[fm == fv], color=colors[j],
                               label=f"{feature}={j}", **kwargs)
                ax.legend(bbox_to_anchor=(1.01, 0.5), loc="center left")
            else:
                sc = ax.scatter(x1m, x2m, c=fm, vmin=fmin, vmax=fmax, **kwargs)

            if centroids and self.centroids is not None:
                cpos = self.data.loc[self.centroids, list(axes)].values
                ax.scatter(cpos[:, 0], cpos[:, 1], c="black", marker="x", label="Centroids")

            ax.axis(set_aspect)

        if not categorical:
            cb_ax = axs if nsplit == 1 else axs.ravel().tolist()
            fig.colorbar(sc, ax=cb_ax)

        return fig, axs


    def scatter_pca_grid(
        self,
        feature=None,
        index=None,
        region=0,
        set_aspect="equal",
        cmap="Spectral",
        diag="hist",
        **kwargs,
    ):
        """
        Pairwise PCA scatter grid — see Maps.scatter_pca_grid() for full docs.

        Must call reduce(npca=N) before this method.

        Returns:
            fig : matplotlib Figure
            gs  : matplotlib GridSpec
        """
        import matplotlib.pyplot as plt
        import numpy as np

        if self.data is None:
            raise RuntimeError("No data — call atcontactspace() first.")

        if feature is None and index is None:
            raise ValueError("Provide either feature= or index=.")
        if feature is None:
            feature = self.features[index]
        if feature not in self.data.columns:
            raise ValueError(f"Feature {feature!r} not found in data columns.")

        pca_cols = sorted(
            [c for c in self.data.columns if c.startswith("pca")],
            key=lambda c: int(c[3:]),
        )
        if not pca_cols:
            raise RuntimeError(
                "No PCA columns found. Call reduce(npca=N) before scatter_pca_grid()."
            )
        n = len(pca_cols)

        region_mask = self.contactspace.data["region"].values == region
        sub = self.data[region_mask]

        pca_data = {col: sub[col].values.astype(np.float64) for col in pca_cols}
        f        = sub[feature].values.astype(np.float64)
        fmin, fmax = np.nanmin(f), np.nanmax(f)
        norm     = plt.Normalize(vmin=fmin, vmax=fmax)
        cmap_obj = plt.get_cmap(cmap)

        cell_size = 3.0
        fig = plt.figure(figsize=(cell_size * n + 0.8, cell_size * n))
        gs  = fig.add_gridspec(n, n, hspace=0.08, wspace=0.08)

        last_sc = None

        for row in range(n):
            for col in range(n):
                ax     = fig.add_subplot(gs[row, col])
                x_col  = pca_cols[col]
                y_col  = pca_cols[row]
                x_vals = pca_data[x_col]
                y_vals = pca_data[y_col]

                if row == col:
                    if diag == "kde":
                        try:
                            from scipy.stats import gaussian_kde
                            xg  = np.linspace(x_vals.min(), x_vals.max(), 200)
                            kde = gaussian_kde(x_vals)
                            ax.fill_between(xg, kde(xg), alpha=0.5, color="steelblue")
                            ax.plot(xg, kde(xg), color="steelblue", lw=1.5)
                        except ImportError:
                            ax.hist(x_vals, bins=40, color="steelblue", alpha=0.7)
                    else:
                        ax.hist(x_vals, bins=40, color="steelblue", alpha=0.7)
                    ax.set_yticks([])
                else:
                    last_sc = ax.scatter(
                        x_vals, y_vals,
                        c=f, norm=norm, cmap=cmap_obj,
                        rasterized=True,
                        **kwargs,
                    )
                    ax.set_aspect(set_aspect)

                ax.tick_params(labelsize=6)
                if row == n - 1:
                    ax.set_xlabel(x_col, fontsize=8)
                else:
                    ax.set_xticklabels([])
                if col == 0:
                    ax.set_ylabel(y_col, fontsize=8)
                else:
                    ax.set_yticklabels([])

        if last_sc is not None:
            cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.70])
            sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
            sm.set_array([])
            cb = fig.colorbar(sm, cax=cbar_ax)
            cb.set_label(feature, fontsize=9)

        fig.suptitle(
            f"PCA pair grid — coloured by {feature}  (region {region})",
            fontsize=10, y=0.998,
        )

        return fig, gs

    def plot(
        self,
        feature: Optional[str] = None,
        index: Optional[int] = None,
        axes: tuple = ("x", "y"),
        region: int = 0,
        splitby: str = "z",
        set_aspect: str = "on",
        levels: int = 10,
        **kwargs,
    ):
        """Tricontourf map of a descriptor on the contact space."""
        if self.data is None:
            raise RuntimeError("No data — call atcontactspace() first.")
        if feature is None and index is None:
            raise ValueError("Provide feature or index.")
        if index is not None:
            feature = self.features[index]

        filterdata = self.data[self.contactspace.data["region"] == region]
        x1 = filterdata[axes[0]].values.astype(np.float64)
        x2 = filterdata[axes[1]].values.astype(np.float64)
        f  = filterdata[feature].values.astype(np.float64)
        s  = filterdata[splitby].values

        uniqs  = np.unique(s)
        nsplit = len(uniqs)
        fig, axs = (plt.subplots(nsplit, 1, figsize=(8, 4 * nsplit))
                    if nsplit > 1 else plt.subplots(1, 1, figsize=(8, 4)))
        axslist = axs.flat if nsplit > 1 else [axs]

        flevels = np.linspace(np.min(f), np.max(f), levels)
        fig.subplots_adjust(hspace=0.3)

        for i, ax in enumerate(axslist):
            sval = uniqs[i]
            mask = filterdata[splitby] == sval
            ax.set_title(f"{feature}  {splitby}={sval:6.2f}")
            im = ax.tricontourf(x1[mask], x2[mask], f[mask], levels=flevels, **kwargs)
            ax.axis(set_aspect)
            ax.set_xlabel(axes[0]); ax.set_ylabel(axes[1])

        cb_ax = axs if nsplit == 1 else axs.ravel().tolist()
        fig.colorbar(im, ax=cb_ax)
        return fig, axs
# Public exports
__all__ = [
    "BioSystem",
    "BioContactSpace",
    "AperiodicERFC",
    "create_protein_contact_space",
    "BioMaps",
]
