"""
Behler-Parrinello Symmetry Functions (BPSF)

Key improvements over earlier implementation:
- G1/G2/G3: true batch vectorisation over N query points
  using (N, M) distance matrix — no per-point Python loop.
- G4/G5: vectorised over atom pairs (upper-triangle einsum)
  for each query point; O(N_pairs) NumPy ops replace O(N²) Python loop.
- combo lookup: O(1) dict replaces O(N_combos) list.index() (Bug 8 fix).
- GPU path: G1/G2/G3 run entirely on CuPy; G4/G5 fall back to vectorised CPU.
- values_batch() public API: (N, M) distances → (N, n_features) result.
"""

from __future__ import annotations

import itertools
import numpy as np
import numpy.typing as npt
from typing import Callable, Dict, List, Optional, Tuple

from mapsy.symfunc.input import SymFuncModel
from mapsy.symfunc import SymmetryFunction
from mapsy.utils import cutoff as make_cutoff
from mapsy.utils.cutoff import get_array_module
# Parser
class BPSFParser:
    def __init__(self, symfuncmodel: SymFuncModel) -> None:
        self.cutoff = symfuncmodel.cutoff
        self.radius = symfuncmodel.radius
        self.order   = np.array(symfuncmodel.order, dtype=np.int64)
        self.etas    = np.array(symfuncmodel.etas)
        self.rss     = np.array(symfuncmodel.rss)
        self.zetas   = np.array(symfuncmodel.zetas)
        self.lambdas = np.array(symfuncmodel.lambdas)
        self.kappas  = np.array(symfuncmodel.kappas)

    def parse(self) -> list[SymmetryFunction]:
        sfs = []
        for i in self.order:
            sfs.append(
                BPSymmetryFunction(
                    int(i) + 1,
                    self.radius,
                    self.cutoff,
                    self.etas,
                    self.rss,
                    self.lambdas,
                    self.kappas,
                    self.zetas,
                )
            )
        return sfs
# Vectorised batch functions (CPU — NumPy or CuPy via xp)
def _fc_batch(fc: Callable, distances: npt.NDArray) -> npt.NDArray:
    """Apply cutoff function to (N, M) or (M,) distance array."""
    return fc(distances)


# ---- G1 -------------------------------------------------------------------

def G1_batch(
    fc: Callable,
    distances: npt.NDArray,   # (N, M)
    types: npt.NDArray,       # (M,) int
    ntypes: int,
) -> npt.NDArray:             # (N, ntypes)
    xp = get_array_module(distances)
    types_xp = xp.asarray(types)
    N = distances.shape[0]
    fc_vals = _fc_batch(fc, distances)           # (N, M)
    g1 = xp.zeros((N, ntypes), dtype=xp.float64)
    for itype in range(ntypes):
        mask = types_xp == itype                 # (M,) bool
        if xp.any(mask):
            g1[:, itype] = xp.sum(fc_vals[:, mask], axis=1)
    return g1                                    # (N, ntypes)


# ---- G2 -------------------------------------------------------------------

def G2_batch(
    fc: Callable,
    distances: npt.NDArray,   # (N, M)
    etas: npt.NDArray,        # (E,)
    rss: npt.NDArray,         # (E,)
    types: npt.NDArray,       # (M,) int
    ntypes: int,
) -> npt.NDArray:             # (N, E*ntypes)
    xp = get_array_module(distances)
    types_xp = xp.asarray(types)
    etas_xp  = xp.asarray(etas)
    rss_xp   = xp.asarray(rss)

    N, M = distances.shape
    E = len(etas)

    fc_vals = _fc_batch(fc, distances)           # (N, M)

    # (N, M, E): Gaussian factor × cutoff
    diff_sq  = (distances[:, :, None] - rss_xp[None, None, :]) ** 2   # (N, M, E)
    weighted = xp.exp(-etas_xp[None, None, :] * diff_sq) * fc_vals[:, :, None]

    g2 = xp.zeros((N, E, ntypes), dtype=xp.float64)
    for itype in range(ntypes):
        mask = types_xp == itype                 # (M,)
        if xp.any(mask):
            g2[:, :, itype] = xp.sum(weighted[:, mask, :], axis=1)

    return g2.reshape(N, E * ntypes)


# ---- G3 -------------------------------------------------------------------

def G3_batch(
    fc: Callable,
    distances: npt.NDArray,   # (N, M)
    kappas: npt.NDArray,      # (K,)
    types: npt.NDArray,       # (M,) int
    ntypes: int,
) -> npt.NDArray:             # (N, K*ntypes)
    xp = get_array_module(distances)
    types_xp  = xp.asarray(types)
    kappas_xp = xp.asarray(kappas)

    N, M = distances.shape
    K = len(kappas)

    fc_vals = _fc_batch(fc, distances)           # (N, M)

    # (N, M, K): cosine term × cutoff
    cos_term = xp.cos(kappas_xp[None, None, :] * distances[:, :, None])  # (N, M, K)
    weighted  = cos_term * fc_vals[:, :, None]

    g3 = xp.zeros((N, K, ntypes), dtype=xp.float64)
    for itype in range(ntypes):
        mask = types_xp == itype
        if xp.any(mask):
            g3[:, :, itype] = xp.sum(weighted[:, mask, :], axis=1)

    return g3.reshape(N, K * ntypes)


# ---- G4 (vectorised pairs within a single query point) --------------------

def G4_vectorized(
    fc: Callable,
    distances: npt.NDArray,    # (M,)
    vectors: npt.NDArray,      # (M, 3)
    zetas: npt.NDArray,        # (Z,)
    etas: npt.NDArray,         # (E,)
    lambdas: npt.NDArray,      # (L,)
    types: npt.NDArray,        # (M,) int
    combos_dict: Dict[Tuple[int,int], int],
    ncombos: int,
) -> npt.NDArray:              # (Z*E*L*ncombos,)
    """
    Vectorised G4 for one query point.

    Replaces the O(N²×Z×E×L) pure-Python triple loop with:
        O(N_pairs) pair selection  +  (E, N_pairs) NumPy matmul per (Z,L).
    """
    fc_vals = fc(distances)              # (M,)
    in_cut  = fc_vals > 1e-8
    idx     = np.where(in_cut)[0]
    n       = len(idx)

    Z, E, L, C = len(zetas), len(etas), len(lambdas), ncombos
    g4 = np.zeros((Z, E, L, C), dtype=np.float64)

    if n < 2:
        return g4.reshape(-1)

    d      = distances[idx]              # (n,)
    v      = vectors[idx]                # (n, 3)
    fc_sub = fc_vals[idx]               # (n,)
    t      = types[idx]                  # (n,)

    # All pairs (j, k) with j < k via upper-triangle indices
    J, K = np.triu_indices(n, k=1)      # (P,) each

    d_j   = d[J];    d_k   = d[K]       # (P,)
    v_j   = v[J];    v_k   = v[K]       # (P, 3)
    fc_j  = fc_sub[J]; fc_k = fc_sub[K]

    # G4 uses the j–k distance (3-body)
    v_jk = v_j - v_k                    # (P, 3)
    d_jk = np.linalg.norm(v_jk, axis=1) # (P,)
    fc_jk = fc(d_jk)                    # (P,)

    # cos θ_{jik}
    dots      = np.einsum('pq,pq->p', v_j, v_k)             # (P,)
    cos_theta = np.clip(dots / (d_j * d_k), -1.0, 1.0)      # (P,)

    # r² sum for Gaussian exponent
    exp_arg = d_j**2 + d_k**2 + d_jk**2                     # (P,)

    # fc product per pair
    fc_prod = fc_j * fc_k * fc_jk                           # (P,)

    # Gaussian factor: (E, P)
    exp_fac = np.exp(-etas[:, None] * exp_arg[None, :])

    # combo index per pair — O(1) dict lookup
    t_j = t[J];  t_k = t[K]
    icombo = np.array(
        [combos_dict.get((min(a, b), max(a, b)), 0) for a, b in zip(t_j, t_k)],
        dtype=np.int64
    )                                                        # (P,)

    # One-hot for combo accumulation: (P, C)
    combo_onehot = np.zeros((len(J), C), dtype=np.float64)
    combo_onehot[np.arange(len(J)), icombo] = 1.0

    for il, lambd in enumerate(lambdas):
        ang = 1.0 + lambd * cos_theta                        # (P,)
        for iz, zeta in enumerate(zetas):
            ang_pow = (2.0 ** (1.0 - zeta)) * (np.clip(ang, 0, None) ** zeta)  # (P,)
            # Total per-pair weight: (E, P)
            weight  = exp_fac * (ang_pow * fc_prod)[None, :]
            # Accumulate into combos: (E, P) @ (P, C) → (E, C)
            g4[iz, :, il, :] += weight @ combo_onehot

    return g4.reshape(-1)


# ---- G5 (vectorised pairs — no rjk cutoff) --------------------------------

def G5_vectorized(
    fc: Callable,
    distances: npt.NDArray,    # (M,)
    vectors: npt.NDArray,      # (M, 3)
    zetas: npt.NDArray,
    etas: npt.NDArray,
    lambdas: npt.NDArray,
    types: npt.NDArray,
    combos_dict: Dict[Tuple[int,int], int],
    ncombos: int,
) -> npt.NDArray:
    """Vectorised G5 — same as G4 but without the rjk cutoff contribution."""
    fc_vals = fc(distances)
    in_cut  = fc_vals > 1e-8
    idx     = np.where(in_cut)[0]
    n       = len(idx)

    Z, E, L, C = len(zetas), len(etas), len(lambdas), ncombos
    g5 = np.zeros((Z, E, L, C), dtype=np.float64)

    if n < 2:
        return g5.reshape(-1)

    d      = distances[idx]
    v      = vectors[idx]
    fc_sub = fc_vals[idx]
    t      = types[idx]

    J, K = np.triu_indices(n, k=1)

    d_j   = d[J];    d_k   = d[K]
    v_j   = v[J];    v_k   = v[K]
    fc_j  = fc_sub[J]; fc_k = fc_sub[K]

    dots      = np.einsum('pq,pq->p', v_j, v_k)
    cos_theta = np.clip(dots / (d_j * d_k), -1.0, 1.0)

    exp_arg = d_j**2 + d_k**2                               # G5: no rjk
    fc_prod = fc_j * fc_k                                   # G5: no fc_jk

    exp_fac = np.exp(-etas[:, None] * exp_arg[None, :])

    t_j = t[J];  t_k = t[K]
    icombo = np.array(
        [combos_dict.get((min(a, b), max(a, b)), 0) for a, b in zip(t_j, t_k)],
        dtype=np.int64
    )

    combo_onehot = np.zeros((len(J), C), dtype=np.float64)
    combo_onehot[np.arange(len(J)), icombo] = 1.0

    for il, lambd in enumerate(lambdas):
        ang = 1.0 + lambd * cos_theta
        for iz, zeta in enumerate(zetas):
            ang_pow = (2.0 ** (1.0 - zeta)) * (np.clip(ang, 0, None) ** zeta)
            weight  = exp_fac * (ang_pow * fc_prod)[None, :]
            g5[iz, :, il, :] += weight @ combo_onehot

    return g5.reshape(-1)
# BPSymmetryFunction
class BPSymmetryFunction(SymmetryFunction):
    """Behler-Parrinello symmetry function (G1–G5)."""

    order:   int                           = 0
    rcut:    float                         = 0.0
    fc:      Callable                      = None
    etas:    npt.NDArray[np.float64]       = None
    rss:     npt.NDArray[np.float64]       = None
    lambdas: npt.NDArray[np.float64]       = None
    kappas:  npt.NDArray[np.float64]       = None
    zetas:   npt.NDArray[np.float64]       = None
    types:   list[str]                     = None
    itypes:  npt.NDArray[np.int64]         = None
    icombos: list                          = None
    _combos_dict: Dict[Tuple[int,int],int] = None

    def __init__(
        self,
        order: int,
        radius: float,
        cutofftype: str,
        etas: Optional[npt.NDArray[np.float64]] = None,
        rss: Optional[npt.NDArray[np.float64]] = None,
        lambdas: Optional[npt.NDArray[np.float64]] = None,
        kappas: Optional[npt.NDArray[np.float64]] = None,
        zetas: Optional[npt.NDArray[np.float64]] = None,
    ) -> None:
        super().__init__(kind=1, label="BP-G")
        self.order = order
        self.fc    = make_cutoff(cutofftype, radius)
        self.rcut  = radius

        if self.order == 2:
            self.etas = np.asarray(etas, dtype=np.float64)
            self.rss  = np.asarray(rss,  dtype=np.float64)
        elif self.order == 3:
            self.kappas = np.asarray(kappas, dtype=np.float64)
        elif self.order >= 4:
            self.etas    = np.asarray(etas,    dtype=np.float64)
            self.zetas   = np.asarray(zetas,   dtype=np.float64)
            self.lambdas = np.asarray(lambdas, dtype=np.float64)
    # Setup (called once before any compute, passing the atoms object)
    def setup(self, atoms) -> None:
        elements = atoms.get_chemical_symbols()
        self.types  = sorted(list(dict.fromkeys(elements)))
        self.itypes = np.array([self.types.index(e) for e in elements], dtype=np.int64)

        unique = np.unique(self.itypes)
        self.icombos = list(itertools.combinations_with_replacement(unique, 2))

        # O(1) combo lookup dict  — fix for Bug 8
        self._combos_dict: Dict[Tuple[int, int], int] = {
            combo: i for i, combo in enumerate(self.icombos)
        }
        self._ntypes = int(np.max(self.itypes)) + 1
        self._ncombos = len(self.icombos)
    # SymmetryFunction ABC hooks
    @property
    def order(self) -> int:
        return self.__order

    @order.setter
    def order(self, order: int) -> None:
        if not 1 <= order <= 5:
            raise ValueError(f"BPSF order must be 1–5, got {order}")
        self.__order = order

    def _check_angular(self) -> bool:
        return self.order >= 4

    def _check_atomic(self) -> bool:
        return True

    def _generate_keys(self) -> list[str]:
        keys = []
        if self.order == 1:
            for elem in self.types:
                keys.append(f"{self.label}1_{elem}_r{self.rcut}")
        elif self.order == 2:
            for eta, rs in zip(self.etas, self.rss):
                for elem in self.types:
                    keys.append(f"{self.label}2_{elem}_r{self.rcut}_eta{eta:.3g}_rs{rs:.3g}")
        elif self.order == 3:
            for kappa in self.kappas:
                for elem in self.types:
                    keys.append(f"{self.label}3_{elem}_r{self.rcut}_kappa{kappa}")
        elif self.order in (4, 5):
            for zeta in self.zetas:
                for eta in self.etas:
                    for lambd in self.lambdas:
                        for combo in self.icombos:
                            e0, e1 = self.types[combo[0]], self.types[combo[1]]
                            keys.append(
                                f"{self.label}{self.order}_{e0}_{e1}"
                                f"_r{self.rcut}_l{lambd}_z{zeta}_eta{eta:.2f}"
                            )
        self.__keys = keys
        return keys
    # Single-point compute (backward compatible — kept for fallback)
    def _compute_values(
        self,
        distances: npt.NDArray,
        vectors: Optional[npt.NDArray] = None,
    ) -> npt.NDArray:
        if self.order == 1:
            return G1_batch(self.fc, distances[np.newaxis], self.itypes, self._ntypes)[0]
        elif self.order == 2:
            return G2_batch(self.fc, distances[np.newaxis], self.etas, self.rss, self.itypes, self._ntypes)[0]
        elif self.order == 3:
            return G3_batch(self.fc, distances[np.newaxis], self.kappas, self.itypes, self._ntypes)[0]
        elif self.order == 4:
            return G4_vectorized(self.fc, distances, vectors, self.zetas, self.etas,
                                 self.lambdas, self.itypes, self._combos_dict, self._ncombos)
        elif self.order == 5:
            return G5_vectorized(self.fc, distances, vectors, self.zetas, self.etas,
                                 self.lambdas, self.itypes, self._combos_dict, self._ncombos)
    # Batch compute — NEW vectorised API
    def values_batch(
        self,
        distances_batch: npt.NDArray,              # (N, M)
        vectors_batch: Optional[npt.NDArray] = None,  # (N, M, 3) or None
        use_gpu: bool = False,
    ) -> npt.NDArray:                               # (N, n_features)
        """
        Compute BPSF for a batch of N query points simultaneously.

        For G1/G2/G3 (radial): fully vectorised over N via NumPy/CuPy.
        For G4/G5 (angular):   vectorised pairs within each point, sequential
                               over N (pair-level NumPy ops replace triple loop).

        Args:
            distances_batch: (N, M) distances to all M atoms
            vectors_batch:   (N, M, 3) displacement vectors (required for angular)
            use_gpu:         if True, attempt CuPy for radial SFs

        Returns:
            (N, n_features) float64 array
        """
        N = distances_batch.shape[0]

        if self.order == 1:
            xp = _choose_xp(use_gpu)
            dist = xp.asarray(distances_batch)
            res  = G1_batch(self.fc, dist, self.itypes, self._ntypes)
            return _to_numpy(res)                 # (N, ntypes)

        elif self.order == 2:
            xp = _choose_xp(use_gpu)
            dist = xp.asarray(distances_batch)
            res  = G2_batch(self.fc, dist, self.etas, self.rss, self.itypes, self._ntypes)
            return _to_numpy(res)                 # (N, E*ntypes)

        elif self.order == 3:
            xp = _choose_xp(use_gpu)
            dist = xp.asarray(distances_batch)
            res  = G3_batch(self.fc, dist, self.kappas, self.itypes, self._ntypes)
            return _to_numpy(res)                 # (N, K*ntypes)

        elif self.order in (4, 5):
            # Angular: must loop over points, but pair ops are vectorised
            fn = G4_vectorized if self.order == 4 else G5_vectorized
            results = []
            for i in range(N):
                r = fn(
                    self.fc,
                    np.asarray(distances_batch[i]),
                    np.asarray(vectors_batch[i]) if vectors_batch is not None else None,
                    self.zetas, self.etas, self.lambdas,
                    self.itypes, self._combos_dict, self._ncombos,
                )
                results.append(r)
            return np.stack(results, axis=0)      # (N, Z*E*L*C)

        else:
            raise ValueError(f"Unsupported BPSF order: {self.order}")
# Internal helpers
def _choose_xp(use_gpu: bool):
    """Return numpy or cupy based on use_gpu flag and availability."""
    if not use_gpu:
        return np
    try:
        from mapsy.utils.gpu_utils import GPU_AVAILABLE, BACKEND
        if GPU_AVAILABLE:
            return BACKEND['xp']
    except ImportError:
        pass
    return np


def _to_numpy(arr) -> np.ndarray:
    """Move array to CPU if it's on GPU."""
    try:
        import cupy as cp
        if isinstance(arr, cp.ndarray):
            return arr.get()
    except ImportError:
        pass
    return np.asarray(arr)
