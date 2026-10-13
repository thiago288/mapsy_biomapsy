"""
Behler-Parrinello Symmetry Functions (BPSF)

Optimizations matching what was done to ACSF:

  Phase B — True vectorization:
    G1: fc @ one_hot  replaces Python loop over ntypes
    G2: einsum('nme,mt->net') replaces Python loop over ntypes
    G3: same pattern as G2
    G4/G5: outer (lambda, zeta) loops replaced with full tensor ops

  Phase C — GPU-native:
    values_batch() auto-detects CuPy input via hasattr(x, 'get'),
    identical to ACSF. G1/G2/G3 run end-to-end on CuPy with zero
    Python loops and a single .get() transfer at the end.
    G4/G5 remain CPU-vectorized (pair operations are hard to parallelize
    across N on GPU without ragged tensors).

  Bug fixes carried forward:
    - O(1) combo dict replaces O(N) list.index()
    - All per-instance state initialized in __init__, not class body
"""

from __future__ import annotations

import itertools
import numpy as np
import numpy.typing as npt
import warnings
from typing import Callable, Dict, List, Optional, Tuple

from mapsy.symfunc.input import SymFuncModel
from mapsy.symfunc import SymmetryFunction
from mapsy.utils import cutoff as make_cutoff
from mapsy.utils.cutoff import get_array_module
# Parser
class BPSFParser:
    def __init__(self, symfuncmodel: SymFuncModel) -> None:
        self.cutoff  = symfuncmodel.cutoff
        self.radius  = symfuncmodel.radius
        self.order   = np.array(symfuncmodel.order, dtype=np.int64)
        self.etas    = np.array(symfuncmodel.etas)
        self.rss     = np.array(symfuncmodel.rss)
        self.zetas   = np.array(symfuncmodel.zetas)
        self.lambdas = np.array(symfuncmodel.lambdas)
        self.kappas  = np.array(symfuncmodel.kappas)

    def parse(self) -> list[SymmetryFunction]:
        sfs = []
        for i in self.order:
            sfs.append(BPSymmetryFunction(
                int(i) + 1, self.radius, self.cutoff,
                self.etas, self.rss, self.lambdas,
                self.kappas, self.zetas,
            ))
        return sfs
# Radial batch functions — GPU-native (G1, G2, G3)
def G1_batch(
    fc: Callable,
    distances,          # (N, M) — numpy or cupy
    types: npt.NDArray, # (M,) int, always numpy
    ntypes: int,
):                      # (N, ntypes)
    """
    G1 = Σ_m fc(r_m)  partitioned by element type.

    Vectorized: fc (N,M) @ one_hot (M, ntypes) = (N, ntypes)
    No Python loop over ntypes.
    Runs on whichever device 'distances' lives on.
    """
    xp       = get_array_module(distances)
    types_xp = xp.asarray(types)                              # (M,)
    fc_vals  = fc(distances)                                  # (N, M)

    # one_hot: (M, ntypes) — types_xp[:, None] == arange[None, :]
    one_hot = (
        types_xp[:, None] == xp.arange(ntypes, dtype=types_xp.dtype)[None, :]
    ).astype(xp.float64)                                      # (M, ntypes)

    return fc_vals @ one_hot                                  # (N, ntypes)


def G2_batch(
    fc: Callable,
    distances,          # (N, M)
    etas: npt.NDArray,  # (E,)
    rss: npt.NDArray,   # (E,)
    types: npt.NDArray, # (M,) int
    ntypes: int,
):                      # (N, E*ntypes)
    """
    G2 = Σ_m exp(-η(r_m - r_s)²) × fc(r_m)  per (η, r_s, elem).

    Vectorized: no Python loop over ntypes or etas.
    weighted (N,M,E) contracted with one_hot (M,ntypes) via einsum.
    """
    xp       = get_array_module(distances)
    types_xp = xp.asarray(types)
    etas_xp  = xp.asarray(etas)
    rss_xp   = xp.asarray(rss)

    N, M = distances.shape
    E    = len(etas)

    fc_vals = fc(distances)                                   # (N, M)

    # Gaussian factor × cutoff: (N, M, E)
    diff_sq  = (distances[:, :, None] - rss_xp[None, None, :]) ** 2
    weighted = xp.exp(-etas_xp[None, None, :] * diff_sq) * fc_vals[:, :, None]

    # one_hot: (M, ntypes)
    one_hot = (
        types_xp[:, None] == xp.arange(ntypes, dtype=types_xp.dtype)[None, :]
    ).astype(xp.float64)

    # (N, M, E) contracted with (M, ntypes) → (N, E, ntypes)
    g2 = xp.einsum('nme,mt->net', weighted, one_hot)

    return g2.reshape(N, E * ntypes)                         # (N, E*ntypes)


def G3_batch(
    fc: Callable,
    distances,          # (N, M)
    kappas: npt.NDArray,# (K,)
    types: npt.NDArray, # (M,) int
    ntypes: int,
):                      # (N, K*ntypes)
    """
    G3 = Σ_m cos(κ r_m) × fc(r_m)  per (κ, elem).

    Same pattern as G2: no Python loop over ntypes or kappas.
    """
    xp        = get_array_module(distances)
    types_xp  = xp.asarray(types)
    kappas_xp = xp.asarray(kappas)

    N, M = distances.shape
    K    = len(kappas)

    fc_vals  = fc(distances)                                  # (N, M)
    cos_term = xp.cos(kappas_xp[None, None, :] * distances[:, :, None])
    weighted  = cos_term * fc_vals[:, :, None]               # (N, M, K)

    one_hot = (
        types_xp[:, None] == xp.arange(ntypes, dtype=types_xp.dtype)[None, :]
    ).astype(xp.float64)                                      # (M, ntypes)

    # (N, M, K) contracted with (M, ntypes) → (N, K, ntypes)
    g3 = xp.einsum('nmk,mt->nkt', weighted, one_hot)

    return g3.reshape(N, K * ntypes)                         # (N, K*ntypes)
# Angular functions — CPU vectorized pairs (G4, G5)
# These operate on one query point at a time.
# GPU is not used here: the pair tensor is ragged (variable n per point)
# and the (lambda, zeta) outer loops are short (typically 2-4 values each).
def _angular_pairs(fc, distances, vectors, types, combos_dict, ncombos, include_rjk):
    """
    Shared pair setup for G4 and G5.
    Returns: J, K, d_j, d_k, cos_theta, exp_arg, fc_prod, combo_onehot, exp_fac
    (all numpy arrays)
    """
    fc_vals = fc(distances)
    idx     = np.where((fc_vals > 1e-8) & (distances > 1e-10))[0]
    n       = len(idx)

    if n < 2:
        return None

    d      = distances[idx]; v = vectors[idx]
    fc_sub = fc_vals[idx];   t = types[idx]

    J, K  = np.triu_indices(n, k=1)
    d_j   = d[J]; d_k = d[K]
    v_j   = v[J]; v_k = v[K]
    fc_j  = fc_sub[J]; fc_k = fc_sub[K]

    dots      = np.einsum('pq,pq->p', v_j, v_k)
    cos_theta = np.clip(dots / (d_j * d_k + 1e-30), -1.0, 1.0)

    if include_rjk:
        v_jk  = v_j - v_k
        d_jk  = np.linalg.norm(v_jk, axis=1)
        fc_jk = fc(d_jk)
        exp_arg = d_j**2 + d_k**2 + d_jk**2
        fc_prod = fc_j * fc_k * fc_jk
    else:
        exp_arg = d_j**2 + d_k**2
        fc_prod = fc_j * fc_k

    t_j = t[J]; t_k = t[K]
    icombo = np.array(
        [combos_dict.get((min(a, b), max(a, b)), 0) for a, b in zip(t_j, t_k)],
        dtype=np.int64,
    )
    C = ncombos
    combo_onehot = np.zeros((len(J), C), dtype=np.float64)
    combo_onehot[np.arange(len(J)), icombo] = 1.0

    return cos_theta, exp_arg, fc_prod, combo_onehot


def _angular_kernel(
    cos_theta, exp_arg, fc_prod, combo_onehot,
    zetas, etas, lambdas, ncombos, xp=np,
):
    """
    Fully tensor-ops angular accumulation — runs on numpy or cupy.

    Inputs are all (P,) or (P,C) numpy arrays (built on CPU during pair
    selection). When xp=cp, they are transferred once to GPU here.

    Eliminates the (lambda, zeta) Python double loop:
      ang:     (L, P)      = 1 + λ cos_θ
      ang_pow: (Z, L, P)   = 2^(1-ζ) ang^ζ
      exp_fac: (E, P)      = exp(-η r²)
      weight:  (Z, E, L, P) product
      result:  (Z, E, L, C) via (Z*E*L, P) @ (P, C)  — cuBLAS DGEMM on GPU
    """
    Z, E, L, C = len(zetas), len(etas), len(lambdas), ncombos
    P          = len(exp_arg)

    # Move to device once
    cos_t   = xp.asarray(cos_theta,    dtype=xp.float64)  # (P,)
    exp_a   = xp.asarray(exp_arg,      dtype=xp.float64)  # (P,)
    fc_p    = xp.asarray(fc_prod,      dtype=xp.float64)  # (P,)
    c_oh    = xp.asarray(combo_onehot, dtype=xp.float64)  # (P, C)
    zetas_  = xp.asarray(zetas,        dtype=xp.float64)
    etas_   = xp.asarray(etas,         dtype=xp.float64)
    lams_   = xp.asarray(lambdas,      dtype=xp.float64)

    exp_fac     = xp.exp(-etas_[:, None] * exp_a[None, :])          # (E, P)
    ang         = 1.0 + lams_[:, None] * cos_t[None, :]             # (L, P)
    ang_clipped = xp.clip(ang, 0.0, None)[None, :, :]               # (1, L, P)
    scale       = (2.0 ** (1.0 - zetas_))[:, None, None]            # (Z, 1, 1)
    ang_pow     = scale * (ang_clipped ** zetas_[:, None, None])     # (Z, L, P)

    weight = (exp_fac[None, :, None, :] *
              ang_pow[:, None, :, :] *
              fc_p[None, None, None, :])                             # (Z, E, L, P)

    result = weight.reshape(Z * E * L, P) @ c_oh                    # (Z*E*L, C)
    out    = result.reshape(-1)
    return out.get() if hasattr(out, 'get') else np.asarray(out)


# ---- GPU batch kernel for G4/G5 (padded pairs across N points) ------------

def _angular_batch_gpu(
    fc, distances_batch, vectors_batch,
    zetas, etas, lambdas, types, combos_dict, ncombos,
    include_rjk, xp,
):
    """
    GPU-accelerated G4/G5 for a batch of N CS points.

    Strategy: pair selection is ragged (different P_i per point), so we:
      1. Find pairs for each point on CPU  — O(N) fast Python, not the bottleneck
      2. Pad to (N, max_pairs) arrays      — regular tensors, GPU-friendly
      3. Run the (Z, E, L, N, max_pairs) kernel on GPU (xp=cp)

    Per-point accumulation uses einsum over padded pairs, preserving the
    point index. Pair discovery remains on CPU; tensor reduction runs on xp.

    Padding uses zeros for fc_prod and cos_theta — they contribute 0 to the sum.
    """
    N = distances_batch.shape[0]
    Z, E, L, C = len(zetas), len(etas), len(lambdas), ncombos

    # -- Step 1: collect pairs for each point on CPU --
    all_cos    = []
    all_exparg = []
    all_fcprod = []
    all_combo  = []
    pair_counts = []

    for i in range(N):
        d = np.asarray(distances_batch[i])
        v = np.asarray(vectors_batch[i]) if vectors_batch is not None else None
        res = _angular_pairs(fc, d, v, types, combos_dict, ncombos, include_rjk)
        if res is None:
            pair_counts.append(0)
            all_cos.append(np.zeros(0))
            all_exparg.append(np.zeros(0))
            all_fcprod.append(np.zeros(0))
            all_combo.append(np.zeros((0, C)))
        else:
            cos_theta, exp_arg, fc_prod, combo_oh = res
            pair_counts.append(len(cos_theta))
            all_cos.append(cos_theta)
            all_exparg.append(exp_arg)
            all_fcprod.append(fc_prod)
            all_combo.append(combo_oh)

    max_P = max(pair_counts) if pair_counts else 1

    # -- Step 2: build padded (N, max_P) arrays on CPU --
    cos_pad   = np.zeros((N, max_P), dtype=np.float64)
    exp_pad   = np.zeros((N, max_P), dtype=np.float64)
    fc_pad    = np.zeros((N, max_P), dtype=np.float64)  # 0 → no contribution
    combo_pad = np.zeros((N, max_P, C), dtype=np.float64)

    for i, P_i in enumerate(pair_counts):
        if P_i > 0:
            cos_pad[i, :P_i]      = all_cos[i]
            exp_pad[i, :P_i]      = all_exparg[i]
            fc_pad[i, :P_i]       = all_fcprod[i]
            combo_pad[i, :P_i, :] = all_combo[i]

    # -- Step 3: transfer to GPU and compute --
    cos_g   = xp.asarray(cos_pad,   dtype=xp.float64)  # (N, max_P)
    exp_g   = xp.asarray(exp_pad,   dtype=xp.float64)  # (N, max_P)
    fc_g    = xp.asarray(fc_pad,    dtype=xp.float64)  # (N, max_P)
    combo_g = xp.asarray(combo_pad, dtype=xp.float64)  # (N, max_P, C)
    zetas_  = xp.asarray(zetas,     dtype=xp.float64)
    etas_   = xp.asarray(etas,      dtype=xp.float64)
    lams_   = xp.asarray(lambdas,   dtype=xp.float64)

    # exp_fac: (E, N, max_P)
    exp_fac = xp.exp(-etas_[:, None, None] * exp_g[None, :, :])

    # ang: (L, N, max_P)
    ang     = 1.0 + lams_[:, None, None] * cos_g[None, :, :]
    scale   = (2.0 ** (1.0 - zetas_))[:, None, None, None]   # (Z,1,1,1)
    ang_pow = scale * xp.clip(ang, 0.0, None)[None, :, :, :] ** zetas_[:, None, None, None]
    # ang_pow: (Z, L, N, max_P)

    # weight: (Z, E, L, N, max_P)
    weight = (exp_fac[None, :, None, :, :] *
              ang_pow[:, None, :, :, :] *
              fc_g[None, None, None, :, :])

    # Per-point pair reduction; do not compute a discarded cross-point matmul.
    w_npc  = weight.reshape(Z * E * L, N, max_P)          # (Z*E*L, N, max_P)
    result = xp.einsum('znp,npc->znc', w_npc, combo_g)    # (Z*E*L, N, C)
    result = result.reshape(Z, E, L, N, C)
    # Transpose to (N, Z, E, L, C) → (N, Z*E*L*C)
    out    = result.transpose(3, 0, 1, 2, 4).reshape(N, -1)

    return out.get() if hasattr(out, 'get') else np.asarray(out)


def G4_vectorized(
    fc, distances, vectors, zetas, etas, lambdas, types, combos_dict, ncombos,
):
    """G4 — includes r_jk cutoff."""
    Z, E, L, C = len(zetas), len(etas), len(lambdas), ncombos
    g = np.zeros(Z * E * L * C, dtype=np.float64)
    pairs = _angular_pairs(fc, distances, vectors, types, combos_dict, ncombos,
                           include_rjk=True)
    if pairs is None:
        return g
    return _angular_kernel(*pairs, zetas, etas, lambdas, ncombos)


def G5_vectorized(
    fc, distances, vectors, zetas, etas, lambdas, types, combos_dict, ncombos,
):
    """G5 — no r_jk cutoff."""
    Z, E, L, C = len(zetas), len(etas), len(lambdas), ncombos
    g = np.zeros(Z * E * L * C, dtype=np.float64)
    pairs = _angular_pairs(fc, distances, vectors, types, combos_dict, ncombos,
                           include_rjk=False)
    if pairs is None:
        return g
    return _angular_kernel(*pairs, zetas, etas, lambdas, ncombos)
# BPSymmetryFunction
class BPSymmetryFunction(SymmetryFunction):
    """Behler-Parrinello symmetry function (G1–G5)."""

    def __init__(
        self,
        order: int,
        radius: float,
        cutofftype: str,
        etas=None, rss=None, lambdas=None, kappas=None, zetas=None,
    ) -> None:
        super().__init__(kind=1, label="BP-G")
        self.order = order
        self.fc    = make_cutoff(cutofftype, radius)
        self.rcut  = float(radius)

        # Defaults match the input model; scalar parameters become 1-D arrays.
        self.etas = np.atleast_1d(np.asarray([1.0] if etas is None else etas, dtype=float))
        self.rss = np.atleast_1d(np.asarray([0.0] if rss is None else rss, dtype=float))
        self.kappas = np.atleast_1d(np.asarray([1.0] if kappas is None else kappas, dtype=float))
        self.zetas = np.atleast_1d(np.asarray([1.0] if zetas is None else zetas, dtype=float))
        self.lambdas = np.atleast_1d(np.asarray([-1.0, 1.0] if lambdas is None else lambdas, dtype=float))
        for values in (self.etas, self.rss, self.kappas, self.zetas, self.lambdas):
            if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
                raise ValueError("BPSF parameters must be nonempty finite one-dimensional arrays")
        if self.order == 2 and len(self.etas) != len(self.rss):
            raise ValueError("G2 requires matching eta and rs parameter counts")
        if np.any(self.etas < 0) or np.any(self.rss < 0) or np.any(self.zetas < 1):
            raise ValueError("eta and rs must be nonnegative; zeta must be at least one")
        if np.any(np.abs(self.lambdas) > 1):
            raise ValueError("Angular lambda must lie between -1 and 1")

        self.types        = None
        self.itypes       = None
        self._ntypes      = 0
        self._ncombos     = 0
        self.icombos      = []
        self._combos_dict = {}
    # Setup
    def setup(self, atoms) -> None:
        elements    = atoms.get_chemical_symbols()
        self.types  = sorted(list(dict.fromkeys(elements)))
        self.itypes = np.array([self.types.index(e) for e in elements], dtype=np.int64)
        unique      = np.unique(self.itypes)
        self.icombos      = list(itertools.combinations_with_replacement(unique, 2))
        self._combos_dict = {combo: i for i, combo in enumerate(self.icombos)}
        self._ntypes      = len(self.types)
        self._ncombos     = len(self.icombos)
    # ABC hooks
    @property
    def order(self):
        return self.__order

    @order.setter
    def order(self, order):
        if not 1 <= order <= 5:
            raise ValueError(f"BPSF order must be 1–5, got {order}")
        self.__order = order

    def _check_angular(self):  return self.order >= 4
    def _check_atomic(self):   return True

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
        if len(set(keys)) != len(keys):
            raise ValueError("BPSF parameter formatting produces duplicate feature names; choose distinguishable parameters")
        self.__keys = keys
        return keys
    # Single-point fallback (backward compatible)
    def _compute_values(self, distances, vectors=None):
        if self.order == 1:
            return G1_batch(self.fc, distances[np.newaxis], self.itypes, self._ntypes)[0]
        elif self.order == 2:
            return G2_batch(self.fc, distances[np.newaxis], self.etas, self.rss,
                            self.itypes, self._ntypes)[0]
        elif self.order == 3:
            return G3_batch(self.fc, distances[np.newaxis], self.kappas,
                            self.itypes, self._ntypes)[0]
        elif self.order == 4:
            return G4_vectorized(self.fc, distances, vectors, self.zetas, self.etas,
                                  self.lambdas, self.itypes, self._combos_dict, self._ncombos)
        elif self.order == 5:
            return G5_vectorized(self.fc, distances, vectors, self.zetas, self.etas,
                                  self.lambdas, self.itypes, self._combos_dict, self._ncombos)
    # Batch compute — GPU-native for G1/G2/G3
    def values_batch(
        self,
        distances_batch,              # (N, M) — numpy or cupy
        vectors_batch=None,           # (N, M, 3) or None
        use_gpu: bool = False,
    ) -> npt.NDArray:                 # (N, n_features) always numpy
        """
        Compute BPSF for a batch of N query points.

        G1/G2/G3 (radial):
            Auto-detects CuPy input via hasattr(x, 'get') — identical
            pattern to ACSF. Runs the full computation on the device,
            transfers only the (N, n_features) result to CPU.
            No Python loops over ntypes, etas, or kappas.

        G4/G5 (angular):
            Vectorized pair ops per query point, sequential over N.
            GPU not used — pair tensors are ragged and the (Z,L) outer
            product is already fast as NumPy tensor ops.
        """
        if self.order in (1, 2, 3):
            # -- GPU path --
            is_cupy = hasattr(distances_batch, 'get')
            if is_cupy or use_gpu:
                try:
                    import cupy as cp
                    if not is_cupy:
                        distances_batch = cp.asarray(distances_batch, dtype=cp.float64)
                    if self.order == 1:
                        res = G1_batch(self.fc, distances_batch, self.itypes, self._ntypes)
                    elif self.order == 2:
                        res = G2_batch(self.fc, distances_batch, self.etas, self.rss,
                                       self.itypes, self._ntypes)
                    else:
                        res = G3_batch(self.fc, distances_batch, self.kappas,
                                       self.itypes, self._ntypes)
                    return res.get() if hasattr(res, 'get') else np.asarray(res)
                except Exception as e:
                    warnings.warn(f"[BPSF] GPU path failed ({e}), falling back to CPU.")
                    if hasattr(distances_batch, 'get'):
                        distances_batch = distances_batch.get()

            # -- CPU path --
            distances_batch = np.asarray(distances_batch, dtype=np.float64)
            if self.order == 1:
                return np.asarray(G1_batch(self.fc, distances_batch, self.itypes, self._ntypes))
            elif self.order == 2:
                return np.asarray(G2_batch(self.fc, distances_batch, self.etas, self.rss,
                                           self.itypes, self._ntypes))
            else:
                return np.asarray(G3_batch(self.fc, distances_batch, self.kappas,
                                           self.itypes, self._ntypes))

        elif self.order in (4, 5):
            # Angular: GPU batch kernel when CuPy input detected or use_gpu=True
            is_cupy = hasattr(distances_batch, 'get')
            if is_cupy or use_gpu:
                try:
                    import cupy as cp
                    xp = cp
                    dist_np = distances_batch.get() if is_cupy else np.asarray(distances_batch)
                    vec_np  = (vectors_batch.get()
                               if (vectors_batch is not None and hasattr(vectors_batch, 'get'))
                               else np.asarray(vectors_batch) if vectors_batch is not None
                               else None)
                    return _angular_batch_gpu(
                        self.fc, dist_np, vec_np,
                        self.zetas, self.etas, self.lambdas,
                        self.itypes, self._combos_dict, self._ncombos,
                        include_rjk=(self.order == 4), xp=cp,
                    )
                except Exception as e:
                    warnings.warn(f"[BPSF G{self.order}] GPU path failed ({e}), falling back to CPU.")

            # CPU fallback must explicitly move device arrays to host.
            if hasattr(distances_batch, "get"):
                distances_batch = distances_batch.get()
            if vectors_batch is not None and hasattr(vectors_batch, "get"):
                vectors_batch = vectors_batch.get()
            N  = distances_batch.shape[0]
            if N == 0:
                return np.empty((0, len(self.keys)), dtype=np.float64)
            fn = G4_vectorized if self.order == 4 else G5_vectorized
            results = []
            for i in range(N):
                results.append(fn(
                    self.fc,
                    np.asarray(distances_batch[i]),
                    np.asarray(vectors_batch[i]) if vectors_batch is not None else None,
                    self.zetas, self.etas, self.lambdas,
                    self.itypes, self._combos_dict, self._ncombos,
                ))
            return np.stack(results, axis=0)

        else:
            raise ValueError(f"Unsupported BPSF order: {self.order}")