"""Atom-centered descriptors built from a Chebyshev basis.

Radial batches evaluate all query/atom pairs together and stream polynomial
orders through a forward recurrence. NumPy and CuPy use the same calculation.
Angular descriptors evaluate neighbor pairs within each query environment.
"""

from __future__ import annotations

from abc import ABC
from typing import Callable, List, Optional

import numpy as np
import numpy.typing as npt
from numpy.polynomial.chebyshev import Chebyshev as cheb

from mapsy.symfunc.input import SymFuncModel
from mapsy.symfunc import SymmetryFunction
from mapsy.utils import cutoff as make_cutoff
from mapsy.utils.cutoff import get_array_module
# Chebyshev utilities
def wrapcheby(f: Callable, order: int, rcut: float) -> Callable:
    def wrapped(*args):
        return f(order, rcut, *args)
    return wrapped


def dual_basis_function(order: int, rcut: float, x: float):
    """Angular Chebyshev basis (argument in [0, rcut=pi])."""
    return cheb.basis(order)(2 * x / rcut - 1)


def basis_function(order: int, rcut: float, x: float):
    """Radial Chebyshev basis with orthogonality weight."""
    k = 1
    if order == 0:
        k = 2
    return (
        k
        / (2 * np.pi * np.sqrt(x / rcut - x ** 2 / rcut ** 2))
        / 4
        * rcut
        * cheb.basis(order)(2 * x / rcut - 1)
    )


def _angular_cheby_matrix(orders: npt.NDArray, theta: npt.NDArray) -> npt.NDArray:
    """Evaluate angular Chebyshev basis for all orders at angle values.

    Args:
        orders: (N_order,) int array
        theta:  (P,) pair angles in [0, pi]

    Returns:
        (N_order, P) array
    """
    z = 2.0 * theta / np.pi - 1.0     # theta in [0,pi] -> z in [-1, 1]
    result = np.zeros((len(orders), len(theta)), dtype=np.float64)
    for i, order in enumerate(orders):
        result[i] = cheb.basis(int(order))(z)
    return result
# GPU-native forward Chebyshev recurrence
def _chebyshev_recurrence(z, max_order: int, xp):
    """
    Evaluate T_0(z), T_1(z), ..., T_{max_order}(z) at every element of z.

    Uses the three-term recurrence:
        T_0(x) = 1
        T_1(x) = x
        T_n(x) = 2x * T_{n-1}(x) - T_{n-2}(x)

    This runs entirely on the device (xp), requiring only max_order Python
    iterations — each iteration is a single element-wise GPU operation over
    the full z array (shape (...)).

    Args:
        z:         (...) array — arguments pre-clipped to [-1+eps, 1-eps]
        max_order: int — highest order needed
        xp:        numpy or cupy

    Returns:
        T: (max_order+1, ...) array — T[n] = T_n(z) at all points
    """
    shape = z.shape
    n_levels = max_order + 1

    T = xp.empty((n_levels,) + shape, dtype=xp.float64)
    T[0] = xp.ones(shape, dtype=xp.float64)

    if max_order == 0:
        return T

    T[1] = z.astype(xp.float64)

    two_z = (2.0 * z).astype(xp.float64)   # compute once, reused
    for n in range(2, n_levels):
        T[n] = two_z * T[n - 1] - T[n - 2]

    return T   # (max_order+1, ...)
# Parser
class ACSFParser:
    def __init__(self, symfuncmodel: SymFuncModel) -> None:
        self.order         = np.array(symfuncmodel.order, dtype=np.int64)
        self.cutoff        = symfuncmodel.cutoff
        self.radius        = symfuncmodel.radius
        self.radial        = symfuncmodel.radial
        self.compositional = symfuncmodel.compositional
        self.structural    = symfuncmodel.structural

    def parse(self) -> list[SymmetryFunction]:
        return [
            ACSymmetryFunction(
                self.order,
                self.radius,
                self.cutoff,
                self.compositional,
                self.structural,
                self.radial,
            )
        ]
# ACSymmetryFunction
class ACSymmetryFunction(SymmetryFunction):

    order:         npt.NDArray[np.int64] = None
    radial:        bool = True
    cutoff:        float = 0.0
    fc:            Callable = None
    rcut:          float = 0.0
    compositional: bool = False
    structural:    bool = False
    weights:       npt.NDArray[np.float64] = None
    polynomia:     List[Callable] = []
    types:         list[str] = None
    itypes:        npt.NDArray[np.int64] = None

    def __init__(
        self,
        order: npt.NDArray[np.int64],
        radius: float,
        cutofftype: str,
        compositional: bool = False,
        structural: bool = False,
        radial: bool = True,
    ) -> None:
        super().__init__(kind=2, label="ACSF")
        self.order = order
        self.fc    = make_cutoff(cutofftype, radius)
        self.rcut  = float(radius)
        self.compositional = compositional
        self.structural    = structural
        self.radial        = radial

        if self.radial:
            if radius == 0:
                raise ValueError("ACSF: radius must be > 0 for radial mode")
            self.cutoff = float(radius)
        else:
            self.cutoff = float(np.pi)

        self.polynomia: List[Callable] = []
        self._generate_functions()
    # Basis function construction (for single-point fallback)
    def _generate_functions(self) -> None:
        function = basis_function if self.radial else dual_basis_function
        self.polynomia = [wrapcheby(function, int(o), self.cutoff) for o in self.order]
    # Setup
    def setup(self, atoms=None) -> None:
        elements = atoms.get_chemical_symbols()
        self.types   = sorted(list(dict.fromkeys(elements)))
        self.itypes  = np.array([self.types.index(e) for e in elements], dtype=np.int64)
        self.ntypes  = int(np.max(self.itypes)) + 1
        if self.compositional:
            self._init_weights()

    def _init_weights(self) -> None:
        n = self.ntypes // 2
        weights = list(range(-n, n + 1))
        if self.ntypes % 2 == 0:
            weights.remove(0)
        self.weights = np.array([weights[i] for i in self.itypes], dtype=np.float64)
    # ABC hooks
    @property
    def order(self) -> npt.NDArray[np.int64]:
        return self.__order

    @order.setter
    def order(self, order) -> None:
        for i in np.asarray(order, dtype=np.int64):
            if not 0 <= i < 1000:
                raise ValueError(f"ACSF order {i} out of range")
        self.__order = np.asarray(order, dtype=np.int64)

    def _check_angular(self) -> bool:
        return not self.radial

    def _check_atomic(self) -> bool:
        return True

    def _generate_keys(self) -> list[str]:
        keys = []
        typelabel = "R" if self.radial else "A"
        if self.structural:
            for o in self.order:
                keys.append(f"{self.label}_{typelabel}S_r{self.rcut}_{o:03d}")
        if self.compositional:
            for o in self.order:
                keys.append(f"{self.label}_{typelabel}C_r{self.rcut}_{o:03d}")
        self.__keys = keys
        return keys
    # Single-point compute (backward compatible)
    def _compute_values(
        self,
        distances: npt.NDArray,
        vectors: Optional[npt.NDArray] = None,
    ) -> npt.NDArray:
        if self.radial:
            return self._calculate_radial_single(distances)
        else:
            return self._calculate_angular_single(distances, vectors)

    def _calculate_radial_single(self, distances: npt.NDArray) -> npt.NDArray:
        """Single-point radial: calls the vectorized GPU function with N=1."""
        dist2d = distances[np.newaxis, :]          # (1, M)
        result = self._calculate_radial_batch_gpu(dist2d, np)  # (1, n_features)
        return result[0]

    def _calculate_angular_single(
        self, distances: npt.NDArray, vectors: npt.NDArray
    ) -> npt.NDArray:
        fci = self.fc(distances)
        in_cut = (fci > 1e-8) & (distances > 1e-10)
        idx = np.where(in_cut)[0]
        n   = len(idx)

        N_ord = len(self.order)
        struct_coef = np.zeros(N_ord) if self.structural    else None
        comp_coef   = np.zeros(N_ord) if self.compositional else None

        if n < 2:
            return self._pack(struct_coef, comp_coef)

        d   = distances[idx]
        v   = vectors[idx]
        fc  = fci[idx]

        J, K = np.triu_indices(n, k=1)
        d_j = d[J];    d_k = d[K]
        v_j = v[J];    v_k = v[K]
        fc_j = fc[J];  fc_k = fc[K]

        cos_t = np.clip(
            np.einsum('pq,pq->p', v_j, v_k) / (d_j * d_k), -1.0, 1.0
        )
        theta    = np.arccos(cos_t)
        fc_pair  = fc_j * fc_k
        basis    = _angular_cheby_matrix(self.order, theta)

        if self.structural:
            struct_coef = np.sum(basis * fc_pair[None, :], axis=1)
        if self.compositional:
            w   = self.weights
            w_j = w[idx][J]
            w_k = w[idx][K]
            comp_coef = np.sum(basis * fc_pair[None, :] * (w_j * w_k)[None, :], axis=1)

        return self._pack(struct_coef, comp_coef)

    def _pack(self, struct_coef, comp_coef) -> npt.NDArray:
        parts = []
        if self.structural:
            parts.append(struct_coef if struct_coef is not None else np.zeros(len(self.order)))
        if self.compositional:
            parts.append(comp_coef if comp_coef is not None else np.zeros(len(self.order)))
        if not parts:
            return np.zeros(0, dtype=np.float64)
        return np.concatenate(parts)
    # Batch compute — public API
    def values_batch(
        self,
        distances_batch,                         # (N, M) — numpy or cupy
        vectors_batch = None,                    # (N, M, 3) or None
        use_gpu: bool = False,
    ) -> npt.NDArray:                            # (N, n_features) — always numpy
        """
        Compute ACSF for a batch of N query points.

        Radial GPU path (activated when distances_batch is a CuPy array OR
        use_gpu=True):
            - forward Chebyshev recurrence evaluates T_0...T_max simultaneously
              across the full (N, M) grid in O(max_order) GPU kernel launches.
            - No Python loops over N or M.
            - Memory: O(N × M + N × n_orders), with only two polynomial states retained.

        Radial CPU fallback:
            - Vectorized NumPy over (N, M), streaming requested polynomial orders.

        Angular:
            - Vectorized pairs within each point, sequential over N.
        """
        if self.radial:
            # Detect CuPy input
            is_cupy = hasattr(distances_batch, 'get')

            if is_cupy or use_gpu:
                try:
                    import cupy as cp
                    xp = cp
                    if not is_cupy:
                        distances_batch = cp.asarray(distances_batch, dtype=cp.float64)
                    return self._calculate_radial_batch_gpu(distances_batch, xp)
                except ImportError:
                    pass
                except Exception as e:
                    import warnings
                    warnings.warn(f"[ACSF] GPU path failed ({e}), falling back to CPU.")
                    if hasattr(distances_batch, 'get'):
                        distances_batch = distances_batch.get()

            return self._calculate_radial_batch_cpu(
                np.asarray(distances_batch, dtype=np.float64)
            )

        else:
            # Angular: vectorise pairs per point, loop over N
            if hasattr(distances_batch, "get"):
                distances_batch = distances_batch.get()
            if vectors_batch is not None and hasattr(vectors_batch, "get"):
                vectors_batch = vectors_batch.get()
            N = distances_batch.shape[0]
            if N == 0:
                return np.empty((0, len(self.keys)), dtype=np.float64)
            results = []
            for i in range(N):
                r = self._calculate_angular_single(
                    np.asarray(distances_batch[i]),
                    np.asarray(vectors_batch[i]) if vectors_batch is not None else None,
                )
                results.append(r)
            return np.stack(results, axis=0)
    # Radial calculation on NumPy or CuPy arrays
    def _calculate_radial_batch_gpu(
        self,
        distances_batch,   # (N, M) — CuPy or NumPy array
        xp,                # cupy or numpy module
    ) -> npt.NDArray:      # (N, n_features) — always returned as numpy
        """
        Evaluate radial ACSF using a forward Chebyshev recurrence.

        All point/atom pairs are vectorized on xp (NumPy or CuPy). Orders
        are streamed with two polynomial states, avoiding an order-sized
        temporary tensor. Only the final feature matrix returns to CPU.
        """
        N, M = distances_batch.shape
        if bool(xp.any(distances_batch <= 1e-10)):
            raise ValueError("Radial ACSF is singular at an atom center (distance <= 1e-10 Angstrom)")
        fc = self.fc(distances_batch)
        x_norm = distances_batch / float(self.rcut)
        z = xp.clip(2.0 * x_norm - 1.0, -1.0, 1.0)
        supported = (x_norm > 0.0) & (x_norm < 1.0) & (fc != 0)
        denom = xp.sqrt(xp.where(supported, x_norm * (1.0 - x_norm), 1.0))
        fw = xp.where(supported, fc * float(self.rcut) / (8.0 * np.pi * denom), 0.0)
        order_list = [int(o) for o in self.order]
        struct = xp.empty((N, len(order_list)), dtype=xp.float64)
        comp = xp.empty_like(struct)
        w_atoms = xp.asarray(self.weights, dtype=xp.float64)
        previous = xp.ones_like(z)
        current = z
        for order in range(max(order_list) + 1):
            if order == 0:
                polynomial = previous
            elif order == 1:
                polynomial = current
            else:
                polynomial = 2.0 * z * current - previous
                previous, current = current, polynomial
            if order in order_list:
                contribution = fw * polynomial * (2.0 if order == 0 else 1.0)
                for column, requested in enumerate(order_list):
                    if requested == order:
                        if self.structural:
                            struct[:, column] = xp.sum(contribution, axis=1)
                        if self.compositional:
                            comp[:, column] = xp.sum(contribution * w_atoms, axis=1)
        parts = []
        if self.structural:
            parts.append(struct)
        if self.compositional:
            parts.append(comp)
        result = xp.concatenate(parts, axis=1) if parts else xp.empty((N, 0))
        return result.get() if hasattr(result, 'get') else np.asarray(result)
    # Radial CPU calculation; query/atom pairs are vectorized and orders are streamed
    def _calculate_radial_batch_cpu(
        self,
        distances_batch: npt.NDArray,   # (N, M) numpy
    ) -> npt.NDArray:                   # (N, n_features) numpy
        """
        Fully vectorized CPU radial ACSF — same algorithm as the GPU path
        but runs on NumPy. Eliminates the previous double Python loop over
        orders and points.
        """
        return self._calculate_radial_batch_gpu(distances_batch, np)
