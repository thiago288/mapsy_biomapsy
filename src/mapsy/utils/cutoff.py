"""
Cutoff Functions — backend-agnostic (NumPy and CuPy).

The cutoff callables returned here work transparently on both
numpy.ndarray (CPU) and cupy.ndarray (GPU) inputs because we detect
the array module at call time via `get_array_module`.

Public API (unchanged from original):
    cutoff(type, rcut) -> Callable
"""

import numpy as np
import numpy.typing as npt
from typing import Callable


def get_array_module(arr):
    """
    Return the array module (numpy or cupy) for the given array.

    This avoids a hard CuPy dependency: if CuPy is not installed,
    all arrays are numpy arrays and we always return numpy.
    """
    try:
        import cupy as cp
        if isinstance(arr, cp.ndarray):
            return cp
    except ImportError:
        pass
    return np
# Core cutoff functions
def cosfc(rcut: float, rij: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Cosine cutoff function.

    f_c(r) = 0.5 * (cos(π r / r_cut) + 1)   for r ≤ r_cut
             0                                  otherwise

    Works with both NumPy and CuPy arrays.
    """
    xp = get_array_module(rij)
    return xp.where(
        rij <= rcut,
        xp.float64(0.5) * (xp.cos(xp.float64(np.pi) * rij / xp.float64(rcut)) + xp.float64(1.0)),
        xp.float64(0.0),
    ).astype(xp.float64)


def tanhfc(rcut: float, rij: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Tanh cutoff function.

    f_c(r) = tanh³(1 - r / r_cut)   for r ≤ r_cut
             0                        otherwise

    Works with both NumPy and CuPy arrays.
    """
    xp = get_array_module(rij)
    return xp.where(
        rij <= rcut,
        xp.tanh(xp.float64(1.0) - rij / xp.float64(rcut)) ** xp.float64(3.0),
        xp.float64(0.0),
    ).astype(xp.float64)
# Wrappers
def wraprcut(f: Callable, rcut: float) -> Callable:
    """Bind rcut into the cutoff callable."""
    def wrapped(rij):
        return f(rcut, rij)
    wrapped.__name__ = f"{f.__name__}_rcut{rcut}"
    return wrapped


def cutoff(type: str, rcut: float) -> Callable:
    """
    Return a cutoff callable for the given type and cutoff radius.

    Args:
        type:  "cos" or "tanh"
        rcut:  cutoff radius in Angstroms

    Returns:
        Callable f(rij) -> array of same shape/device as rij
    """
    if not np.isfinite(rcut) or rcut <= 0:
        raise ValueError("Cutoff radius must be finite and positive")
    if type == "cos":
        fc = cosfc
    elif type == "tanh":
        fc = tanhfc
    else:
        raise ValueError(f"Unknown cutoff type {type!r}. Choose 'cos' or 'tanh'.")
    return wraprcut(fc, rcut)
