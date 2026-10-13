"""CPU and GPU array helpers, device selection, and periodic distances.

NumPy and CuPy share the batched distance calculation. If a GPU operation fails,
the fallback decorator retries the calculation on the CPU.
"""

import numpy as np
import os
import warnings
from typing import Optional, Callable, Any
from functools import wraps
# Configuration
GPU_CONFIG = {
    'enabled': True,
    'min_atoms_for_gpu': 500,
    'min_points_for_gpu': 1000,
    'gpu_memory_fraction': 0.80,
    'cpu_fallback_on_oom': True,
}

BACKEND = {'xp': np}
GPU_AVAILABLE = False
CUPY_AVAILABLE = False
# Initialisation
def setup_gpu() -> None:
    """Try to initialise CuPy; set GPU_AVAILABLE accordingly."""
    global GPU_AVAILABLE, CUPY_AVAILABLE, BACKEND

    if os.environ.get('MAPSY_DISABLE_GPU', '0') == '1':
        GPU_CONFIG['enabled'] = False
        return

    try:
        import cupy as cp

        # Smoke test
        _t = cp.array([1.0, 2.0, 3.0])
        _ = _t.get()

        CUPY_AVAILABLE = True
        GPU_AVAILABLE = True
        BACKEND['xp'] = cp

        device = cp.cuda.Device()
        mem_info = device.mem_info  # (free, total)
        props = cp.cuda.runtime.getDeviceProperties(int(device))
        name = props.get('name', b'GPU')
        if isinstance(name, (bytes, bytearray)):
            name = name.decode(errors='ignore')
        total_gb = mem_info[1] / 1e9
        free_gb  = mem_info[0] / 1e9
        print(f"[MapSy GPU] CuPy {cp.__version__} | Device: {name} "
              f"| {free_gb:.1f}/{total_gb:.1f} GB free")

        mempool = cp.get_default_memory_pool()
        if mempool is not None:
            mempool.set_limit(size=int(mem_info[1] * GPU_CONFIG['gpu_memory_fraction']))

    except ImportError:
        pass
    except Exception as e:
        warnings.warn(f"[MapSy GPU] Initialisation failed: {e}")
# Backend helpers
def get_backend(use_gpu: Optional[bool] = None):
    """Return numpy or cupy depending on GPU availability and preference."""
    if use_gpu is False or not GPU_AVAILABLE or not GPU_CONFIG['enabled']:
        return np
    return BACKEND['xp']


def should_use_gpu(n_atoms: int, n_points: int) -> bool:
    """
    Decide whether GPU is worth using.

    We base the decision on the dominant compute cost:
        n_atoms × n_points  (distance matrix size)

    A rough threshold: if computing the full (n_points × n_atoms) matrix
    would be > 1 M elements, GPU wins.
    """
    if not GPU_AVAILABLE or not GPU_CONFIG['enabled']:
        return False
    work = n_atoms * n_points
    return work >= 1_000_000


def to_gpu(arr: np.ndarray) -> Any:
    """Transfer a NumPy array to the GPU (no-op if CuPy not available)."""
    if CUPY_AVAILABLE and GPU_AVAILABLE:
        import cupy as cp
        return cp.asarray(arr)
    return arr


def to_cpu(arr: Any) -> np.ndarray:
    """Transfer an array from GPU to CPU (no-op if already NumPy)."""
    if CUPY_AVAILABLE and hasattr(arr, 'get'):
        return arr.get()
    return np.asarray(arr)
# Retry failed GPU operations on the CPU
#
# Original broken behaviour: injected `use_gpu=False` as a kwarg on retry,
# but `atpoints()` never accepted that kwarg → TypeError on fallback.
#
# Fixed behaviour: sets a temporary flag on the Maps instance (self._force_cpu)
# and retries. Maps.atpoints() checks this flag before choosing the backend.
def gpu_fallback_on_error(func: Callable) -> Callable:
    """
    Decorator for Maps methods that may run on GPU.

    On an out-of-memory (or other GPU) error, clears GPU memory and
    retries the same call using the CPU path by setting self._force_cpu=True
    temporarily.
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not GPU_CONFIG['cpu_fallback_on_oom'] or not CUPY_AVAILABLE:
            return func(self, *args, **kwargs)
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            err = str(e).lower()
            if any(tok in err for tok in ('memory', 'oom', 'out of memory', 'cuda')):
                warnings.warn(
                    f"[MapSy GPU] OOM / CUDA error — falling back to CPU: {e}"
                )
                import cupy as cp
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
                # Signal CPU path
                prev = getattr(self, '_force_cpu', False)
                self._force_cpu = True
                try:
                    result = func(self, *args, **kwargs)
                finally:
                    self._force_cpu = prev
                return result
            raise
    return wrapper
# Core distance utility (used by Maps and BioMaps)
from functools import lru_cache
from itertools import product
from ase.geometry import complete_cell, minkowski_reduce


@lru_cache(maxsize=64)
def _prepare_mic(cell_values, pbc_values):
    """Reduce once per distinct cell; preserve its periodic lattice exactly."""
    cell_cpu = np.asarray(cell_values, dtype=float).reshape(3, 3)
    pbc_arr = np.asarray(pbc_values, dtype=bool)
    periodic = cell_cpu[pbc_arr]
    if np.linalg.matrix_rank(periodic) != len(periodic):
        raise ValueError("Periodic lattice vectors must be nonzero and independent")
    if np.linalg.matrix_rank(cell_cpu) < 3:
        cell_cpu = cell_cpu.copy()
        cell_cpu[~pbc_arr] = 0.0
    cell_cpu = complete_cell(cell_cpu)
    gram = cell_cpu @ cell_cpu.T
    lengths = np.sqrt(np.diag(gram))
    orthogonal = np.all(np.abs(gram - np.diag(np.diag(gram))) <=
                        1e-12 * lengths[:, None] * lengths[None, :])
    if not orthogonal:
        cell_cpu, _ = minkowski_reduce(cell_cpu, pbc=pbc_arr)
    inv_cell_cpu = np.linalg.inv(cell_cpu)
    shifts = np.asarray(list(product(*[(-1, 0, 1) if p else (0,)
                                       for p in pbc_arr])), dtype=float) @ cell_cpu
    return cell_cpu, inv_cell_cpu, shifts, orthogonal


def batch_distances_periodic(
    positions,
    atom_positions,
    cell,
    pbc,
    xp=np,
) -> tuple:
    """
    Vectorised (N_pos, N_atoms, 3) displacement vectors and (N_pos, N_atoms)
    distances with full minimum-image-convention (MIC) support.

    Works transparently with xp=numpy (CPU) or xp=cupy (GPU).

    Args:
        positions:      (N, 3)  query points (Cartesian, Angstroms)
        atom_positions: (M, 3)  atom positions
        cell:           (3, 3)  lattice vectors (rows), may be zero for aperiodic
        pbc:            (3,)    periodic boundary flags
        xp:             array module — numpy or cupy

    Returns:
        distances: (N, M)    distances in Angstroms
        vectors:   (N, M, 3) displacement vectors (atom - query_point)
    """
    # --- Always coerce inputs to the target device (xp) ---
    pos      = xp.asarray(positions,      dtype=xp.float64)   # (N, 3)
    atom_pos = xp.asarray(atom_positions, dtype=xp.float64)   # (M, 3)
    cell_xp  = xp.asarray(cell,           dtype=xp.float64)   # (3, 3)

    # Raw displacements: vectors[n, m] = atom_pos[m] - pos[n]
    vecs = atom_pos[xp.newaxis, :, :] - pos[:, xp.newaxis, :]  # (N, M, 3)

    # --- MIC checks must use plain NumPy (no GPU arrays) ---
    # Explicitly pull to CPU no matter what xp is.
    def _to_numpy(a):
        """Move any array (numpy or cupy) to a plain numpy array."""
        if hasattr(a, 'get'):          # cupy array
            return a.get()
        return np.asarray(a)

    pbc_arr  = np.asarray(pbc, dtype=bool)          # always numpy
    cell_cpu = _to_numpy(cell)                       # always numpy (3, 3)

    if np.any(pbc_arr):
        cell_cpu, inv_cell_cpu, shifts, orthogonal = _prepare_mic(
            tuple(np.asarray(cell_cpu, dtype=float).reshape(-1)), tuple(pbc_arr)
        )
        cell_xp = xp.asarray(cell_cpu, dtype=xp.float64)
        inv_cell = xp.asarray(inv_cell_cpu, dtype=xp.float64)
        vecs_frac = vecs @ inv_cell
        for dim in range(3):
            if pbc_arr[dim]:
                if orthogonal:
                    vecs_frac[:, :, dim] -= xp.round(vecs_frac[:, :, dim])
                else:
                    vecs_frac[:, :, dim] -= xp.floor(vecs_frac[:, :, dim])
        vecs = vecs_frac @ cell_xp
        if not orthogonal:
            base = vecs
            best = xp.sum(base * base, axis=-1)
            for shift in shifts:
                candidate = base + xp.asarray(shift, dtype=xp.float64)
                distance2 = xp.sum(candidate * candidate, axis=-1)
                improve = distance2 < best
                vecs = xp.where(improve[..., None], candidate, vecs)
                best = xp.minimum(best, distance2)

    # Distances: (N, M)
    dists = xp.sqrt(xp.sum(vecs * vecs, axis=-1))

    return dists, vecs
# Info helpers
def get_gpu_info() -> dict:
    if not GPU_AVAILABLE:
        return {"available": False}
    try:
        import cupy as cp
        device = cp.cuda.Device()
        mem_info = device.mem_info
        return {
            "available": True,
            "free_gb": mem_info[0] / 1e9,
            "total_gb": mem_info[1] / 1e9,
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


# Run setup on import
setup_gpu()
