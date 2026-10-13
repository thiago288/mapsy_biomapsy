"""
GPU Utilities for MapSy - Provides automatic CPU/GPU switching and memory management.

This module detects GPU availability and automatically selects the best backend
for computations based on system size and available hardware.
"""
import numpy as np
import os
import warnings
from typing import Optional, Callable, Any
from functools import wraps

# GPU Configuration
GPU_CONFIG = {
    'enabled': True,
    'min_atoms_for_gpu': 500,      # Minimum atoms to consider GPU
    'min_points_for_gpu': 1000,    # Minimum contact points to consider GPU
    'gpu_memory_fraction': 0.8,     # Fraction of GPU memory to use
    'cpu_fallback_on_oom': True,    # Fall back to CPU on out-of-memory
}

# Backend selection
BACKEND = {'np': np, 'xp': np}  # xp will be CuPy if available

# GPU availability flag
GPU_AVAILABLE = False
CUPY_AVAILABLE = False


def setup_gpu():
    """Initialize GPU support if available."""
    global GPU_AVAILABLE, CUPY_AVAILABLE, BACKEND
    
    # Check if GPU is disabled via environment variable
    if os.environ.get('MAPSY_DISABLE_GPU', '0') == '1':
        GPU_CONFIG['enabled'] = False
        print("[MapSy GPU] GPU support disabled via environment variable")
        return
    
    # Try to import CuPy
    try:
        import cupy as cp
        from cupy.cuda import memory_hooks
        
        # Test if GPU is actually accessible
        test_array = cp.array([1.0, 2.0, 3.0])
        _ = test_array.get()
        
        CUPY_AVAILABLE = True
        GPU_AVAILABLE = True
        BACKEND['xp'] = cp
        
        # Get GPU info
        device = cp.cuda.Device()
        mem_info = device.mem_info
        print(f"[MapSy GPU] CuPy {cp.__version__} initialized successfully")
        # CuPy 14+: Device() has no `.name`; use runtime device properties
        _props = cp.cuda.runtime.getDeviceProperties(int(device))
        _dname = _props.get('name', 'GPU')
        if isinstance(_dname, (bytes, bytearray)):
            _dname = _dname.decode(errors='ignore')
        print(f"[MapSy GPU] Device: {_dname}")
        print(f"[MapSy GPU] Memory: {mem_info[1] / 1e9:.2f} GB total, {mem_info[0] / 1e9:.2f} GB free")
        
        # Set memory pool limit
        if hasattr(cp, 'cuda'):
            mempool = cp.get_default_memory_pool()
            if mempool is not None:
                mem_limit = int(mem_info[1] * GPU_CONFIG['gpu_memory_fraction'])
                mempool.set_limit(size=mem_limit)
                
    except ImportError:
        print("[MapSy GPU] CuPy not available. Install with: pip install cupy-cuda11x")
        GPU_AVAILABLE = False
    except Exception as e:
        warnings.warn(f"[MapSy GPU] GPU initialization failed: {e}")
        GPU_AVAILABLE = False


def get_backend(use_gpu: Optional[bool] = None):
    """
    Get the appropriate backend (numpy or cupy).
    
    Args:
        use_gpu: Force GPU (True) or CPU (False). Auto-detect if None.
        
    Returns:
        Module: numpy or cupy module
    """
    if use_gpu is False or not GPU_AVAILABLE or not GPU_CONFIG['enabled']:
        return np
    return BACKEND['xp']


def should_use_gpu(n_atoms: int, n_points: int) -> bool:
    """
    Determine if GPU should be used based on system size.
    
    Args:
        n_atoms: Number of atoms in the system
        n_points: Number of contact space points
        
    Returns:
        bool: True if GPU should be used
    """
    if not GPU_AVAILABLE or not GPU_CONFIG['enabled']:
        return False
    
    return (n_atoms >= GPU_CONFIG['min_atoms_for_gpu'] or 
            n_points >= GPU_CONFIG['min_points_for_gpu'])


def to_gpu(arr: np.ndarray) -> Any:
    """Transfer array to GPU if using CuPy."""
    if CUPY_AVAILABLE and GPU_AVAILABLE:
        import cupy as cp
        return cp.asarray(arr)
    return arr


def to_cpu(arr: Any) -> np.ndarray:
    """Transfer array from GPU to CPU."""
    if CUPY_AVAILABLE and hasattr(arr, 'get'):
        return arr.get()
    return np.asarray(arr)


def gpu_fallback_on_error(func: Callable) -> Callable:
    """Decorator that falls back to CPU on GPU memory errors."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not GPU_CONFIG['cpu_fallback_on_oom'] or not CUPY_AVAILABLE:
            return func(*args, **kwargs)
        
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_str = str(e).lower()
            if 'memory' in error_str or 'oom' in error_str or 'out of memory' in error_str:
                warnings.warn(f"GPU out of memory, falling back to CPU: {e}")
                # Clear GPU memory
                import cupy as cp
                cp.get_default_memory_pool().free_all_blocks()
                # Retry with CPU
                kwargs['use_gpu'] = False
                return func(*args, **kwargs)
            raise
    return wrapper


class GPUBatchProcessor:
    """
    Process large arrays in GPU-friendly batches to avoid OOM errors.
    """
    
    def __init__(self, batch_size: int = 10000):
        self.batch_size = batch_size
    
    def process(self, data: np.ndarray, func: Callable, use_gpu: bool = True) -> np.ndarray:
        """
        Process data in batches.
        
        Args:
            data: Input data array
            func: Function to apply to each batch
            use_gpu: Whether to use GPU
            
        Returns:
            Processed data
        """
        xp = get_backend(use_gpu)
        n_total = len(data)
        
        if n_total <= self.batch_size:
            # Small enough to process at once
            if use_gpu and CUPY_AVAILABLE:
                data_gpu = to_gpu(data)
                result = func(data_gpu)
                return to_cpu(result)
            return func(data)
        
        # Process in batches
        results = []
        for i in range(0, n_total, self.batch_size):
            batch = data[i:i + self.batch_size]
            if use_gpu and CUPY_AVAILABLE:
                batch_gpu = to_gpu(batch)
                batch_result = func(batch_gpu)
                results.append(to_cpu(batch_result))
            else:
                results.append(func(batch))
        
        return np.concatenate(results, axis=0)


def get_gpu_info():
    """Get GPU information if available."""
    if not GPU_AVAILABLE:
        return "GPU not available"
    
    try:
        import cupy as cp
        device = cp.cuda.Device()
        mem_info = device.mem_info
        return {
            "device": device.name,
            "total_memory_gb": mem_info[1] / 1e9,
            "free_memory_gb": mem_info[0] / 1e9,
        }
    except Exception as e:
        return f"Error getting GPU info: {e}"


# Initialize GPU on module import
setup_gpu()
