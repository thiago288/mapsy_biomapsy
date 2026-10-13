"""
Optimized Multiprocessing for MapSy

Provides efficient parallel processing with support for:
- Process-based parallelism for CPU-bound tasks
- Thread-based parallelism for I/O-bound tasks
- GPU batch processing for large systems
- Automatic backend selection based on problem size
"""
import numpy as np
import numpy.typing as npt
from typing import Callable, Optional, List, Tuple, Any
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial
import warnings

from mapsy.utils.gpu_utils_old import (
    get_backend, should_use_gpu, to_gpu, to_cpu,
    GPU_AVAILABLE, CUPY_AVAILABLE, GPUBatchProcessor
)

# Configuration
MAX_WORKERS = mp.cpu_count()
CHUNK_SIZE_FACTOR = 4  # Number of chunks per worker


def get_optimal_workers(n_tasks: int) -> int:
    """Get optimal number of workers based on task count."""
    return min(MAX_WORKERS, max(1, n_tasks // CHUNK_SIZE_FACTOR))


def full2chunk(inputs: npt.NDArray, n_chunks: Optional[int] = None) -> List[npt.NDArray]:
    """
    Split input array into chunks for parallel processing.

    Args:
        inputs: Input array to split
        n_chunks: Number of chunks (default: optimal based on CPU count)

    Returns:
        List of array chunks
    """
    n_inputs = len(inputs)
    if n_chunks is None:
        n_chunks = get_optimal_workers(n_inputs)

    n_chunks = min(n_chunks, n_inputs)
    return np.array_split(inputs, n_chunks, axis=0)


def chunk2full(outputs: List[List], n: int) -> List[List]:
    """
    Combine chunked results back into full results.

    Args:
        outputs: List of chunked results
        n: Number of result arrays expected

    Returns:
        Combined results
    """
    results: List[List] = [[] for _ in range(n)]
    for chunk in outputs:
        for i in range(n):
            if i < len(chunk):
                results[i].extend(chunk[i])
    return results


def multiproc(
    function: Callable,
    args: List[Tuple],
    use_processes: bool = True,
    max_workers: Optional[int] = None
) -> List:
    """
    Parallel processing using process or thread pools.

    Args:
        function: Function to parallelize
        args: List of argument tuples
        use_processes: Use processes (True) or threads (False)
        max_workers: Maximum number of workers

    Returns:
        List of results
    """
    if len(args) == 1:
        # No need for parallelization
        return [function(args[0])]

    if max_workers is None:
        max_workers = get_optimal_workers(len(args))

    Executor = ProcessPoolExecutor if use_processes else ThreadPoolExecutor

    with Executor(max_workers=max_workers) as executor:
        futures = [executor.submit(function, arg) for arg in args]
        results = [f.result() for f in futures]

    return results


def vectorized_multiproc(
    function: Callable,
    data: npt.NDArray,
    chunk_size: Optional[int] = None,
    use_gpu: Optional[bool] = None,
    n_atoms: int = 0,
    n_points: int = 0
) -> npt.NDArray:
    """
    Vectorized parallel processing with automatic GPU/CPU selection.

    This function automatically selects the best backend:
    - GPU for large systems (if available)
    - Vectorized NumPy for medium systems
    - Parallel CPU for small systems with many chunks

    Args:
        function: Vectorized function to apply
        data: Input data array
        chunk_size: Size of chunks for processing
        use_gpu: Force GPU/CPU usage
        n_atoms: Number of atoms (for GPU decision)
        n_points: Number of points (for GPU decision)

    Returns:
        Processed data
    """
    # Auto-detect GPU usage
    if use_gpu is None:
        use_gpu = should_use_gpu(n_atoms, n_points)

    # Use GPU batch processor for large arrays
    if use_gpu and GPU_AVAILABLE:
        processor = GPUBatchProcessor(batch_size=chunk_size or 5000)
        return processor.process(data, function, use_gpu=True)

    # Use vectorized NumPy for CPU
    return function(data)


def parallel_map(
    func: Callable,
    items: List[Any],
    max_workers: Optional[int] = None,
    use_processes: bool = True,
    progress_callback: Optional[Callable] = None
) -> List[Any]:
    """
    Map function over items in parallel with optional progress tracking.

    Args:
        func: Function to apply
        items: List of items to process
        max_workers: Maximum number of workers
        use_processes: Use processes vs threads
        progress_callback: Optional callback(current, total)

    Returns:
        List of results in original order
    """
    if len(items) == 1:
        return [func(items[0])]

    if max_workers is None:
        max_workers = get_optimal_workers(len(items))

    results = [None] * len(items)

    Executor = ProcessPoolExecutor if use_processes else ThreadPoolExecutor

    with Executor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(func, item): i
            for i, item in enumerate(items)
        }

        completed = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                warnings.warn(f"Task {idx} failed: {e}")
                raise

            completed += 1
            if progress_callback:
                progress_callback(completed, len(items))

    return results


class BatchProcessor:
    """
    Efficient batch processor for symmetry function calculations.
    Handles both CPU and GPU backends with memory management.
    """

    def __init__(
        self,
        batch_size: int = 1000,
        use_gpu: Optional[bool] = None,
        n_atoms: int = 0
    ):
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.n_atoms = n_atoms
        self.xp = get_backend(use_gpu)

        # Adjust batch size for GPU
        if use_gpu and GPU_AVAILABLE and n_atoms > 1000:
            # Smaller batches for large systems on GPU to avoid OOM
            self.batch_size = min(batch_size, 500)

    def process_batches(
        self,
        positions: npt.NDArray,
        compute_func: Callable,
        *args,
        **kwargs
    ) -> List:
        """
        Process positions in batches.

        Args:
            positions: Array of positions to process
            compute_func: Function(positions_batch, *args, **kwargs) -> results
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Combined results from all batches
        """
        n_positions = len(positions)
        all_results = []

        for i in range(0, n_positions, self.batch_size):
            batch = positions[i:i + self.batch_size]

            # Transfer to GPU if needed
            if self.use_gpu and CUPY_AVAILABLE:
                batch = to_gpu(batch)
                args = tuple(to_gpu(arg) if isinstance(arg, np.ndarray) else arg
                           for arg in args)

            # Compute
            batch_results = compute_func(batch, *args, **kwargs)

            # Transfer back to CPU
            if self.use_gpu and CUPY_AVAILABLE:
                batch_results = [[to_cpu(v) for v in r] if isinstance(r, list)
                                else to_cpu(r) for r in batch_results]

            all_results.append(batch_results)

            # Clear GPU memory periodically
            if self.use_gpu and CUPY_AVAILABLE and i % (self.batch_size * 10) == 0:
                import cupy as cp
                cp.get_default_memory_pool().free_all_blocks()

        # Combine results
        return self._combine_results(all_results)

    def _combine_results(self, batch_results: List) -> List:
        """Combine results from multiple batches."""
        if not batch_results:
            return []

        n_outputs = len(batch_results[0])
        combined = [[] for _ in range(n_outputs)]

        for batch in batch_results:
            for i in range(n_outputs):
                combined[i].extend(batch[i])

        return combined


def optimized_compute(
    positions: npt.NDArray,
    compute_func: Callable,
    system_atoms,
    symmetry_functions: List,
    use_gpu: Optional[bool] = None,
    batch_size: int = 1000
) -> List:
    """
    Optimized computation of symmetry functions with automatic backend selection.

    Args:
        positions: Contact space positions
        compute_func: Core computation function
        system_atoms: ASE Atoms object
        symmetry_functions: List of symmetry functions
        use_gpu: Force GPU/CPU usage
        batch_size: Batch size for processing

    Returns:
        Computed symmetry function values
    """
    n_atoms = len(system_atoms)
    n_points = len(positions)

    # Auto-select GPU
    if use_gpu is None:
        use_gpu = should_use_gpu(n_atoms, n_points)

    # Create batch processor
    processor = BatchProcessor(
        batch_size=batch_size,
        use_gpu=use_gpu,
        n_atoms=n_atoms
    )

    # Process in batches
    return processor.process_batches(positions, compute_func, system_atoms, symmetry_functions)
