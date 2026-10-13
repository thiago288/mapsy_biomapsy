"""
Multiprocessing utilities for MapSy

Uses ThreadPool to avoid pickling issues with symmetry functions.
"""
import numpy as np
import numpy.typing as npt
from typing import Callable, Optional, List, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial
import warnings


def get_optimal_workers(n_tasks: int) -> int:
    """Get optimal number of workers based on task count."""
    max_workers = mp.cpu_count()
    return min(max_workers, max(1, n_tasks // 4))


def full2chunk(inputs: npt.NDArray, n_chunks: Optional[int] = None, *, n_jobs: Optional[int] = None) -> List[npt.NDArray]:
    """
    Split input array into chunks for parallel processing.
    
    Args:
        inputs: Input array to split
        n_chunks: Number of chunks (default: optimal based on CPU count)
        
    Returns:
        List of array chunks
    """
    inputs = np.asarray(inputs)
    if inputs.dtype.kind not in "biufc":
        raise TypeError("inputs must be a numeric array")
    if n_jobs is not None:
        if n_chunks is not None:
            raise ValueError("Specify n_chunks or n_jobs, not both")
        n_chunks = n_jobs
    if n_chunks is not None and n_chunks < 1:
        raise ValueError("n_chunks must be positive")
    n_inputs = len(inputs)
    if not n_inputs:
        return []
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
    use_processes: bool = False,
    max_workers: Optional[int] = None
) -> List:
    """
    Parallel processing using thread or process pools.
    
    Default is ThreadPool to avoid pickling issues with symmetry functions.
    
    Args:
        function: Function to parallelize
        args: List of argument tuples
        use_processes: Use processes (True) or threads (False, default)
        max_workers: Maximum number of workers
        
    Returns:
        List of results
    """
    if len(args) == 1:
        # No need for parallelization
        return [function(args[0])]
    
    if max_workers is None:
        max_workers = get_optimal_workers(len(args))
    
    # Use ThreadPool by default (no pickling issues)
    Executor = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
    
    with Executor(max_workers=max_workers) as executor:
        futures = [executor.submit(function, arg) for arg in args]
        results = [f.result() for f in futures]
    
    return results
