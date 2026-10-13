"""
MapSy Benchmark Utility
=======================

Profiles CPU (vectorised) vs GPU performance as a function of the number
of contact-space points, for a given system and symmetry functions.

Usage
-----
    from mapsy.benchmark import run_crossover_benchmark, plot_crossover

    results = run_crossover_benchmark(
        system,
        symmetryfunctions,
        contactspace,
        point_counts=[100, 250, 500, 1000, 2000, 5000, 10000],
        batch_size=1000,
        n_repeats=3,
    )
    fig = plot_crossover(results)
    fig.savefig("crossover.png", dpi=150)

The benchmark:
  1. Draws random subsets of the contact space of increasing size.
  2. Times both the vectorised CPU path and (if GPU is available) the GPU path.
  3. Returns a DataFrame and a plot showing where GPU becomes faster.

Key insight
-----------
  batch_size does NOT control whether GPU is used — use_gpu does.
  batch_size controls how many points are processed per kernel launch.

  Too-small batch_size  → many small kernel launches, GPU underutilised.
  Too-large batch_size  → GPU OOM on large systems.

  This benchmark also sweeps batch_size (for GPU) to find the sweet spot
  for your hardware.
"""

from __future__ import annotations

import time
import warnings
from typing import List, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgs

from mapsy.data import System
from mapsy.boundaries import ContactSpace
from mapsy.symfunc import SymmetryFunction
from mapsy.utils.gpu_utils import GPU_AVAILABLE
# Core timer
def _time_atpoints(
    maps_obj,
    positions: npt.NDArray,
    n_repeats: int = 3,
) -> float:
    """
    Return the median wall-clock time (seconds) for maps_obj.atpoints(positions).
    Runs n_repeats times and returns the median to reduce noise.
    """
    if n_repeats < 1:
        raise ValueError("n_repeats must be positive")
    maps_obj.atpoints(positions)  # untimed warmup
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        maps_obj.atpoints(positions)
        if getattr(maps_obj, "last_backend", "cpu") == "gpu":
            import cupy as cp
            cp.cuda.get_current_stream().synchronize()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))
# Main benchmark
def run_crossover_benchmark(
    system: System,
    symmetryfunctions: List[SymmetryFunction],
    contactspace: ContactSpace,
    point_counts: Optional[List[int]] = None,
    batch_size: int = 1000,
    gpu_batch_sizes: Optional[List[int]] = None,
    n_repeats: int = 3,
    seed: int = 42,
    verbosity: int = 1,
) -> pd.DataFrame:
    """
    Profile CPU (vectorised) vs GPU wall-clock time as a function of
    the number of contact-space points being processed.

    Parameters
    ----------
    system              : MapSy System object
    symmetryfunctions   : list of SymmetryFunction objects
    contactspace        : ContactSpace (the full contact space)
    point_counts        : list of n_points to test.
                          Default: 8 logarithmically spaced values from
                          100 to len(contactspace.data).
    batch_size          : batch size used for the vectorised CPU path.
    gpu_batch_sizes     : list of batch sizes to test on GPU.
                          Default: [256, 512, 1000, 2000, 4000].
    n_repeats           : number of timing repetitions (median is kept).
    seed                : random seed for reproducible position sampling.
    verbosity           : 0 = silent, 1 = progress.

    Returns
    -------
    pd.DataFrame with columns:
        n_points, cpu_time, gpu_{bs}_time (one per gpu batch size),
        gpu_best_time, best_batch_size, speedup
    """
    from mapsy.maps import Maps

    rng = np.random.default_rng(seed)

    all_positions = contactspace.data[["x", "y", "z"]].values
    n_total       = len(all_positions)

    if point_counts is None:
        point_counts = [
            int(v) for v in np.unique(
                np.geomspace(100, n_total, num=8).astype(int)
            ) if v <= n_total
        ]

    if gpu_batch_sizes is None:
        gpu_batch_sizes = [256, 512, 1000, 2000, 4000]

    records = []

    # Build a CPU Maps object once (re-used for all point counts)
    cpu_maps = Maps(
        system, symmetryfunctions, contactspace,
        use_gpu=False, vectorized=True, batch_size=batch_size,
    )

    # Build GPU Maps objects (one per batch size)
    gpu_maps_dict = {}
    if GPU_AVAILABLE:
        for bs in gpu_batch_sizes:
            try:
                gpu_maps_dict[bs] = Maps(
                    system, symmetryfunctions, contactspace,
                    use_gpu=True, batch_size=bs,
                )
            except Exception as e:
                warnings.warn(f"[Benchmark] Could not init GPU maps (bs={bs}): {e}")
    else:
        if verbosity:
            print("[Benchmark] GPU not available — CPU-only benchmark.")

    for n_pts in point_counts:
        # Sample n_pts positions without replacement
        idx       = rng.choice(n_total, size=min(n_pts, n_total), replace=False)
        positions = all_positions[idx]

        if verbosity:
            print(f"  n_points={n_pts:6d}", end="", flush=True)

        # --- CPU timing ---
        t_cpu = _time_atpoints(cpu_maps, positions, n_repeats)
        if verbosity:
            print(f"  cpu={t_cpu:.3f}s", end="", flush=True)

        row = {"n_points": len(positions), "requested_n_points": n_pts, "cpu_time": t_cpu, "cpu_backend": cpu_maps.last_backend}

        # --- GPU timing (each batch size) ---
        best_gpu  = np.inf
        best_bs   = None

        for bs, gm in gpu_maps_dict.items():
            try:
                reference = cpu_maps.atpoints(positions).to_numpy(dtype=float)
                candidate = gm.atpoints(positions).to_numpy(dtype=float)
                if gm.last_backend != "gpu":
                    raise RuntimeError("Requested GPU execution fell back to CPU")
                if not np.allclose(reference, candidate, rtol=1e-10, atol=1e-12):
                    raise RuntimeError("GPU output fails numerical equivalence gate")
                t_gpu = _time_atpoints(gm, positions, n_repeats)
                if gm.last_backend != "gpu":
                    raise RuntimeError("GPU timed execution fell back to CPU")
                row[f"gpu_bs{bs}_backend"] = gm.last_backend
                row[f"gpu_bs{bs}_time"] = t_gpu
                if t_gpu < best_gpu:
                    best_gpu = t_gpu
                    best_bs  = bs
                if verbosity:
                    print(f"  gpu(bs={bs})={t_gpu:.3f}s", end="", flush=True)
            except Exception as e:
                warnings.warn(f"[Benchmark] GPU bs={bs} failed at n={n_pts}: {e}")
                row[f"gpu_bs{bs}_time"] = np.nan

        row["gpu_best_time"]   = best_gpu if best_gpu < np.inf else np.nan
        row["best_batch_size"] = best_bs
        row["speedup"]         = t_cpu / best_gpu if best_gpu < np.inf and best_gpu > 0 else np.nan

        if verbosity:
            if best_gpu < np.inf:
                marker = "GPU faster" if best_gpu < t_cpu else "CPU faster"
                print(f"  -> {marker} (x{row['speedup']:.2f})")
            else:
                print()

        records.append(row)

    return pd.DataFrame(records)
# Batch-size sweep (GPU only)
def run_batchsize_benchmark(
    system: System,
    symmetryfunctions: List[SymmetryFunction],
    contactspace: ContactSpace,
    n_points: int = 2000,
    batch_sizes: Optional[List[int]] = None,
    n_repeats: int = 5,
    seed: int = 42,
    verbosity: int = 1,
) -> pd.DataFrame:
    """
    Sweep GPU batch sizes at a fixed number of contact-space points to find
    the optimal batch size for your GPU.

    Returns a DataFrame with columns: batch_size, gpu_time.
    """
    from mapsy.maps import Maps

    if not GPU_AVAILABLE:
        raise RuntimeError("GPU not available.")

    if batch_sizes is None:
        batch_sizes = [64, 128, 256, 512, 1000, 2000, 4000, 8000]

    rng       = np.random.default_rng(seed)
    all_pos   = contactspace.data[["x", "y", "z"]].values
    positions = all_pos[rng.choice(len(all_pos), size=min(n_points, len(all_pos)), replace=False)]

    records = []
    for bs in batch_sizes:
        try:
            gm    = Maps(system, symmetryfunctions, contactspace, use_gpu=True, batch_size=bs)
            t_gpu = _time_atpoints(gm, positions, n_repeats)
            records.append({"batch_size": bs, "gpu_time": t_gpu})
            if verbosity:
                print(f"  batch_size={bs:5d}  gpu_time={t_gpu:.3f}s")
        except Exception as e:
            warnings.warn(f"[Benchmark] batch_size={bs} failed: {e}")
            records.append({"batch_size": bs, "gpu_time": np.nan})

    return pd.DataFrame(records)
# Plots
def plot_crossover(
    results: pd.DataFrame,
    title: str = "CPU vs GPU crossover",
    gpu_batch_sizes: Optional[List[int]] = None,
) -> plt.Figure:
    """
    Plot CPU time, GPU times (per batch size), and speedup ratio.

    Parameters
    ----------
    results         : DataFrame returned by run_crossover_benchmark()
    title           : figure title
    gpu_batch_sizes : list of GPU batch sizes that were benchmarked.
                      If None, auto-detected from column names.

    Returns
    -------
    matplotlib Figure
    """
    if gpu_batch_sizes is None:
        gpu_batch_sizes = [
            int(c.split("bs")[1].split("_")[0])
            for c in results.columns
            if c.startswith("gpu_bs") and c.endswith("_time")
        ]

    n_pts   = results["n_points"].values
    t_cpu   = results["cpu_time"].values
    speedup = results["speedup"].values if "speedup" in results.columns else None

    has_gpu = len(gpu_batch_sizes) > 0 and not results["gpu_best_time"].isna().all()

    fig = plt.figure(figsize=(13, 5))
    gs  = mgs.GridSpec(1, 3 if has_gpu else 2, figure=fig, wspace=0.35)

    # --- Panel 1: absolute times ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(n_pts, t_cpu, 'o-', color='steelblue', label='CPU (vectorised)', lw=2)

    if has_gpu:
        cmap   = plt.get_cmap("Reds")
        n_bs   = len(gpu_batch_sizes)
        for k, bs in enumerate(gpu_batch_sizes):
            col = f"gpu_bs{bs}_time"
            if col in results.columns:
                t_gpu = results[col].values
                color = cmap(0.4 + 0.5 * k / max(n_bs - 1, 1))
                ax1.plot(n_pts, t_gpu, 's--', color=color,
                         label=f'GPU (bs={bs})', lw=1.2, alpha=0.8)

        # Best GPU
        t_best = results["gpu_best_time"].values
        ax1.plot(n_pts, t_best, 'D-', color='crimson',
                 label='GPU (best bs)', lw=2.5, zorder=5)

        # Mark crossover
        crossover = _find_crossover(n_pts, t_cpu, t_best)
        if crossover is not None:
            ax1.axvline(crossover, color='gray', linestyle=':', lw=1.5,
                        label=f'Crossover ~{crossover:.0f} pts')

    ax1.set_xlabel("Number of contact-space points")
    ax1.set_ylabel("Wall-clock time (s)")
    ax1.set_title("Absolute timing")
    ax1.legend(fontsize=7, loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')

    # --- Panel 2: batch size sensitivity (GPU) ---
    ax2 = fig.add_subplot(gs[0, 1])
    if has_gpu and len(gpu_batch_sizes) > 1:
        for col in results.columns:
            if col.startswith("gpu_bs") and col.endswith("_time"):
                bs    = int(col.split("bs")[1].split("_")[0])
                t_gpu = results[col].values
                ax2.plot(n_pts, t_gpu, 'o-', label=f'bs={bs}', lw=1.5)
        ax2.set_xlabel("Number of contact-space points")
        ax2.set_ylabel("GPU time (s)")
        ax2.set_title("GPU batch-size sensitivity")
        ax2.legend(fontsize=7)
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
        ax2.set_yscale('log')
    else:
        ax2.text(0.5, 0.5, "GPU not available\nor only 1 batch size",
                 ha='center', va='center', transform=ax2.transAxes, fontsize=11)
        ax2.axis('off')

    # --- Panel 3: speedup ---
    if has_gpu and speedup is not None:
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(n_pts, speedup, 'D-', color='darkgreen', lw=2)
        ax3.axhline(1.0, color='gray', linestyle='--', lw=1.2, label='Break-even')
        ax3.fill_between(n_pts, speedup, 1.0,
                         where=(np.asarray(speedup) > 1),
                         alpha=0.15, color='green', label='GPU faster')
        ax3.fill_between(n_pts, speedup, 1.0,
                         where=(np.asarray(speedup) < 1),
                         alpha=0.15, color='red', label='CPU faster')
        crossover = _find_crossover(n_pts, t_cpu, results["gpu_best_time"].values)
        if crossover is not None:
            ax3.axvline(crossover, color='gray', linestyle=':', lw=1.5,
                        label=f'Crossover ~{crossover:.0f} pts')
        ax3.set_xlabel("Number of contact-space points")
        ax3.set_ylabel("Speedup  (CPU time / GPU time)")
        ax3.set_title("GPU speedup over CPU")
        ax3.legend(fontsize=7)
        ax3.grid(True, alpha=0.3)
        ax3.set_xscale('log')

    fig.suptitle(title, fontsize=13, y=1.01)
    return fig


def plot_batchsize_sweep(results: pd.DataFrame) -> plt.Figure:
    """Plot GPU time vs batch size from run_batchsize_benchmark()."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(results["batch_size"], results["gpu_time"], 'o-', color='crimson', lw=2)
    best_idx = results["gpu_time"].idxmin()
    best_bs  = results.loc[best_idx, "batch_size"]
    best_t   = results.loc[best_idx, "gpu_time"]
    ax.axvline(best_bs, color='gray', linestyle='--',
               label=f'Optimal bs = {best_bs}  ({best_t:.3f} s)')
    ax.set_xlabel("batch_size")
    ax.set_ylabel("GPU time (s)")
    ax.set_title("GPU time vs batch_size")
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
# Utilities
def _find_crossover(
    n_pts: npt.NDArray,
    t_cpu: npt.NDArray,
    t_gpu: npt.NDArray,
) -> Optional[float]:
    """
    Estimate where GPU becomes faster than CPU by linear interpolation
    between the two bracketing points around the sign change of (t_cpu - t_gpu).
    """
    diff = np.asarray(t_cpu, dtype=float) - np.asarray(t_gpu, dtype=float)
    valid = ~np.isnan(diff)
    diff  = diff[valid]
    pts   = np.asarray(n_pts, dtype=float)[valid]

    if len(diff) < 2:
        return None

    # Find sign change
    for i in range(len(diff) - 1):
        if diff[i] < 0 and diff[i + 1] >= 0:
            # CPU was faster, GPU becomes faster
            t = -diff[i] / (diff[i + 1] - diff[i])
            return float(pts[i] + t * (pts[i + 1] - pts[i]))
        if diff[i] >= 0 and diff[i + 1] < 0:
            t = diff[i] / (diff[i] - diff[i + 1])
            return float(pts[i] + t * (pts[i + 1] - pts[i]))
    return None


def recommend_settings(results):
    """Recommend only within measured sizes; handle either crossover direction."""
    rows = results.sort_values("n_points")
    cpu = rows["cpu_time"].to_numpy(dtype=float)
    gpu = rows.get("gpu_best_time", pd.Series(np.nan, index=rows.index)).to_numpy(dtype=float)
    points = rows["n_points"].to_numpy(dtype=float)
    valid = np.isfinite(cpu) & np.isfinite(gpu) & (cpu > 0) & (gpu > 0)
    speedup = cpu[valid] / gpu[valid]
    diff = cpu[valid] - gpu[valid]
    pts = points[valid]
    crossings = []
    for i in range(len(diff) - 1):
        if (diff[i] > 0) != (diff[i + 1] > 0):
            t = -diff[i] / (diff[i + 1] - diff[i])
            crossings.append({"n_points": float(pts[i] + t * (pts[i + 1] - pts[i])),
                              "gpu_faster_above": bool(diff[i + 1] > 0)})
    best_bs = None
    if len(speedup):
        row = rows.iloc[np.flatnonzero(valid)[int(np.argmax(speedup))]]
        candidates = {int(c.split("bs")[1].split("_")[0]): float(row[c])
                      for c in rows.columns if c.startswith("gpu_bs") and c.endswith("_time")
                      and np.isfinite(row[c]) and row[c] > 0}
        if candidates:
            best_bs = min(candidates, key=candidates.get)
    summaries = ["Recommendations apply only to measured sizes; extrapolation is unverified."]
    for point, ratio in zip(pts, speedup):
        summaries.append(f"n_points={point:g}: {'GPU' if ratio > 1 else 'CPU'} faster (CPU/GPU={ratio:.3g})")
    for crossing in crossings:
        direction = 'above' if crossing['gpu_faster_above'] else 'below'
        summaries.append(f"Local crossover near {crossing['n_points']:.0f}: GPU faster {direction} this crossing.")
    if not len(speedup):
        summaries.append("No verified GPU measurements; use_gpu=False.")
    summary = "\n".join(summaries)
    print(summary)
    upward = len(crossings) == 1 and crossings[0]['gpu_faster_above']
    return {"use_gpu_always": bool(len(speedup) and np.all(speedup > 1)),
            "use_gpu_threshold": int(crossings[0]['n_points']) if upward else None,
            "gpu_faster_above": crossings[0]['gpu_faster_above'] if len(crossings) == 1 else None,
            "crossovers": crossings, "best_gpu_batch_size": best_bs,
            "meaningful_speedup": bool(len(speedup) and np.max(speedup) > 1.05),
            "max_speedup": float(np.max(speedup)) if len(speedup) else None,
            "mean_speedup": float(np.mean(speedup)) if len(speedup) else None,
            "summary": summary}
