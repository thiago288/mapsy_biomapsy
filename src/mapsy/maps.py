"""MapSy computes local-environment descriptors for periodic structures.

Distances use the minimum-image convention. Orthogonal cells use fractional
wrapping; skew cells use a cached reduced lattice and nearby image candidates.
"""

from __future__ import annotations

from abc import ABC
from copy import deepcopy
from itertools import combinations
from typing import List, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.gridspec import GridSpec
import scipy.spatial.distance as distance
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score

from mapsy.utils.gpu_utils import (
    GPU_AVAILABLE,
    CUPY_AVAILABLE,
    should_use_gpu,
    gpu_fallback_on_error,
    batch_distances_periodic,
    to_cpu,
    get_backend,
)
from mapsy.utils import full2chunk, chunk2full, multiproc
from mapsy.io import Input, SystemParser, ContactSpaceGenerator
from mapsy.data import System, ScalarField
from mapsy.boundaries import ContactSpace
from mapsy.symfunc import SymmetryFunction, SymmetryFunctionsParser
# Internal helpers
def _has_batch_api(sf):
    return hasattr(sf, 'values_batch') and callable(getattr(sf, 'values_batch'))


def _sf_values_single(sf, dist, vec):
    if sf.angular:
        return sf.values(dist, vec)
    return sf.values(dist)
# Maps (ABC)
class Maps(ABC):
    """
    Compute symmetry-function descriptors on the contact space of a
    periodic crystalline system.

    Public API is identical to the original MapSy Maps class.
    Performance improvements (vectorised CPU, GPU) are internal only.
    """

    debug:     bool = False
    verbosity: int  = 0

    def __init__(
        self,
        system,
        symmetryfunctions,
        contactspace=None,
        batch_size=1000,
        use_gpu=None,
        vectorized=True,
        verbosity=0,
        debug=False,
    ):
        # per-instance mutable state
        self.symmetryfunctions = deepcopy(symmetryfunctions)
        self.features          = []
        self.data              = None
        self.centroids         = None
        self.best_clusters     = None
        self.cluster_screening = None
        self.cluster_centers   = None
        self.cluster_graph     = None
        self.cluster_edges     = None
        self.cluster_sizes     = None
        self.cluster_features  = []
        self.nclusters         = 0
        self.npca              = None
        self.random_state      = None

        self.debug     = debug
        self.verbosity = verbosity

        self.hascs        = contactspace is not None
        self.contactspace = contactspace
        self.system       = system
        self.nsfs         = len(self.symmetryfunctions)
        self.hasatomicsf  = any(sf.atomic for sf in self.symmetryfunctions)

        for sf in self.symmetryfunctions:
            if not getattr(sf, "atomic", True):
                raise NotImplementedError("Non-atomic symmetry functions are not supported")
            sf.setup(self.system.atoms)

        if np.any(self.system.atoms.pbc):
            import warnings
            from ase.geometry import minkowski_reduce
            reduced, _ = minkowski_reduce(self.system.atoms.cell, pbc=self.system.atoms.pbc)
            shortest = np.min(np.linalg.norm(np.asarray(reduced)[self.system.atoms.pbc], axis=1))
            if any(getattr(sf, "rcut", 0) > shortest / 2 for sf in self.symmetryfunctions):
                warnings.warn("Descriptor cutoff exceeds half the shortest periodic lattice vector; Maps uses one minimum image per atom, not an image-complete environment", UserWarning)
        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.vectorized = bool(vectorized)
        self._force_cpu = False

        if use_gpu is None:
            n_atoms  = len(system.atoms) if system.atoms is not None else 0
            n_points = len(contactspace.data) if contactspace is not None else 0
            self.use_gpu = should_use_gpu(n_atoms, n_points)
        else:
            self.use_gpu = bool(use_gpu)

        if self.use_gpu and self.verbosity > 0:
            n_pts = len(contactspace.data) if contactspace else 0
            print(f"[MapSy] GPU selected: {len(system.atoms)} atoms, {n_pts} CS points")

    def len(self):
        return len(self.symmetryfunctions)
    # Public compute API
    def atcontactspace(self):
        if not self.hascs:
            raise RuntimeError("Trying to use maps.compute() without contact space")
        positions     = self.contactspace.data[["x", "y", "z"]].values
        self.data     = self.atpoints(positions)
        self.features = self.data.columns.drop(['x', 'y', 'z'])
        return self.data

    @gpu_fallback_on_error
    def atpoints(self, positions):
        positions = np.asarray(positions, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 3 or not np.isfinite(positions).all():
            raise ValueError("positions must be a finite (N, 3) array")
        if len(positions) == 0:
            results = [np.empty((0, len(sf.keys))) for sf in self.symmetryfunctions]
            return self._results2df(positions, results)
        use_gpu_now = self.use_gpu and GPU_AVAILABLE and not self._force_cpu

        self.last_backend = "gpu" if use_gpu_now else "cpu"
        if use_gpu_now:
            results = self._process_gpu(positions)
        elif self.vectorized:
            results = self._process_vectorized(positions)
        else:
            args          = full2chunk(positions)
            chunk_results = multiproc(self._process_chunk, args)
            results       = chunk2full(chunk_results, self.nsfs)

        return self._results2df(positions, results)
    # Batched CPU calculation
    def _process_vectorized(self, positions):
        n_pts    = len(positions)
        atom_pos = self.system.atoms.positions
        cell     = np.asarray(self.system.atoms.cell)
        pbc      = np.asarray(self.system.atoms.pbc, dtype=bool)
        sf_results = [[] for _ in range(self.nsfs)]

        # Conservative 64 MiB target for dense temporary pair arrays.
        # One point may still exceed it for exceptionally large atom counts.
        batch_size = min(self.batch_size, max(1, (64 * 1024**2) // (max(1, len(self.system.atoms)) * 16 * 8)))
        for start in range(0, n_pts, batch_size):
            end   = min(start + batch_size, n_pts)
            batch = positions[start:end]
            dist_batch, vec_batch = batch_distances_periodic(batch, atom_pos, cell, pbc, xp=np)

            for i, sf in enumerate(self.symmetryfunctions):
                if not sf.atomic:
                    n_keys = len(sf.keys) if sf.keys else 0
                    sf_results[i].append(np.zeros((len(batch), n_keys), dtype=np.float64))
                    continue
                if _has_batch_api(sf):
                    batch_result = sf.values_batch(
                        dist_batch, vec_batch if sf.angular else None, use_gpu=False
                    )
                else:
                    rows = [_sf_values_single(sf, dist_batch[b], vec_batch[b])
                            for b in range(len(batch))]
                    batch_result = np.stack(rows, axis=0)
                sf_results[i].append(np.asarray(batch_result, dtype=np.float64))

        return [np.concatenate(frags, axis=0) for frags in sf_results]
    # Batched GPU calculation
    def _process_gpu(self, positions):
        import cupy as cp
        n_pts        = len(positions)
        atom_pos     = self.system.atoms.positions
        cell         = np.asarray(self.system.atoms.cell)
        pbc          = np.asarray(self.system.atoms.pbc, dtype=bool)
        atom_pos_gpu = cp.asarray(atom_pos, dtype=cp.float64)
        cell_gpu     = cp.asarray(cell,     dtype=cp.float64)
        sf_results   = [[] for _ in range(self.nsfs)]

        # Conservative 64 MiB target for dense temporary pair arrays.
        # One point may still exceed it for exceptionally large atom counts.
        batch_size = min(self.batch_size, max(1, (64 * 1024**2) // (max(1, len(self.system.atoms)) * 16 * 8)))
        for start in range(0, n_pts, batch_size):
            end       = min(start + batch_size, n_pts)
            batch_gpu = cp.asarray(positions[start:end], dtype=cp.float64)
            dist_gpu, vec_gpu = batch_distances_periodic(
                batch_gpu, atom_pos_gpu, cell_gpu, pbc, xp=cp
            )

            for i, sf in enumerate(self.symmetryfunctions):
                if not sf.atomic:
                    n_keys = len(sf.keys) if sf.keys else 0
                    sf_results[i].append(np.zeros((end-start, n_keys), dtype=np.float64))
                    continue

                if _has_batch_api(sf):
                    if sf.angular:
                        # Angular SFs need CPU for pair operations
                        dist_cpu = to_cpu(dist_gpu)
                        vec_cpu  = to_cpu(vec_gpu)
                        batch_result = sf.values_batch(dist_cpu, vec_cpu, use_gpu=False)
                    else:
                        # Radial SFs: pass CuPy array directly — Chebyshev runs on GPU.
                        # values_batch detects CuPy input via hasattr(x, 'get') and uses
                        # _calculate_radial_batch_gpu. Result returned as numpy.
                        batch_result = sf.values_batch(dist_gpu, None, use_gpu=True)
                else:
                    # No batch API: fall back to serial CPU
                    dist_cpu = to_cpu(dist_gpu)
                    vec_cpu  = to_cpu(vec_gpu) if sf.angular else None
                    rows = [_sf_values_single(sf, dist_cpu[b],
                                              vec_cpu[b] if vec_cpu is not None else None)
                            for b in range(end - start)]
                    batch_result = np.stack(rows, axis=0)

                sf_results[i].append(np.asarray(batch_result, dtype=np.float64))

            if start % (self.batch_size * 10) == 0:
                cp.get_default_memory_pool().free_all_blocks()

        return [np.concatenate(frags, axis=0) for frags in sf_results]
    # Serial reference processor
    def _process_chunk(self, chunk):
        from ase.geometry import get_distances
        chunk_results = [[] for _ in range(self.nsfs)]
        for position in chunk:
            if self.hasatomicsf:
                atvect, atdist = get_distances(
                    position,
                    self.system.atoms.positions,
                    cell=self.system.atoms.cell,
                    pbc=self.system.atoms.pbc,
                )
            for i, sf in enumerate(self.symmetryfunctions):
                if sf.atomic:
                    if sf.angular:
                        values = sf.values(atdist[0], atvect[0])
                    else:
                        values = sf.values(atdist[0])
                    chunk_results[i].append(values)
        return chunk_results
    # Results to DataFrame
    def _results2df(self, positions, results):
        data = pd.DataFrame(data=positions, columns=["x", "y", "z"])
        for i, sf in enumerate(self.symmetryfunctions):
            values = np.array(results[i])
            if values.ndim == 1:
                values = values.reshape(1, -1)
            for j, key in enumerate(sf.keys):
                y = pd.DataFrame({key: values[:, j]}, dtype="Float64")
                data.loc[:, key] = y
        return data
    # plot — public analysis API
    def plot(self, feature=None, index=None, axes=['x','y'], region=0,
             splitby='z', set_aspect='on', levels=10, **kwargs):
        if self.data is None:
            raise RuntimeError("No contact space data available.")
        filterdata = self.data[self.contactspace.data['region'] == region]
        if feature is not None:
            if feature not in self.data.columns:
                raise ValueError(f"Feature {feature} not found in maps data.")
        elif index is not None:
            if index >= len(self.features) or index < 0:
                raise ValueError(f"Index {index} out of bounds.")
            feature = self.features[index]
        else:
            raise ValueError("Either feature or index must be provided.")
        for axis in axes:
            if axis not in self.data.columns:
                raise ValueError(f"Axis {axis} not found in maps data.")
        if set_aspect not in ['on','off','equal','scaled']:
            raise ValueError(f"set_aspect must be one of ['on','off','equal','scaled']")
        x1 = filterdata[axes[0]].values.astype(np.float64)
        x2 = filterdata[axes[1]].values.astype(np.float64)
        f  = filterdata[feature].values.astype(np.float64)
        if splitby not in self.data.columns:
            raise ValueError(f"Split axis {splitby} not found in maps data.")
        s      = filterdata[splitby].values
        nsplit = np.unique(s).size
        if nsplit == 1:
            fig, axs = plt.subplots(1, 1, figsize=(8, 4))
            axslist = [axs]
        else:
            fig, axs = plt.subplots(nsplit, 1, figsize=(8, 4*nsplit))
            axslist = axs.flat
        fig.subplots_adjust(hspace=0.3)
        fmin   = np.min(f); fmax = np.max(f)
        flevels = np.linspace(fmin, fmax, levels)
        for i, ax in enumerate(axslist):
            sval = np.unique(s)[i]
            ax.set_title(f"Map of {feature} for {splitby} = {sval:6.2f}")
            mask = filterdata[splitby] == sval
            im   = ax.tricontourf(x1[mask], x2[mask], f[mask], levels=flevels, **kwargs)
            ax.axis(set_aspect)
            ax.set_xlabel(f"{axes[0]}")
            ax.set_ylabel(f"{axes[1]}")
        if nsplit == 1:
            fig.colorbar(im, ax=axs)
        else:
            fig.colorbar(im, ax=axs.ravel().tolist())
        return fig, axs
    # scatter — public analysis API
    def scatter(self, feature=None, index=None, axes=['x','y'], region=0,
                categorical=False, centroids=False, splitby=None,
                set_aspect='on', **kwargs):
        if self.data is None:
            raise RuntimeError("No contact space data available.")
        filterdata = self.data[self.contactspace.data['region'] == region]
        if feature is not None:
            if feature not in self.data.columns:
                raise ValueError(f"Feature {feature} not found in maps data.")
        elif index is not None:
            if index >= len(self.features) or index < 0:
                raise ValueError(f"Index {index} out of bounds.")
            feature = self.features[index]
            print(f"Plotting feature {feature}")
        else:
            raise ValueError("Either feature or index must be provided.")
        for axis in axes:
            if axis not in self.data.columns:
                raise ValueError(f"Axis {axis} not found in maps data.")
        if set_aspect not in ['on','off','equal','scaled']:
            raise ValueError(f"set_aspect must be one of ['on','off','equal','scaled']")
        x1 = filterdata[axes[0]].values.astype(np.float64)
        x2 = filterdata[axes[1]].values.astype(np.float64)
        f  = filterdata[feature].values.astype(np.float64)
        if splitby is not None:
            if splitby not in self.data.columns:
                raise ValueError(f"Split axis {splitby} not found in maps data.")
            s      = filterdata[splitby].values
            nsplit = np.unique(s).size
            fig, axs = plt.subplots(nsplit, 1, figsize=(8, 4*nsplit))
            axslist  = np.atleast_1d(axs).flat
        else:
            nsplit   = 1
            fig, axs = plt.subplots(nsplit, 1, figsize=(8, 4*nsplit))
            axslist  = [axs]
        if categorical:
            nf     = np.unique(f).size
            colors = [c for c in map(plt.cm.tab20, range(nf))]
        else:
            fmin = np.min(f); fmax = np.max(f)
        fig.subplots_adjust(hspace=0.3)
        for i, ax in enumerate(axslist):
            if splitby is not None:
                sval = np.unique(s)[i]
                ax.set_title(f"Map of {feature} for {splitby} = {sval:6.2f}")
                mask = filterdata[splitby] == sval
                x1m, x2m, fm = x1[mask], x2[mask], f[mask]
            else:
                ax.set_title(f"Map of {feature}")
                x1m, x2m, fm = x1, x2, f
            ax.set_xlabel(f"{axes[0]}")
            ax.set_ylabel(f"{axes[1]}")
            if not categorical:
                if f is None:
                    scatter = ax.scatter(x1m, x2m, **kwargs)
                else:
                    scatter = ax.scatter(x1m, x2m, c=fm, vmin=fmin, vmax=fmax, **kwargs)
            else:
                for j, fvalue in enumerate(np.unique(f)):
                    scatter = ax.scatter(x1m[fm==fvalue], x2m[fm==fvalue],
                                         color=colors[j], label=f'{feature} = {j:2}', **kwargs)
            if centroids:
                if self.centroids is None:
                    raise RuntimeError("No centroids available.")
                pos = self.data.loc[self.centroids, axes].values
                if splitby is not None:
                    posmask = self.data.loc[self.centroids, splitby].values == sval
                else:
                    posmask = np.ones(pos.shape[0], dtype=bool)
                ax.scatter(pos[posmask,0], pos[posmask,1], c='black', marker='x',
                           label='Centroids', **kwargs)
            if categorical:
                leg = ax.legend(bbox_to_anchor=(1.01, 0.5), loc="center left")
                for lh in leg.legend_handles:
                    lh.set_alpha(1)
            ax.axis(set_aspect)
        if not categorical and f is not None:
            if nsplit > 1:
                colorbar = fig.colorbar(scatter, ax=axs.ravel().tolist())
            else:
                colorbar = fig.colorbar(scatter, ax=ax)
            colorbar.solids.set_alpha(1.0)
        return fig, axs
    # scatter_pca_grid — public analysis API
    def scatter_pca_grid(self, feature=None, index=None, set_aspect='on', **kwargs):
        """
        Upper-triangle PCA pair grid coloured by a symmetry-function feature.
        Must call reduce(npca=N) first.
        Pass cmap, s, alpha etc. via **kwargs.
        Returns fig, gs.
        """
        if self.data is None:
            raise RuntimeError("No contact space data available.")
        if feature is not None:
            if feature not in self.data.columns:
                raise ValueError(f"Feature {feature} not found in maps data.")
        elif index is not None:
            if index >= len(self.features) or index < 0:
                raise ValueError(f"Index {index} out of bounds.")
            feature = self.features[index]
            print(f"Plotting feature {feature}")
        else:
            raise ValueError("Either feature or index must be provided.")
        if self.npca is None:
            raise ValueError("Missing principal components. Call reduce(npca=N) first.")
        if set_aspect not in ['on','off','equal','scaled']:
            raise ValueError(f"set_aspect must be one of ['on','off','equal','scaled']")

        pcalabels     = [f'pca{i}' for i in range(self.npca)]
        pcamaxs       = [self.data[pcalabels[i]].max() for i in range(self.npca)]
        pcamins       = [self.data[pcalabels[i]].min() for i in range(self.npca)]
        pcaranges     = [pcamaxs[i] - pcamins[i] for i in range(self.npca)]
        maxrange      = max(pcaranges)
        pcaproportions = [pcaranges[i] / maxrange for i in range(self.npca)]
        xratios       = [pcaproportions[i+1] for i in range(self.npca-1)]
        yratios       = [pcaproportions[i]   for i in range(self.npca-1)]

        y      = self.data[feature].values.astype(np.float64)
        n_cols = len(xratios)
        n_rows = len(yratios)

        fig = plt.figure(figsize=(12, 12))
        gs  = GridSpec(n_cols, n_rows, figure=fig,
                       height_ratios=yratios, width_ratios=xratios)

        for i in range(self.npca - 1):
            for j in range(i+1, self.npca):
                ax = fig.add_subplot(gs[i, j-1])
                X1 = self.data[pcalabels[i]].values.astype(np.float64)
                X2 = self.data[pcalabels[j]].values.astype(np.float64)
                scatter = ax.scatter(X2, X1, c=y, edgecolor='k', **kwargs)
                ax.set_title(f'PC{j+1} vs PC{i+1}')
                ax.set_xlim(np.min(X2)-0.5, np.max(X2)+0.5)
                ax.set_ylim(np.min(X1)-0.5, np.max(X1)+0.5)
                ax.axis(set_aspect)
                ax.xaxis.set_major_locator(MultipleLocator(2.5))
                ax.yaxis.set_major_locator(MultipleLocator(2.5))

        cbar_ax  = fig.add_subplot(gs[n_rows-1, 0:n_rows-1])
        colorbar = fig.colorbar(scatter, cax=cbar_ax, orientation='horizontal')
        colorbar.solids.set_alpha(1.0)

        if n_rows == 2:
            fig.suptitle(f"Maps of {feature}\n\n in PC Space", fontsize=22)
        else:
            title_ax = fig.add_subplot(gs[n_rows-2, 0:n_rows-2])
            title_ax.text(0.5, 0.5, f"Maps of {feature}\n\n in PC Space",
                          ha='center', va='center', fontsize=22)
            title_ax.axis('off')

        return fig, gs
    # Export a descriptor as a volumetric field
    def tovolumetric(self, feature=None, index=None, reference=0.0):
        """Convert contact space data to a volumetric ScalarField on the grid."""
        if self.data is None:
            raise RuntimeError("No contact space data available.")
        if feature is not None:
            if feature not in self.data.columns:
                raise ValueError(f"Feature {feature} not found in maps data.")
            f = self.data[feature].values.astype(np.float64)
        elif index is not None:
            if index >= len(self.features) or index < 0:
                raise ValueError(f"Index {index} out of bounds.")
            print(f"Converting feature {feature}")
            f = self.data.iloc[:, index+3].values.astype(np.float64)
        else:
            raise ValueError("Either feature or index must be provided.")

        if reference == 0.0:
            data = np.zeros(self.contactspace.grid.coordinates.shape[1:])
        else:
            data = reference * np.ones(self.contactspace.grid.coordinates.shape[1:])
        data[self.contactspace.mask] = f
        return ScalarField(self.contactspace.grid, data=data)
    # reduce — public analysis API
    def reduce(self, npca=None, scale=False):
        if self.data is None:
            raise RuntimeError("No contact space data available.")
        if self.data is None:
            raise RuntimeError("No data — call atcontactspace() first")
        features = self.data[self.features].values.astype(np.float64)
        if scale:
            features = StandardScaler().fit_transform(features)
        if npca is not None:
            self.npca        = npca
            pca              = PCA(npca)
            pca_features     = [f'pca{i:1d}' for i in range(npca)]
            self.data[pca_features] = pca.fit_transform(features)
        else:
            pca = PCA()
            pca.fit(features)
            explained_variance = np.cumsum(pca.explained_variance_ratio_)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            ax1.plot(range(1, len(explained_variance)+1), explained_variance, marker='o')
            ax1.set_xlabel('Number of Components')
            ax1.set_ylabel('Cumulative Explained Variance')
            ax1.grid(True)
            ax1.set_title('Optimal Number of Components in PCA')
            ax1.axhline(y=0.98, color='r', linestyle='--')
            ax2.plot(range(1, len(pca.explained_variance_ratio_)+1),
                     pca.explained_variance_ratio_, marker='o')
            ax2.set_xlabel('Number of Components')
            ax2.set_ylabel('Explained Variance Ratio')
            ax2.set_title('Scree Plot')
            ax2.grid(True)
            return fig, ax1, ax2
    # graph — public analysis API
    def graph(self, clusters):
        from mapsy.utils.graph import cluster_graph
        return cluster_graph(clusters, self.contactspace.neighbors)
    # cluster — public analysis API (smart random_state reuse)
    def cluster(self, nclusters=None, features=None, maxclusters=20,
                ntries=1, random_state=None, scale=False):
        if features is None:
            self.cluster_features = self.features
        else:
            self.cluster_features = features
        if self.data is None:
            raise RuntimeError("No data — call atcontactspace() first")
        X = self.data[self.cluster_features].values.astype(np.float64)
        if len(X) < 2 or X.shape[1] == 0 or not np.isfinite(X).all():
            raise ValueError("Clustering requires at least two finite samples and one feature")
        if nclusters is not None and not 1 <= int(nclusters) < len(X):
            raise ValueError("nclusters must be between 1 and n_samples - 1")
        if nclusters is None and min(maxclusters, len(X)) <= 2:
            raise ValueError("Cluster screening requires at least three samples")
        self._cluster_scaler = StandardScaler().fit(X) if scale else None
        if self._cluster_scaler is not None:
            X = self._cluster_scaler.transform(X)

        if nclusters is not None:
            if random_state is None:
                if self.best_clusters is not None:
                    if nclusters in self.best_clusters['nclusters'].values:
                        self.random_state = self.best_clusters[
                            self.best_clusters['nclusters'] == nclusters
                        ]['random_state'].values[0]
                        print(f'Use best random state = {self.random_state} from screening')
                    else:
                        self.random_state = np.random.randint(0, 1000)
                        print(f'Use new random state = {self.random_state}')
                else:
                    self.random_state = np.random.randint(0, 1000)
                    print(f'Use new random state = {self.random_state}')
            else:
                self.random_state = random_state
                print(f'Use given random state = {self.random_state}')

            labels = SpectralClustering(
                n_clusters=nclusters, random_state=self.random_state
            ).fit_predict(X)
            self.data['Cluster']  = labels
            self.nclusters        = nclusters
            self.cluster_centers  = (
                self.data.groupby('Cluster')[self.cluster_features].mean()
                .values.astype(np.float64)
            )
            self.cluster_sizes  = self.data.groupby('Cluster').size().values
            self.cluster_graph  = self.graph(self.data['Cluster'])
            self.cluster_edges  = self.cluster_graph.copy()
            for i in range(nclusters):
                self.cluster_edges[i, i] = 0
        else:
            cluster_range         = range(2, min(maxclusters, len(X)))
            cluster_random_states = []
            cluster_sizes_list    = []
            silhouette_scores     = []
            db_indexes            = []
            for nc in cluster_range:
                random_states = np.random.randint(0, 1000, ntries)
                for rs in random_states:
                    cluster_random_states.append(rs)
                    labels           = SpectralClustering(n_clusters=nc, random_state=rs).fit_predict(X)
                    actual_nc        = len(np.unique(labels))
                    cluster_sizes_list.append(actual_nc)
                    silhouette_scores.append(silhouette_score(X, labels))
                    db_indexes.append(davies_bouldin_score(X, labels))
            self.cluster_screening = pd.DataFrame({
                'nclusters':        cluster_sizes_list,
                'random_state':     cluster_random_states,
                'silhouette_score': silhouette_scores,
                'db_index':         db_indexes,
            })
            best_db  = self.cluster_screening.loc[
                self.cluster_screening.groupby('nclusters')['db_index'].idxmin()
            ]
            best_sil = self.cluster_screening.loc[
                self.cluster_screening.groupby('nclusters')['silhouette_score'].idxmax()
            ]
            self.best_clusters = best_db
            fig, ax1 = plt.subplots()
            ax1.scatter(cluster_sizes_list, silhouette_scores, color='b', marker='o',
                        label='Silhouette Score')
            ax1.plot(best_db['nclusters'],  best_db['silhouette_score'],  '-', color='b')
            ax1.plot(best_sil['nclusters'], best_sil['silhouette_score'], ':', color='b')
            ax1.set_xlabel('Number of Clusters')
            ax1.set_ylabel('Silhouette Score', color='b')
            ax1.tick_params(axis='y', labelcolor='b')
            ax2 = ax1.twinx()
            ax2.scatter(cluster_sizes_list, db_indexes, color='r', marker='s',
                        label='Davies-Bouldin Index')
            ax2.plot(best_db['nclusters'],  best_db['db_index'],  '-', color='r')
            ax2.plot(best_sil['nclusters'], best_sil['db_index'], ':', color='r')
            ax2.set_ylabel('Davies-Bouldin Index', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            ax1.set_title('Silhouette Score and Davies-Bouldin Index vs. Number of Clusters')
            ax1.grid(True)
            return fig, ax1, ax2
    # sites — public analysis API
    def sites(self, region=0):
        if self.data is None:
            raise RuntimeError("No contact space data available.")
        filterdata = self.data[self.contactspace.data['region'] == region]
        if self.nclusters == 0:
            raise RuntimeError("No clusters have been generated.")
        centroid_indexes = []
        cluster_sizes    = np.unique(filterdata['Cluster'].values)
        for cluster_index in cluster_sizes:
            cluster_points  = filterdata[
                filterdata['Cluster'] == cluster_index
            ][self.cluster_features].values.astype(np.float64)
            cluster_indexes = filterdata[
                filterdata['Cluster'] == cluster_index
            ].index.values
            centers = self.cluster_centers
            scaler = getattr(self, "_cluster_scaler", None)
            if scaler is not None:
                cluster_points = scaler.transform(cluster_points)
                centers = scaler.transform(centers)
            dist      = distance.cdist(cluster_points, centers)
            indexes   = np.where(self.cluster_edges[cluster_index, :] != 0)[0]
            full_dist = np.sum(self.cluster_sizes[indexes] / (dist[:, indexes]**2 + 1e-30), axis=1)
            cluster_centroid = np.argmin(full_dist)
            centroid_indexes.append(cluster_indexes[cluster_centroid])
        self.centroids = np.array(centroid_indexes, dtype=int)

    def tofile(self):
        raise NotImplementedError("Maps.tofile has no defined output format; use data.to_csv")
# MapsFromFile
class MapsFromFile(Maps):

    def __init__(self, filename, **kwargs):
        inp = Input(filename)
        system = SystemParser(inp.system).parse()
        super().__init__(
            system=system,
            symmetryfunctions=SymmetryFunctionsParser(inp.symmetryfunctions).parse(),
            contactspace=ContactSpaceGenerator(inp.contactspace).generate(
                system
            ),
            verbosity=inp.control.verbosity,
            debug=inp.control.debug,
            **kwargs,
        )
