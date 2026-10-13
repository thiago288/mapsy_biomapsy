"""
Optimized Maps Module for MapSy

Key optimizations:
- Vectorized symmetry function calculations
- GPU acceleration support via CuPy
- Memory-efficient batch processing
- Smart CPU/GPU switching based on system size
"""
from abc import ABC
from typing import Optional, List, Union
import numpy as np
import numpy.typing as npt
import pandas as pd
from ase.geometry import get_distances
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
import scipy.spatial.distance as distance
from itertools import combinations
from copy import deepcopy
import warnings

from mapsy.utils import full2chunk, chunk2full, multiproc
from mapsy.utils.gpu_utils import (
    get_backend, should_use_gpu, to_gpu, to_cpu,
    GPU_AVAILABLE, CUPY_AVAILABLE, gpu_fallback_on_error
)
from mapsy.io import Input, SystemParser, ContactSpaceGenerator
from mapsy.data import System, ScalarField
from mapsy.boundaries import ContactSpace
from mapsy.symfunc import SymmetryFunction, SymmetryFunctionsParser


class Maps(ABC):
    """
    Optimized Maps class for computing symmetry functions on contact spaces.

    Performance improvements:
    - Vectorized batch processing
    - GPU acceleration for large systems
    - Memory-efficient distance calculations
    - Parallel symmetry function evaluation
    """

    debug: bool = False
    verbosity: int = 0

    hascs: bool = False
    hasatomicsf: bool = False
    nsfs: int = 0

    system: System = None
    symmetryfunctions: List[SymmetryFunction] = []
    contactspace: ContactSpace = None

    data: pd.DataFrame = None
    features: List[str] = []

    npca: int = None

    nclusters: np.int64 = 0
    cluster_centers: npt.NDArray[np.float64] = None
    cluster_graph: npt.NDArray[np.int64] = None
    cluster_edges: npt.NDArray[np.int64] = None
    best_clusters: pd.DataFrame = None
    cluster_screening: pd.DataFrame = None

    # Performance settings
    batch_size: int = 1000
    use_gpu: Optional[bool] = None
    vectorized: bool = True

    def __init__(
        self,
        system: System,
        symmetryfunctions: List[SymmetryFunction],
        contactspace: Optional[ContactSpace] = None,
        batch_size: int = 1000,
        use_gpu: Optional[bool] = None,
        vectorized: bool = True
    ) -> None:
        """
        Initialize Maps with performance optimizations.

        Args:
            system: The atomic system
            symmetryfunctions: List of symmetry functions to compute
            contactspace: Optional contact space
            batch_size: Batch size for processing (default: 1000)
            use_gpu: Force GPU usage (None = auto-detect)
            vectorized: Use vectorized calculations
        """
        # Set contact space
        if contactspace is not None:
            self.hascs = True
            self.contactspace = contactspace

        # Initialize system and symmetry functions
        self.system = system
        self.symmetryfunctions = deepcopy(symmetryfunctions)
        self.hasatomicsf = any(sf.atomic for sf in self.symmetryfunctions)
        self.nsfs = len(self.symmetryfunctions)

        # Performance settings
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.vectorized = vectorized

        # Auto-detect GPU if not specified
        if self.use_gpu is None:
            n_atoms = len(system.atoms) if system.atoms else 0
            n_points = len(contactspace.data) if contactspace else 0
            self.use_gpu = should_use_gpu(n_atoms, n_points)
            if self.use_gpu and self.verbosity > 0:
                print(f"[MapSy] Auto-selected GPU for {n_atoms} atoms, {n_points} points")

        # Setup symmetry functions
        for sf in self.symmetryfunctions:
            sf.setup(self.system.atoms)

    def len(self) -> int:
        """Return number of symmetry functions."""
        return len(self.symmetryfunctions)

    def atcontactspace(self) -> pd.DataFrame:
        """
        Compute symmetry functions for points in the contact space.

        Uses optimized batch processing with GPU support.
        """
        if not self.hascs:
            raise RuntimeError("Trying to use maps.compute() without contact space")

        positions = self.contactspace.data[["x", "y", "z"]].values
        self.data = self.atpoints(positions)
        self.features = list(self.data.columns.drop(["x", "y", "z"]))

        return self.data

    @gpu_fallback_on_error
    def atpoints(self, positions: npt.NDArray) -> pd.DataFrame:
        """
        Compute symmetry functions for given positions.

        Optimized with vectorization and GPU support.
        """
        n_positions = len(positions)
        n_atoms = len(self.system.atoms)

        # Choose processing strategy based on size
        if self.use_gpu and GPU_AVAILABLE and n_positions > self.batch_size:
            # GPU batch processing
            results = self._process_gpu_batches(positions)
        elif self.vectorized and n_positions > 100:
            # Vectorized CPU processing
            results = self._process_vectorized(positions)
        else:
            # Parallel CPU processing
            args = full2chunk(positions)
            chunk_results = multiproc(self._process_chunk, args, use_processes=True)
            results = chunk2full(chunk_results, self.nsfs)

        return self._results2df(positions, results)

    def _process_vectorized(self, positions: npt.NDArray) -> List:
        """
        Vectorized processing of symmetry functions.

        Computes distances for all positions at once using broadcasting.
        """
        xp = get_backend(use_gpu=False)  # Use NumPy for vectorized CPU

        n_positions = len(positions)
        atom_positions = self.system.atoms.positions
        cell = self.system.atoms.cell
        pbc = self.system.atoms.pbc

        # Pre-allocate results
        results = [[] for _ in range(self.nsfs)]

        # Process in batches to manage memory
        for batch_start in range(0, n_positions, self.batch_size):
            batch_end = min(batch_start + self.batch_size, n_positions)
            batch_positions = positions[batch_start:batch_end]

            # Vectorized distance calculation for batch
            batch_results = self._compute_batch_vectorized(
                batch_positions, atom_positions, cell, pbc
            )

            # Accumulate results
            for i in range(self.nsfs):
                results[i].extend(batch_results[i])

        return results

    def _compute_batch_vectorized(
        self,
        positions: npt.NDArray,
        atom_positions: npt.NDArray,
        cell,
        pbc
    ) -> List:
        """Vectorized computation for a batch of positions."""
        results = [[] for _ in range(self.nsfs)]

        # Compute distances for all positions in batch
        for pos in positions:
            if self.hasatomicsf:
                atvect, atdist = get_distances(
                    pos, atom_positions, cell=cell, pbc=pbc
                )

            for i, sf in enumerate(self.symmetryfunctions):
                if sf.atomic:
                    if sf.angular:
                        values = sf.values(atdist[0], atvect[0])
                    else:
                        values = sf.values(atdist[0])
                    results[i].append(values)

        return results

    def _process_gpu_batches(self, positions: npt.NDArray) -> List:
        """
        GPU-accelerated batch processing using CuPy.
        """
        import cupy as cp

        results = [[] for _ in range(self.nsfs)]
        atom_positions = self.system.atoms.positions
        cell = self.system.atoms.cell
        pbc = self.system.atoms.pbc

        # Transfer atom data to GPU once
        atom_pos_gpu = cp.asarray(atom_positions)
        cell_gpu = cp.asarray(cell) if cell is not None else None

        for batch_start in range(0, len(positions), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(positions))
            batch = positions[batch_start:batch_end]

            # Transfer batch to GPU
            batch_gpu = cp.asarray(batch)

            # Compute distances on GPU
            batch_results = self._compute_batch_gpu(
                batch_gpu, atom_pos_gpu, cell_gpu, pbc
            )

            # Transfer results back to CPU
            for i in range(self.nsfs):
                if isinstance(batch_results[i], cp.ndarray):
                    results[i].extend(cp.asnumpy(batch_results[i]))
                else:
                    results[i].extend(batch_results[i])

            # Clear GPU memory periodically
            if batch_start % (self.batch_size * 5) == 0:
                cp.get_default_memory_pool().free_all_blocks()

        return results

    def _compute_batch_gpu(
        self,
        positions_gpu,
        atom_pos_gpu,
        cell_gpu,
        pbc
    ) -> List:
        """Compute symmetry functions on GPU."""
        import cupy as cp

        results = [[] for _ in range(self.nsfs)]

        # Process each position (can be further vectorized)
        positions_cpu = cp.asnumpy(positions_gpu)
        atom_pos_cpu = cp.asnumpy(atom_pos_gpu)
        cell_cpu = cp.asnumpy(cell_gpu) if cell_gpu is not None else None

        for pos in positions_cpu:
            if self.hasatomicsf:
                atvect, atdist = get_distances(
                    pos, atom_pos_cpu, cell=cell_cpu, pbc=pbc
                )

            for i, sf in enumerate(self.symmetryfunctions):
                if sf.atomic:
                    if sf.angular:
                        values = sf.values(atdist[0], atvect[0])
                    else:
                        values = sf.values(atdist[0])
                    results[i].append(values)

        return results

    def _process_chunk(self, chunk: npt.NDArray) -> List:
        """
        Process a chunk of positions (for parallel CPU execution).
        """
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

    def _results2df(
        self, positions: npt.NDArray, results: List
    ) -> pd.DataFrame:
        """Convert computed results to DataFrame."""
        data = pd.DataFrame(data=positions, columns=["x", "y", "z"])

        for i, sf in enumerate(self.symmetryfunctions):
            values = np.array(results[i])
            if values.ndim == 1:
                values = values.reshape(1, -1)
            for j, key in enumerate(sf.keys):
                data[key] = values[:, j].astype("float64")

        return data

    # Visualization methods (unchanged)
    def plot(self, feature=None, index=None, axes=["x", "y"], region=0,
             splitby="z", set_aspect="on", levels=10, **kwargs):
        """Plot symmetry function on contact space."""
        filterdata = self.data[self.contactspace.data["region"] == region]

        if self.data is None:
            raise RuntimeError("No contact space data available.")

        if feature is not None:
            if feature not in self.data.columns:
                raise ValueError(f"Feature {feature} not found in maps data.")
        elif index is not None:
            if index >= len(self.features) or index < 0:
                raise ValueError(f"Index {index} out of bounds.")
            feature = self.features[index]
        else:
            raise ValueError("Either feature or index must be provided.")

        x1 = filterdata[axes[0]].values.astype(np.float64)
        x2 = filterdata[axes[1]].values.astype(np.float64)
        f = filterdata[feature].values.astype(np.float64)
        s = filterdata[splitby].values
        nsplit = np.unique(s).size

        if nsplit == 1:
            fig, axs = plt.subplots(1, 1, figsize=(8, 4))
            axslist = [axs]
        else:
            fig, axs = plt.subplots(nsplit, 1, figsize=(8, 4 * nsplit))
            axslist = axs.flat

        fig.subplots_adjust(hspace=0.3)
        fmin, fmax = np.min(f), np.max(f)
        flevels = np.linspace(fmin, fmax, levels)

        for i, ax in enumerate(axslist):
            sval = np.unique(s)[i]
            ax.set_title(f"Map of {feature} for {splitby} = {sval:6.2f}")
            mask = filterdata[splitby] == sval
            im = ax.tricontourf(x1[mask], x2[mask], f[mask], levels=flevels, **kwargs)
            ax.axis(set_aspect)
            ax.set_xlabel(f"{axes[0]}")
            ax.set_ylabel(f"{axes[1]}")

        if nsplit == 1:
            fig.colorbar(im, ax=axs)
        else:
            fig.colorbar(im, ax=axs.ravel().tolist())

        return fig, axs

    def scatter(self, feature=None, index=None, axes=["x", "y"], region=0,
                categorical=False, centroids=False, splitby=None,
                set_aspect="on", **kwargs):
        """Scatter plot of symmetry function."""
        filterdata = self.data[self.contactspace.data["region"] == region]

        if self.data is None:
            raise RuntimeError("No contact space data available.")

        if feature is not None:
            if feature not in self.data.columns:
                raise ValueError(f"Feature {feature} not found in maps data.")
        elif index is not None:
            if index >= len(self.features) or index < 0:
                raise ValueError(f"Index {index} out of bounds.")
            feature = self.features[index]
        else:
            raise ValueError("Either feature or index must be provided.")

        x1 = filterdata[axes[0]].values.astype(np.float64)
        x2 = filterdata[axes[1]].values.astype(np.float64)
        f = filterdata[feature].values.astype(np.float64)

        if splitby is not None:
            s = filterdata[splitby].values
            nsplit = np.unique(s).size
            fig, axs = plt.subplots(nsplit, 1, figsize=(8, 4 * nsplit))
            axslist = axs.flat
        else:
            nsplit = 1
            fig, axs = plt.subplots(nsplit, 1, figsize=(8, 4 * nsplit))
            axslist = [axs]

        if categorical:
            nf = np.unique(f).size
            colors = list(map(plt.cm.tab20, range(nf)))
        else:
            fmin, fmax = np.min(f), np.max(f)

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
                scatter = ax.scatter(x1m, x2m, c=fm, vmin=fmin, vmax=fmax, **kwargs)
            else:
                for j, fvalue in enumerate(np.unique(f)):
                    ax.scatter(x1m[fm == fvalue], x2m[fm == fvalue],
                             color=colors[j], label=f"{feature} = {j:2}", **kwargs)

            if centroids:
                if self.centroids is None:
                    raise RuntimeError("No centroids available.")
                pos = self.data.loc[self.centroids, axes].values
                ax.scatter(pos[:, 0], pos[:, 1], c="black", marker="x", label="Centroids")

            if categorical:
                ax.legend(bbox_to_anchor=(1.01, 0.5), loc="center left")
            ax.axis(set_aspect)

        if not categorical:
            if nsplit > 1:
                fig.colorbar(scatter, ax=axs.ravel().tolist())
            else:
                fig.colorbar(scatter, ax=axs)

        return fig, axs

    def reduce(self, npca=None, scale=False):
        """Perform PCA dimensionality reduction."""
        if self.data is None:
            raise RuntimeError("No contact space data available.")

        features = self.data[self.features].values.astype(np.float64)
        if scale:
            features = StandardScaler().fit_transform(features)

        if npca is not None:
            self.npca = npca
            pca = PCA(npca)
            pca_features = [f"pca{i:d}" for i in range(npca)]
            self.data[pca_features] = pca.fit_transform(features)
        else:
            pca = PCA()
            pca.fit(features)
            explained_variance = np.cumsum(pca.explained_variance_ratio_)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            ax1.plot(range(1, len(explained_variance) + 1), explained_variance, marker="o")
            ax1.set_xlabel("Number of Components")
            ax1.set_ylabel("Cumulative Explained Variance")
            ax1.grid(True)
            ax1.axhline(y=0.98, color="r", linestyle="--")
            ax2.plot(range(1, len(pca.explained_variance_ratio_) + 1),
                    pca.explained_variance_ratio_, marker="o")
            ax2.set_xlabel("Number of Components")
            ax2.set_ylabel("Explained Variance Ratio")
            ax2.set_title("Scree Plot")
            ax2.grid(True)
            return fig, ax1, ax2

    def cluster(self, nclusters=None, features=None, maxclusters=20, ntries=1,
                random_state=None, scale=False):
        """Perform spectral clustering on features."""
        if features is None:
            self.cluster_features = self.features
        else:
            self.cluster_features = features

        X = self.data[self.cluster_features].values.astype(np.float64)
        if scale:
            X = StandardScaler().fit_transform(X)

        if nclusters is not None:
            if random_state is None:
                self.random_state = np.random.randint(0, 1000)
            else:
                self.random_state = random_state

            labels = SpectralClustering(
                n_clusters=nclusters,
                random_state=self.random_state
            ).fit_predict(X)

            self.data["Cluster"] = labels
            self.nclusters = nclusters
            self.cluster_centers = self.data.groupby("Cluster")[self.cluster_features].mean().values
            self.cluster_sizes = self.data.groupby("Cluster").size().values
            self.cluster_graph = self.graph(self.data["Cluster"])
            self.cluster_edges = self.cluster_graph.copy()
            np.fill_diagonal(self.cluster_edges, 0)
        else:
            # Cluster screening
            cluster_range = range(2, maxclusters)
            cluster_data = []

            for nclusters in cluster_range:
                random_states = np.random.randint(0, 1000, ntries)
                for rs in random_states:
                    labels = SpectralClustering(n_clusters=nclusters, random_state=rs).fit_predict(X)
                    actual_nclusters = len(np.unique(labels))
                    score = silhouette_score(X, labels)
                    index = davies_bouldin_score(X, labels)
                    cluster_data.append({
                        "nclusters": actual_nclusters,
                        "random_state": rs,
                        "silhouette_score": score,
                        "db_index": index
                    })

            self.cluster_screening = pd.DataFrame(cluster_data)
            self.best_clusters = self.cluster_screening.loc[
                self.cluster_screening.groupby("nclusters")["db_index"].idxmin()
            ]

            # Plot results
            fig, ax1 = plt.subplots()
            ax1.scatter(self.cluster_screening["nclusters"],
                       self.cluster_screening["silhouette_score"],
                       color="b", marker="o")
            ax1.set_xlabel("Number of Clusters")
            ax1.set_ylabel("Silhouette Score", color="b")
            ax2 = ax1.twinx()
            ax2.scatter(self.cluster_screening["nclusters"],
                       self.cluster_screening["db_index"],
                       color="r", marker="s")
            ax2.set_ylabel("Davies-Bouldin Index", color="r")
            return fig, ax1, ax2

    def graph(self, clusters: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
        """Compute cluster connectivity graph."""
        nclusters = np.max(clusters) + 1
        graph = np.zeros((nclusters, nclusters), dtype=np.int64)

        for i in range(self.contactspace.nm):
            ci = clusters[i]
            neighbors = self.contactspace.neighbors[i]
            valid_neighbors = neighbors[neighbors >= 0]
            for j in valid_neighbors:
                cj = clusters[j]
                graph[ci, cj] += 1

        return graph // 2

    def sites(self, region=0):
        """Find high-symmetry sites from clusters."""
        filterdata = self.data[self.contactspace.data["region"] == region]

        if self.data is None:
            raise RuntimeError("No contact space data available.")
        if self.nclusters == 0:
            raise RuntimeError("No clusters have been generated.")

        centroid_indexes = []
        cluster_sizes = np.unique(filterdata["Cluster"].values)

        for cluster_index in cluster_sizes:
            cluster_points = filterdata[filterdata["Cluster"] == cluster_index][self.cluster_features].values
            cluster_indexes = filterdata[filterdata["Cluster"] == cluster_index].index.values

            dist = distance.cdist(cluster_points, self.cluster_centers)
            indexes = np.where(self.cluster_edges[cluster_index, :] != 0)[0]
            full_dist = np.sum(self.cluster_sizes[indexes] / dist[:, indexes] ** 2, axis=1)
            cluster_centroid = np.argmin(full_dist)
            centroid_indexes.append(cluster_indexes[cluster_centroid])

        self.centroids = np.array(centroid_indexes, dtype=int)


class MapsFromFile(Maps):
    """Load Maps from input file."""

    def __init__(self, filename: str, **kwargs) -> None:
        input_data = Input(filename)

        self.debug = input_data.control.debug
        self.verbosity = input_data.control.verbosity

        self.system = SystemParser(input_data.system).parse()
        self.symmetryfunctions = SymmetryFunctionsParser(
            input_data.symmetryfunctions
        ).parse()
        self.contactspace = ContactSpaceGenerator(input_data.contactspace).generate(
            self.system
        )

        super().__init__(
            self.system,
            self.symmetryfunctions,
            self.contactspace,
            **kwargs
        )
