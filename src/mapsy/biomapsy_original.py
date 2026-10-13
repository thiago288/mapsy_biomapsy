"""
BioMapSy - Optimized Maps for Aperiodic Biological Systems (Proteins)

This module provides specialized handling for protein structures:
- No periodic boundary conditions (PBC=False)
- Optimized neighbor search using KD-trees
- Memory-efficient distance calculations
- GPU acceleration for large proteins (4000+ atoms)
- Streaming processing for very large systems
"""
import numpy as np
import numpy.typing as npt
import pandas as pd
from typing import Optional, List, Callable, Tuple
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
import warnings

from mapsy.maps import Maps
from mapsy.data import System, ScalarField
from mapsy.boundaries import ContactSpace
from mapsy.symfunc import SymmetryFunction
from mapsy.utils.gpu_utils import get_backend, should_use_gpu, to_gpu, to_cpu, GPU_AVAILABLE


class BioSystem(System):
    """
    System class for aperiodic biological structures.
    
    Key differences from periodic System:
    - PBC is always False
    - Uses KD-tree for efficient neighbor search
    - Supports streaming for large proteins
    """
    
    def __init__(self, grid, atoms=None, electrons=None, dimension=0, axis=0):
        super().__init__(grid, atoms, electrons, dimension, axis)
        # Force PBC to False for proteins
        if self.atoms is not None:
            self.atoms.pbc = False
        # Build KD-tree for efficient neighbor search
        self._kdtree = None
        if self.atoms is not None:
            self._build_kdtree()
    
    def _build_kdtree(self):
        """Build KD-tree for fast neighbor searches."""
        self._kdtree = cKDTree(self.atoms.positions)
    
    def query_neighbors(
        self, 
        position: npt.NDArray, 
        radius: float,
        return_distances: bool = True
    ) -> Tuple[npt.NDArray, Optional[npt.NDArray]]:
        """
        Query neighbors within radius using KD-tree.
        
        Much faster than brute-force for large proteins.
        
        Args:
            position: Query position
            radius: Search radius
            return_distances: Return distances along with indices
            
        Returns:
            (indices, distances) or just indices
        """
        if self._kdtree is None:
            self._build_kdtree()
        
        return self._kdtree.query(
            position, 
            k=len(self.atoms), 
            distance_upper_bound=radius,
            return_distance=return_distances
        )
    
    def query_ball_point(
        self, 
        position: npt.NDArray, 
        radius: float
    ) -> List[int]:
        """
        Query all atoms within radius of position.
        
        Args:
            position: Query position
            radius: Search radius
            
        Returns:
            List of atom indices within radius
        """
        if self._kdtree is None:
            self._build_kdtree()
        
        return self._kdtree.query_ball_point(position, radius)


class BioContactSpace:
    """
    Contact space for aperiodic protein systems.
    
    Optimizations:
    - No periodic boundary handling
    - Adaptive grid resolution based on protein size
    - Memory-efficient point generation
    """
    
    def __init__(
        self,
        positions: npt.NDArray,
        probabilities: npt.NDArray,
        grid,
        neighbor_radius: float = 2.0
    ):
        """
        Initialize BioContactSpace.
        
        Args:
            positions: Contact space point coordinates (N, 3)
            probabilities: Probability/weight for each point
            grid: Grid object
            neighbor_radius: Radius for neighbor search
        """
        self.positions = positions
        self.probabilities = probabilities
        self.grid = grid
        self.neighbor_radius = neighbor_radius
        self.nm = len(positions)
        
        # Build KD-tree for neighbor finding
        self._kdtree = cKDTree(positions)
        
        # Compute neighbors and regions
        self._compute_neighbors()
        self._compute_regions()
        
        # Create DataFrame
        _df_dict = {
            "probability": probabilities,
            "x": positions[:, 0],
            "y": positions[:, 1],
            "z": positions[:, 2],
            "nn": self.neighbors,
            "region": self.regions,
        }
        # Sanitize any non-1D columns for pandas:
        # - Nx3 arrays become three columns: <name>_x/_y/_z
        # - other ND arrays become object columns of tuples (row-wise)
        import numpy as _np
        for _k in list(_df_dict.keys()):
            _v = _df_dict[_k]
            if isinstance(_v, _np.ndarray) and getattr(_v, "ndim", 1) > 1:
                if _v.ndim == 2 and _v.shape[1] == 3:
                    _df_dict[_k + "_x"] = _v[:, 0]
                    _df_dict[_k + "_y"] = _v[:, 1]
                    _df_dict[_k + "_z"] = _v[:, 2]
                    del _df_dict[_k]
                else:
                    _v2 = _v.reshape(_v.shape[0], -1)
                    _df_dict[_k] = [tuple(row) for row in _v2]
        # Ensure ragged/ND columns are 1D object columns for pandas
        # neighbors is a list-of-lists; store as tuple per row
        if 'nn' in _df_dict:
            _df_dict['nn'] = [tuple(x) for x in _df_dict['nn']]
        # regions may be list-of-lists or ND array; store as tuple per row if needed
        if 'region' in _df_dict:
            _r = _df_dict['region']
            try:
                import numpy as _np
                if isinstance(_r, _np.ndarray) and getattr(_r, 'ndim', 1) > 1:
                    _df_dict['region'] = [tuple(row) for row in _r.reshape(_r.shape[0], -1)]
                elif isinstance(_r, list) and len(_r) > 0 and isinstance(_r[0], (list, tuple)):
                    _df_dict['region'] = [tuple(row) for row in _r]
            except Exception:
                pass
        self.data = pd.DataFrame(_df_dict)
    
    def _compute_neighbors(self):
        """Compute neighbor indices for each point using KD-tree."""
        self.neighbors = []
        for pos in self.positions:
            neighbor_indices = self._kdtree.query_ball_point(pos, self.neighbor_radius)
            # Pad to fixed length for compatibility
            padded = np.full(6, -1, dtype=np.int64)
            for i, idx in enumerate(neighbor_indices[:6]):
                padded[i] = idx
            self.neighbors.append(padded)
        self.neighbors = np.array(self.neighbors)
    
    def _compute_regions(self):
        """Compute connected regions using DFS."""
        visited = np.zeros(self.nm, dtype=int)
        count = 0
        
        while not visited.all():
            count += 1
            stack = [np.where(visited == 0)[0][0]]
            
            while stack:
                i = stack.pop()
                visited[i] = count
                
                for nn in self.neighbors[i]:
                    if nn >= 0 and not visited[nn]:
                        stack.append(nn)
        
        self.regions = visited - 1
        self.nregions = count


class BioMaps(Maps):
    """
    Optimized Maps for aperiodic protein systems.
    
    Key features:
    - No periodic boundary conditions
    - KD-tree based neighbor search
    - Vectorized distance calculations
    - GPU acceleration for large proteins
    - Streaming for very large systems (>10k atoms)
    """
    
    def __init__(
        self,
        system: BioSystem,
        symmetryfunctions: List[SymmetryFunction],
        contactspace: Optional[BioContactSpace] = None,
        batch_size: int = 500,
        use_gpu: Optional[bool] = None,
        vectorized: bool = True,
        use_kdtree: bool = True,
        streaming_threshold: int = 10000
    ):
        """
        Initialize BioMaps for protein systems.
        
        Args:
            system: BioSystem with protein structure
            symmetryfunctions: List of symmetry functions
            contactspace: BioContactSpace (optional)
            batch_size: Batch size for processing
            use_gpu: Force GPU usage (None = auto-detect)
            vectorized: Use vectorized calculations
            use_kdtree: Use KD-tree for neighbor search
            streaming_threshold: Atom count threshold for streaming mode
        """
        # Convert to BioSystem if needed
        if not isinstance(system, BioSystem):
            system = BioSystem(
                system.grid, 
                system.atoms, 
                system.electrons,
                system.dimension,
                system.axis
            )
        
        super().__init__(
            system, 
            symmetryfunctions, 
            contactspace,
            batch_size=batch_size,
            use_gpu=use_gpu,
            vectorized=vectorized
        )
        
        self.use_kdtree = use_kdtree
        self.streaming_threshold = streaming_threshold
        self.is_streaming = len(system.atoms) > streaming_threshold
        
        if self.is_streaming:
            warnings.warn(
                f"Protein has {len(system.atoms)} atoms. "
                "Using streaming mode for memory efficiency."
            )
    
    def _process_chunk(self, chunk: npt.NDArray) -> List:
        """
        Optimized chunk processing for aperiodic systems.
        
        Uses KD-tree for efficient neighbor finding instead of
        computing all distances.
        """
        chunk_results = [[] for _ in range(self.nsfs)]
        
        for position in chunk:
            if self.hasatomicsf:
                if self.use_kdtree and hasattr(self.system, '_kdtree'):
                    # Use KD-tree for neighbor search
                    atdist, atvect = self._get_distances_kdtree(position)
                else:
                    # Fall back to standard distance calculation (no PBC)
                    atvect = self.system.atoms.positions - position
                    atdist = np.linalg.norm(atvect, axis=1)
            
            for i, sf in enumerate(self.symmetryfunctions):
                if sf.atomic:
                    if sf.angular:
                        values = sf.values(atdist, atvect)
                    else:
                        values = sf.values(atdist)
                    chunk_results[i].append(values)
        
        return chunk_results
    
    def _get_distances_kdtree(
        self, 
        position: npt.NDArray
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        """
        Get distances and vectors using KD-tree.
        
        Args:
            position: Query position
            
        Returns:
            (distances, vectors) for atoms within cutoff
        """
        # Get atoms within cutoff radius
        rcut = max(sf.rcut for sf in self.symmetryfunctions if hasattr(sf, 'rcut'))
        
        indices = self.system.query_ball_point(position, rcut)
        
        if len(indices) == 0:
            return np.array([]), np.array([])
        
        # Compute vectors and distances
        vectors = self.system.atoms.positions[indices] - position
        distances = np.linalg.norm(vectors, axis=1)
        
        return distances, vectors
    
    def _process_vectorized(self, positions: npt.NDArray) -> List:
        """
        Vectorized processing optimized for aperiodic systems.
        
        Uses batched KD-tree queries for efficiency.
        """
        xp = get_backend(use_gpu=False)
        n_positions = len(positions)
        results = [[] for _ in range(self.nsfs)]
        
        # Process in batches
        for batch_start in range(0, n_positions, self.batch_size):
            batch_end = min(batch_start + self.batch_size, n_positions)
            batch_positions = positions[batch_start:batch_end]
            
            # Compute for batch
            for pos in batch_positions:
                if self.hasatomicsf:
                    if self.use_kdtree:
                        atdist, atvect = self._get_distances_kdtree(pos)
                    else:
                        atvect = self.system.atoms.positions - pos
                        atdist = np.linalg.norm(atvect, axis=1)
                
                for i, sf in enumerate(self.symmetryfunctions):
                    if sf.atomic:
                        if sf.angular:
                            values = sf.values(atdist, atvect)
                        else:
                            values = sf.values(atdist)
                        results[i].append(values)
        
        return results
    
    def _process_streaming(self, positions: npt.NDArray) -> List:
        """
        Streaming mode for very large proteins.
        
        Processes atoms in chunks to limit memory usage.
        """
        n_positions = len(positions)
        results = [[] for _ in range(self.nsfs)]
        
        # Atom chunk size for streaming
        atom_chunk_size = 5000
        n_atoms = len(self.system.atoms)
        
        for pos_idx, position in enumerate(positions):
            pos_results = [None] * self.nsfs
            
            # Process atoms in chunks
            for atom_start in range(0, n_atoms, atom_chunk_size):
                atom_end = min(atom_start + atom_chunk_size, n_atoms)
                atom_chunk = slice(atom_start, atom_end)
                
                # Compute distances for this atom chunk
                vectors = self.system.atoms.positions[atom_chunk] - position
                distances = np.linalg.norm(vectors, axis=1)
                
                # Accumulate symmetry function contributions
                for i, sf in enumerate(self.symmetryfunctions):
                    if sf.atomic:
                        if sf.angular:
                            chunk_values = sf.values(distances, vectors)
                        else:
                            chunk_values = sf.values(distances)
                        
                        if pos_results[i] is None:
                            pos_results[i] = chunk_values
                        else:
                            pos_results[i] += chunk_values
            
            # Store results for this position
            for i in range(self.nsfs):
                results[i].append(pos_results[i])
            
            # Progress indicator for long runs
            if (pos_idx + 1) % 1000 == 0 and self.verbosity > 0:
                print(f"[BioMaps] Processed {pos_idx + 1}/{n_positions} positions")
        
        return results
    
    def atpoints(self, positions: npt.NDArray) -> pd.DataFrame:
        """
        Compute symmetry functions with automatic mode selection.
        
        Selects between:
        - Streaming mode for very large proteins
        - GPU processing for large systems
        - Vectorized CPU for medium systems
        - Parallel CPU for small systems
        """
        n_positions = len(positions)
        n_atoms = len(self.system.atoms)
        
        # Use streaming for very large systems
        if self.is_streaming and n_atoms > self.streaming_threshold:
            if self.verbosity > 0:
                print(f"[BioMaps] Using streaming mode for {n_atoms} atoms")
            results = self._process_streaming(positions)
        # Use GPU for large systems
        elif self.use_gpu and GPU_AVAILABLE and n_positions > self.batch_size:
            results = self._process_gpu_batches(positions)
        # Use vectorized CPU
        elif self.vectorized and n_positions > 100:
            results = self._process_vectorized(positions)
        # Use parallel CPU
        else:
            from mapsy.utils import full2chunk, multiproc
            args = full2chunk(positions)
            chunk_results = multiproc(self._process_chunk, args)
            results = chunk2full(chunk_results, self.nsfs)
        
        return self._results2df(positions, results)


def create_protein_contact_space(
    system: BioSystem,
    distance: float = 3.5,
    spread: float = 1.0,
    grid_resolution: float = 0.5,
    threshold: float = 0.1
) -> BioContactSpace:
    """
    Create contact space around a protein.
    
    Args:
        system: BioSystem with protein structure
        distance: Distance from protein surface
        spread: Spread of the boundary function
        grid_resolution: Grid spacing in Angstroms
        threshold: Threshold for selecting contact points
        
    Returns:
        BioContactSpace object
    """
    from mapsy.data import Grid
    
    # Get protein bounding box
    positions = system.atoms.positions
    min_coords = positions.min(axis=0) - distance - spread * 3
    max_coords = positions.max(axis=0) + distance + spread * 3
    
    # Create grid
    box_size = max_coords - min_coords
    scalars = np.ceil(box_size / grid_resolution).astype(int)
    cell = np.diag(box_size)
    
    grid = Grid(scalars=scalars, cell=cell, origin=min_coords)
    
    # Generate boundary function (simplified for proteins)
    # For each grid point, compute distance to nearest atom
    from scipy.spatial.distance import cdist
    
    grid_points = grid.coordinates.reshape(3, -1).T
    
    # Compute minimum distance to any atom for each grid point
    # Use chunked computation for memory efficiency
    chunk_size = 10000
    min_distances = []
    
    for i in range(0, len(grid_points), chunk_size):
        chunk = grid_points[i:i + chunk_size]
        distances = cdist(chunk, positions)
        min_distances.extend(distances.min(axis=1))
    
    min_distances = np.array(min_distances)
    
    # Create boundary function (smooth step at distance)
    boundary = 0.5 * (1 + np.tanh((min_distances - distance) / spread))
    boundary = boundary.reshape(grid.scalars)
    
    # Compute gradient magnitude
    from numpy import gradient
    grad = np.array(np.gradient(boundary))
    grad_magnitude = np.sqrt(np.sum(grad**2, axis=0))
    
    # Select contact space points (high gradient)
    if threshold < 0:
        threshold = np.max(grad_magnitude) * 0.9
    
    mask = grad_magnitude > threshold
    contact_indices = np.where(mask.reshape(-1))[0]
    
    contact_positions = grid_points[contact_indices]
    contact_probs = grad_magnitude.reshape(-1)[contact_indices]
    
    return BioContactSpace(
        positions=contact_positions,
        probabilities=contact_probs,
        grid=grid,
        neighbor_radius=grid_resolution * 2
    )
