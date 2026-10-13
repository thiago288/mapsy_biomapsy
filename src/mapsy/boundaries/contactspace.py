"""
ContactSpace — contact space extraction from a boundary.

Changes from original:
- _get_indexes: replaced the O(N) Python loop (which also reshapes mask
  every iteration) with pure NumPy operations. All other logic is
  preserved exactly to maintain mathematical equivalence.
"""

import numpy as np
import numpy.typing as npt
import pandas as pd

from mapsy.boundaries import Boundary


class ContactSpace:

    # 6 face-neighbour offsets for a 3-D grid
    nn = np.zeros((6, 3), dtype=int)
    nn[0], nn[2], nn[4] = np.eye(3, dtype=int)
    nn[1], nn[3], nn[5] = -np.eye(3, dtype=int)

    def __init__(
        self,
        boundary: Boundary,
        tol: float = 0.1,
        epsilon: float = 0.0001,
    ) -> None:
        self.boundary = boundary
        self.grid     = boundary.grid

        # tol < 0  → only the top-gradient shell
        if tol < 0.0:
            tol = np.max(boundary.gradient.modulus) - epsilon

        self.mask = boundary.gradient.modulus > tol
        self.norm = np.sum(boundary.gradient.modulus[self.mask])

        self._get_indexes()
        self._get_neighbors()
        self._get_regions()

        self.data = pd.DataFrame(
            {
                "probability": boundary.gradient.modulus[self.mask],
                "x": self.grid.coordinates[0, self.mask],
                "y": self.grid.coordinates[1, self.mask],
                "z": self.grid.coordinates[2, self.mask],
                "nn": self.neighbors,
                "region": self.regions,
            }
        )
    # Index maps  (Bug A fix: O(N) Python loop → pure NumPy)
    def _get_indexes(self) -> None:
        """
        Build the forward (i2m) and reverse (m2i) maps between the flat grid
        index space and the contact-space index space.

        Original had a Python for-loop that also called mask.reshape(-1, 1)
        on every iteration.  Replaced with np.where + np.unravel_index.
        """
        self.ni = self.grid.ndata
        self.nm = int(np.count_nonzero(self.mask))

        # Flat indices of contact-space points in the full grid
        flat_cs = np.where(self.mask.reshape(-1))[0]           # (nm,)

        # m2i[flat_grid_idx] = cs_idx  (or -1 if not in CS)
        m2i_flat = -np.ones(self.ni, dtype=np.int64)
        m2i_flat[flat_cs] = np.arange(self.nm, dtype=np.int64)

        # Reshape m2i to the 3-D grid shape for neighbour lookup
        self.m2i = m2i_flat.reshape(self.grid.scalars)         # (nx, ny, nz)

        # i2m[cs_idx] = (ix, iy, iz) multi-index in the grid
        self.i2m = np.array(
            np.unravel_index(flat_cs, self.grid.scalars)
        ).T                                                      # (nm, 3)
    # Neighbour connectivity (unchanged)
    def _get_neighbors(self) -> None:
        """For each CS point, find the CS indices of its 6 grid neighbours."""
        nearest = self.i2m[:, None, :] + self.nn[None, :, :]
        pbc = getattr(self.grid, 'pbc', np.ones(3, dtype=bool))
        valid = np.all((nearest >= 0) & (nearest < self.grid.scalars) | pbc, axis=-1)
        nearest %= self.grid.scalars
        neighbors = self.m2i[tuple(nearest.transpose(2, 0, 1))]
        neighbors[~valid] = -1
        self.neighbors = list(neighbors)
    # Connected regions (unchanged)
    def _get_regions(self) -> None:
        """Depth-first flood fill to label connected components."""
        visited = np.zeros(self.nm, dtype=int)
        count   = 0

        while True:
            if visited.all():
                break
            count += 1
            stack = [np.where(visited == 0)[0][0]]
            while stack:
                i = stack.pop()
                visited[i] = count
                for nn in self.neighbors[i]:
                    if nn >= 0 and visited[nn] == 0:
                        stack.append(int(nn))

        self.regions  = visited - 1          # zero-based
        self.nregions = int(np.max(visited)) if self.nm else 0
