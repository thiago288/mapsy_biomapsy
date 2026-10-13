#
import numpy.typing as npt
from typing import Tuple, Optional
from itertools import product
import numpy as np


class Grid:

    def __init__(self, scalars=None, basis=None, cell=None, origin=None, pbc=True):
        """"""
        self.pbc = np.broadcast_to(np.asarray(pbc, dtype=bool), (3,)).copy()
        if scalars is not None:
            self.scalars = np.array(scalars, dtype=np.int64)
        else:
            self.scalars = np.ones(3, dtype=np.int64)
        if basis is not None:
            self.basis = np.array(basis, dtype=np.float64).T
            self.cell = self.basis * self.scalars[:, None]  # broadcasting
        elif cell is not None:
            self.cell = np.array(cell, dtype=np.float64)
            self.basis = self.cell / self.scalars[:, None]
        else:
            raise ValueError("Grid needs either basis or cell")
        if origin is not None:
            self.origin = np.asarray(origin, dtype=float)
        else:
            self.origin = np.zeros(3, dtype=np.float64)

        self.ndata = np.prod(self.scalars)

        # -- Construct the grid
        mesh: npt.NDArray = np.mgrid[
            0 : self.scalars[0], 0 : self.scalars[1], 0 : self.scalars[2]
        ]
        self.coordinates = (
            np.einsum("ij,jklm->iklm", self.basis.T, mesh)
            + self.origin[:, None, None, None]
        )

        self.corners = -np.array(list(product(range(2), repeat=3))).dot(self.cell)

        self._volume: Optional[np.float64] = None

    def get_min_distance(
        self,
        origin: npt.NDArray,
        dim: int = 0,
        axis: int = 0,
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        """docstring"""
        r = self.coordinates - origin[:, np.newaxis, np.newaxis, np.newaxis]
        r, r2 = self._apply_minimum_image_convension(r, dim, axis)
        return r, r2

    def _get_direction(
        self,
        dim: int = 0,
        axis: int = 0,
    ):
        """docstring"""
        if dim == 0:
            n = np.zeros(3)
        elif dim == 1:
            n = self.cell[axis, :]
        elif dim == 2:
            n1, n2 = self.cell[np.arange(3) != axis, :]
            n = np.cross(n2, n1)
        else:
            raise ValueError("dimensions out of range")

        norm = np.linalg.norm(n)
        if norm > 1.0e-16:
            n = n / norm
        return n

    def _reduce_dimension(
        self,
        r: npt.NDArray,
        n: npt.NDArray,
        dim: int = 0,
    ):
        """docstring"""
        if dim == 0:
            pass
        elif dim == 1:
            r = r - np.einsum("jkl,i->ijkl", np.einsum("ijkl,i->jkl", r, n), n)
        elif dim == 2:
            r = np.einsum("jkl,i->ijkl", np.einsum("ijkl,i->jkl", r, n), n)
        else:
            raise ValueError("dimensions out of range")
        return r

    def _apply_minimum_image_convension(
        self,
        r: npt.NDArray,
        dim: int = 0,
        axis: int = 0,
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        """docstring"""

        n = self._get_direction(dim, axis)

        if dim == 0:
            from mapsy.utils.gpu_utils import batch_distances_periodic
            shape = r.shape[1:]
            flat = r.reshape(3, -1).T
            rmin = np.empty_like(flat)
            r2min = np.empty(len(flat))
            for start in range(0, len(flat), 8192):
                stop = start + 8192
                distances, vectors = batch_distances_periodic(
                    np.zeros((1, 3)), flat[start:stop], self.cell, self.pbc)
                rmin[start:stop] = vectors[0]
                r2min[start:stop] = distances[0] ** 2
            return rmin.T.reshape((3,) + shape), r2min.reshape(shape)

        # Extended line/plane fields minimize projected distance. Their legacy
        # finite-corner treatment is only supported for orthogonal cells.
        gram = self.cell @ self.cell.T
        if not np.allclose(gram, np.diag(np.diag(gram)), atol=1e-12, rtol=1e-12):
            raise NotImplementedError("Skew-cell line/plane boundary fields require a projected-lattice solver")
        s = r.reshape(3, -1).T @ np.linalg.inv(self.cell)
        s[:, self.pbc] -= np.round(s[:, self.pbc])
        rmin = self._reduce_dimension((s @ self.cell).T.reshape(r.shape), n, dim)
        r2min = np.einsum("i...,i...", rmin, rmin)

        return rmin, r2min

    @property
    def volume(self) -> np.float64:
        if self._volume is None:
            self._compute_volume()
        return self._volume

    def _compute_volume(self):
        a1 = self.cell[:, 0]
        a2 = self.cell[:, 1]
        a3 = self.cell[:, 2]
        a2crossa3 = np.cross(a2, a3)
        self._volume = abs(np.dot(a1, a2crossa3))

    def reciprocal(self):

        a1 = self.cell[:, 0]
        a2 = self.cell[:, 1]
        a3 = self.cell[:, 2]
        a2crossa3 = np.cross(a2, a3)
        volume = np.dot(a1, a2crossa3)

        b1 = a2crossa3 / volume
        b2 = np.cross(a3, a1) / volume
        b3 = np.cross(a1, a2) / volume

        return np.stack([b1, b2, b3], axis=0).T
