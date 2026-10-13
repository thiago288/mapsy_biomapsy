import numpy as np

from mapsy.data import Grid, System, ScalarField, GradientField
from mapsy.utils.functions import ERFC
from mapsy.boundaries import Boundary


class SystemBoundary(Boundary):
    """docstring"""

    def __init__(
        self,
        mode: str,
        grid: Grid,
        distance: float,
        spread: float,
        system: System,
        label: str = "",
        reference: str = "center_of_mass",
    ) -> None:

        super().__init__(
            mode,
            grid,
            label,
        )

        self.system = system
        self.grid.pbc = system.atoms.pbc.copy()

        com = self.system.atoms.get_center_of_mass()
        width = distance
        if reference not in ("center_of_mass", "surface"):
            raise ValueError("reference must be center_of_mass or surface")
        if reference == "surface":
            if system.dimension != 2:
                raise ValueError("surface reference requires a two-dimensional system")
            if not len(system.atoms):
                raise ValueError("surface reference requires atoms")
            coordinates = system.atoms.positions[:, system.axis]
            lower, upper = float(np.min(coordinates)), float(np.max(coordinates))
            com[system.axis] = (lower + upper) / 2.0
            width = (upper - lower) / 2.0 + distance
            # A planar MIC envelope must fit within its normal repeat length.
            # Slabs crossing the cell seam must be unwrapped before this call.
            if self.grid.pbc[system.axis] and width >= self.grid.cell[system.axis, system.axis] / 2.0:
                raise ValueError("Surface envelope exceeds half the periodic normal cell length; add vacuum or unwrap the slab")
        self.reference = reference
        self.reference_position = com.copy()
        self.reference_width = width

        self.simple = ERFC(
            grid=self.grid,
            kind=3,
            dim=system.dimension,
            axis=system.axis,
            width=width,
            spread=spread,
            volume=1.0,
            pos=com,
        )

    def update(self) -> None:
        """docstring"""

        self._build()
        self._update_solvent_aware_boundary()

    def _build(self) -> None:
        """docstring"""

        self.switch[:] = self.simple.density
        self.gradient[:] = self.simple.gradient
