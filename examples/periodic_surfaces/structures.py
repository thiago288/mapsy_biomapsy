"""Deterministic ASE fixtures for the public periodic tutorials."""
import numpy as np
from ase import Atoms
from ase.build import fcc111, fcc100, fcc110

def make_surface(face='111', size=None):
    """ASE-built ideal faces or the notebook's vacancy/O decoration recipe."""
    size = tuple(size or ((20, 20, 5) if face == 'defects' else (4, 4, 3)))
    build = {'111': fcc111, '100': fcc100, '110': fcc110, 'defects': fcc111}[str(face)]
    slab = build('Pt', size=size, a=3.94, orthogonal=True, periodic=True, vacuum=10)
    if face != 'defects':
        return slab
    p = slab.positions.copy()
    Lx, Ly = (slab.cell[0, 0], slab.cell[1, 1])
    top = np.sort(np.unique(np.round(p[:, 2], 1)))[-1]
    idx = np.where(abs(p[:, 2] - top) < 0.3)[0]
    s = p[idx]

    def circle(c, r):
        return np.linalg.norm(s[:, :2] - c, axis=1) < r

    def rect(x0, x1, y0, y1):
        return (s[:, 0] > x0) & (s[:, 0] < x1) & (s[:, 1] > y0) & (s[:, 1] < y1)
    vac = circle([0.25 * Lx, 0.25 * Ly], 0.06 * Lx) | circle([0.25 * Lx, 0.75 * Ly], 0.05 * Lx)
    ox = rect(0.65 * Lx,
        0.85 * Lx,
        0.2 * Ly,
        0.45 * Ly) | rect(0.6 * Lx,
        0.8 * Lx,
        0.6 * Ly,
        0.8 * Ly)
    keep = np.ones(len(slab), dtype=bool)
    keep[idx[vac]] = False
    base = slab[keep]
    opos = s[ox].copy()
    opos[:, 2] += 1.8
    return Atoms(list(base.symbols) + ['O'] * len(opos),
        positions=np.vstack([base.positions,
                opos]),
        cell=slab.cell,
        pbc=slab.pbc)
