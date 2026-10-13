"""Notebook-derived physicochemical fields, not a force-field calculation.

Charge lookup: user's original truncated heavy-atom table with generic backbone
fallback; unknown entries are zero. No hydrogen/protonation/terminal correction.
Electrostatic is sum(q_i / max(r_i,0.1 Angstrom)) for r_i <= 8 Angstrom,
not a calibrated electrostatic potential or energy. Hydrophobicity is the
atom-weighted inverse-distance average of residue Kyte-Doolittle values within
5 Angstrom. Both are aperiodic; original notebook formulas are preserved.
"""
from pathlib import Path
import numpy as np
KD_SCALE = {'ALA': 1.8,
    'ARG': -4.5,
    'ASN': -3.5,
    'ASP': -3.5,
    'CYS': 2.5,
    'GLN': -3.5,
    'GLU': -3.5,
    'GLY': -0.4,
    'HIS': -3.2,
    'ILE': 4.5,
    'LEU': 3.8,
    'LYS': -3.9,
    'MET': 1.9,
    'PHE': 2.8,
    'PRO': -1.6,
    'SER': -0.8,
    'THR': -0.7,
    'TRP': -0.9,
    'TYR': -1.3,
    'VAL': 4.2}
PARTIAL_CHARGES = {('*',
        'N'): -0.4157,
    ('*',
        'H'): 0.2719,
    ('*',
        'CA'): 0.0337,
    ('*',
        'HA'): 0.0823,
    ('*',
        'C'): 0.5973,
    ('*',
        'O'): -0.5679,
    ('ALA',
        'CB'): -0.1825,
    ('ARG',
        'CB'): -0.0007,
    ('ARG',
        'CG'): 0.039,
    ('ARG',
        'CD'): 0.0486,
    ('ARG',
        'NE'): -0.5295,
    ('ARG',
        'CZ'): 0.8076,
    ('ARG',
        'NH1'): -0.8627,
    ('ARG',
        'NH2'): -0.8627,
    ('ASP',
        'CB'): -0.0303,
    ('ASP',
        'CG'): 0.7994,
    ('ASP',
        'OD1'): -0.8014,
    ('ASP',
        'OD2'): -0.8014,
    ('GLU',
        'CB'): -0.0642,
    ('GLU',
        'CG'): 0.0136,
    ('GLU',
        'CD'): 0.8054,
    ('GLU',
        'OE1'): -0.8188,
    ('GLU',
        'OE2'): -0.8188,
    ('LYS',
        'CB'): -0.0094,
    ('LYS',
        'CG'): 0.0187,
    ('LYS',
        'CD'): 0.0271,
    ('LYS',
        'CE'): 0.0994,
    ('LYS',
        'NZ'): -0.3854,
    ('HIS',
        'CB'): -0.0414,
    ('HIS',
        'CG'): 0.188,
    ('HIS',
        'ND1'): -0.5432,
    ('HIS',
        'CD2'): 0.1046,
    ('HIS',
        'CE1'): 0.2057,
    ('HIS',
        'NE2'): -0.2786,
    ('PHE',
        'CB'): -0.0343,
    ('PHE',
        'CG'): 0.0118,
    ('PHE',
        'CD1'): -0.1256,
    ('PHE',
        'CD2'): -0.1256,
    ('PHE',
        'CE1'): -0.1704,
    ('PHE',
        'CE2'): -0.1704,
    ('PHE',
        'CZ'): -0.1072,
    ('TYR',
        'CB'): -0.0152,
    ('TYR',
        'CG'): -0.0011,
    ('TYR',
        'CD1'): -0.1906,
    ('TYR',
        'CD2'): -0.1906,
    ('TYR',
        'CE1'): -0.2341,
    ('TYR',
        'CE2'): -0.2341,
    ('TYR',
        'CZ'): 0.3226,
    ('TYR',
        'OH'): -0.5579,
    ('SER',
        'CB'): 0.2117,
    ('SER',
        'OG'): -0.6546,
    ('THR',
        'CB'): 0.3654,
    ('THR',
        'OG1'): -0.6761,
    ('THR',
        'CG2'): -0.2438,
    ('CYS',
        'CB'): -0.1231,
    ('CYS',
        'SG'): -0.3119,
    ('MET',
        'CB'): -0.0269,
    ('MET',
        'CG'): 0.1024,
    ('MET',
        'SD'): -0.2737,
    ('MET',
        'CE'): -0.0536,
    ('ASN',
        'CB'): -0.2041,
    ('ASN',
        'CG'): 0.713,
    ('ASN',
        'OD1'): -0.5931,
    ('ASN',
        'ND2'): -0.9191,
    ('GLN',
        'CB'): -0.0536,
    ('GLN',
        'CG'): 0.0651,
    ('GLN',
        'CD'): 0.6951,
    ('GLN',
        'OE1'): -0.6086,
    ('GLN',
        'NE2'): -0.9407,
    ('TRP',
        'CB'): -0.005,
    ('TRP',
        'CG'): -0.1415,
    ('TRP',
        'CD1'): -0.1638,
    ('TRP',
        'CD2'): 0.1243,
    ('TRP',
        'NE1'): -0.3418,
    ('TRP',
        'CE2'): 0.138,
    ('TRP',
        'CE3'): -0.2387,
    ('TRP',
        'CZ2'): 0.0046,
    ('TRP',
        'CZ3'): -0.1961,
    ('TRP',
        'CH2'): -0.1134,
    ('ILE',
        'CB'): 0.1303,
    ('ILE',
        'CG1'): -0.043,
    ('ILE',
        'CG2'): -0.3204,
    ('ILE',
        'CD1'): -0.066,
    ('LEU',
        'CB'): -0.1102,
    ('LEU',
        'CG'): 0.3531,
    ('LEU',
        'CD1'): -0.4121,
    ('LEU',
        'CD2'): -0.4121,
    ('VAL',
        'CB'): 0.2985,
    ('VAL',
        'CG1'): -0.3192,
    ('VAL',
        'CG2'): -0.3192,
    ('PRO',
        'CB'): -0.007,
    ('PRO',
        'CG'): 0.0189,
    ('PRO',
        'CD'): 0.0192}

def atom_properties(atoms, pdb):
    records = []
    started = False
    for line in Path(pdb).read_text().splitlines():
        if line.startswith('MODEL'):
            if started:
                break
            started = True
        if line.startswith('ENDMDL'):
            break
        if line.startswith(('ATOM  ', 'HETATM')):
            records.append((line[17:20].strip(),
                    line[12:16].strip(),
                    [float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54])]))
    positions = np.array([r[2] for r in records], dtype=float)
    if positions.shape != atoms.positions.shape or not np.allclose(positions,
        atoms.positions,
        rtol=0,
        atol=1e-06):
        raise ValueError('PDB atom records do not match ASE positions/order. Prepare one matching atom selection before assigning fields.')
    charges = []
    hydro = []
    unknown = []
    fallback = []
    for residue, name, _ in records:
        key = (residue, name)
        if key in PARTIAL_CHARGES:
            q = PARTIAL_CHARGES[key]
        elif ('*', name) in PARTIAL_CHARGES:
            q = PARTIAL_CHARGES['*', name]
            fallback.append(key)
        else:
            q = 0.0
            unknown.append(key)
        charges.append(q)
        hydro.append(KD_SCALE.get(residue, 0.0))
    metadata = dict(atom_records=len(records),
        alignment='Exact PDB record order and coordinates checked against ASE, model 0',
        generic_backbone_atoms=len(fallback),
        unknown_charge_atoms=len(unknown),
        unknown_charge_keys=sorted(set(unknown)),
        unknown_hydrophobic_residues=sorted({r for r,
                n,
                p in records if r not in KD_SCALE}),
        total_lookup_charge=float(np.sum(charges)),
        charge_model='Original notebook partial lookup; heavy atoms only; not a complete ff14SB assignment')
    return (np.array(charges), np.array(hydro), metadata)

def fields(atom_positions,
    charges,
    hydrophobicities,
    points,
    *,
    r_cut_elec=8.0,
    r_cut_hydro=5.0,
    batch_size=256):
    """Batched vectorization of the original sum/IDW formulas; float64."""
    positions = np.asarray(atom_positions, dtype=float)
    points = np.asarray(points, dtype=float)
    if batch_size < 1 or min(r_cut_elec, r_cut_hydro) <= 0:
        raise ValueError('Positive batch size and cutoffs required')
    electrostatic = np.zeros(len(points))
    hydrophobicity = np.zeros(len(points))
    for start in range(0, len(points), batch_size):
        distances = np.linalg.norm(positions[None,
                :,
                :] - points[start:start + batch_size,
                None,
                :],
            axis=-1)
        inv = 1 / np.maximum(distances, 0.1)
        ew = np.where(distances <= r_cut_elec, inv, 0)
        hw = np.where(distances <= r_cut_hydro, inv, 0)
        electrostatic[start:start + batch_size] = ew @ charges
        denominator = hw.sum(axis=1)
        hydrophobicity[start:start + batch_size] = np.divide(hw @ hydrophobicities,
            denominator,
            out=np.zeros(len(denominator)),
            where=denominator > 0)
    return (electrostatic, hydrophobicity)

def add_fields(data, atoms, pdb):
    charges, hydro, meta = atom_properties(atoms, pdb)
    xyz = data[['x', 'y', 'z']].to_numpy(float)
    elec, hyd = fields(atoms.positions, charges, hydro, xyz)
    data = data.copy()
    data['electrostatic'] = elec
    data['hydrophobicity'] = hyd
    meta.update(electrostatic_cutoff_angstrom=8.0,
        hydrophobicity_cutoff_angstrom=5.0,
        distance_floor_angstrom=0.1,
        electrostatic_units='lookup charge / Angstrom',
        hydrophobicity_units='Kyte-Doolittle scale',
        electrostatic_std=float(elec.std()),
        hydrophobicity_std=float(hyd.std()))
    return (data, meta)
