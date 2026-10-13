# Calmodulin input provenance

**1CFD is apo (calcium-free source); 1CLL is holo-derived (calcium-bound source, ions omitted here).** Sources: [1CFD](https://www.rcsb.org/structure/1CFD), [1CLL](https://www.rcsb.org/structure/1CLL). [PDB data usage: CC0](https://www.rcsb.org/pages/usage-policy). Cite the PDB DOIs in `provenance.json`.

Files preserve the previously validated protein-only selection: standard protein-residue records, non-hydrogen atoms, original alternate conformations. ASE reads model index 0. Calcium, water and other heteroatoms were removed; no protonation or terminal preparation was applied. Exact hashes and preparation are recorded. The physiochemical helper reads these same ATOM records and checks coordinate/order alignment, including retained alternate conformers.

These are two source protein conformations, not a calcium-inclusive electrostatic comparison. Apo/holo labels indicate source-state provenance, not added chemistry in the calculation.
