## ASE databases

### Database Structure

Every database in this repository has the same structure:

- Name (string) — Unique molecule identifier (e.g. BX-6-NMe2)
- Aromatic_Scaffold (string) — Scaffold classification label used for identification (e.g. BX-6)
- MWFN_MBIS_Atomic_Charges (Natoms), $e$ — MBIS partial charges on each atom calculated by MWFN
- MWFN_MBIS_Atom_Dipole (Natoms × 3), $a_0e$ — Cartesian atomic dipole moments from MBIS partitioning calculated by MWFN
- MWFN_MBIS_Atom_Quadrupole (Natoms × 5), $a_0^2e$ — Atomic quadrupole components in spherical harmonics from MBIS partitioning calculated by MWFN
- MWFN_MBIS_r2 (Natoms × 3), $a_0^2$ — Electronic spatial extent ⟨x2⟩, ⟨y2⟩, ⟨z2⟩ per atom
- MWFN_MBIS_c6 (Natoms), $E_h a_0^6$ — Atomic C6 dispersion coefficients
- MWFN_MBIS_Atomic_Polarizability (Natoms × 2), $a_0^3$ — First element is the isotropic atomicpolarizability from MBIS partitioning calculated by MWFN and the second element is each atom’s contribution to the total molecular polarizability
- MWFN_MBIS_Iodine_Vmin_Vmax (iodine × 2), $E_h/e$ — The VS,min and VS,max respectively ofiodine from MBIS partitioning calculated by MWFN
- ORCA_Mol_Dipole (molecule × 3), $a_0e$ — Cartesian molecular dipole moments calculated by ORCA
- ORCA_Atom_Dipole (Natoms × 3), $a_0e$ — Cartesian atomic dipole moments from ORCA_int
- ORCA_Mol_Quadrupole (molecule × 6), $a_0^2e$ — Molecular quadrupole tensor components (xx, yy, zz, xy, xz, yz) from ORCA
- ORCA_Atomic_Quadrupole (Natoms × 5), $a_0^2e$ — Atomic quadrupole components in spherical harmonics from ORCA (Converted from traceless tensor to allow for comparison with MWFN_MBIS values)
- ORCA_Atomic_Polarizability (Natoms), $a_0^3$ — Isotropic atomic polarizability as calculated by ORCA
- ORCA_MBIS_Atomic_Charges (Natoms), $e$ — MBIS partial charges on each atom calculated by ORCA
- ORCA_MBIS_Atom_Dipole (Natoms × 3), $a_0e$ — Cartesian atomic dipole moments from MBIS partitioning calculated by ORCA
- ORCA_MBIS_Atom_Quadrupole (Natoms × 5), $a_0^2e$ — Atomic quadrupole components in spherical harmonics from MBIS partitioning calculated by ORCA
- Atoms — Final optimised geometry, comprising of:
  - Atomic symbols — (Natoms), string
  - Cartesian positions — (Natoms × 3), $Å$
  - Atomic numbers — (Natoms), int

### mol.db

Contains all 96 isolated scaffold structures and their values

### acceptor.db

Contains the halogen bonding acceptors (F-, Cl-, Br-, I-, Formaldehyde, Benzene, $\mathrm{NH}_3$, $\mathrm{PH}_3$ and $\mathrm{NF}_3$) for the acceptor molecule input parameters

### X-molecules.db - where X = halogen bond acceptor

Contains all the interaction complexes formed by the scaffold and X, for each complex, the interaction energy is stored alongside the above values
