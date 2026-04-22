import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from ase import Atoms
from ase.db import connect
from opi.input.structures.structure import Structure

from utils import (
    MwfnFuzzySpaceOut,
    add_orca_quad_sphm,
    align_ci_xaxis,
    create_mwfn_input,
    dump_mwfn_fuzzy_results,
    get_el_prop_from_prop_file,
    get_elem_list,
    get_iodine_mbis,
    get_mbis_from_prop_file,
    get_opt_calculator,
    get_sp_calculator,
    run_mwfn,
    run_opt_calc,
    run_sp_calc,
    cartesian_to_spherical_quadrupole,
    get_struct_from_smiles,
    charge_mult_from_smiles,
    attach_fragment_most_polar_bond_along_CI_axis,
    atoms_to_structure
)

# create CLI parser
parser = argparse.ArgumentParser()

parser.add_argument(
    "-c", "--cores", help="Number of cores", required=False, type=int, default=1
)
parser.add_argument(
    "-wrkd", "--working_dir", help="Path of working directory", required=True, type=str
)
parser.add_argument(
    "-subd",
    "--submission_dir",
    help="Path of submission directory",
    required=True,
    type=str,
)
parser.add_argument(
    "--atom_refs",
    help="Path to atomic reference calculations for each element",
    required=True,
    type=str,
)
args = parser.parse_args()

df = pd.read_csv("SMILES.csv")

# dict holding electrostatic properties for iodine
iodine_mbis_dict = dict()
db = connect("molecules.db")  # Create ASE SQLite Database
ref_db = connect("mol.db")

for row in ref_db.select():
    initial_atoms=row.toatoms()
    aromatic = row.Aromtic_Scaffold
    keys =row.Name
    smi=row.SMILES
    E_mol = row.E_tot
    q, mult = charge_mult_from_smiles(smi)
    mol_struct = atoms_to_structure(
            ase_atoms = initial_atoms,
            smiles = smi,
            charge = q,
            mult = mult,
    )
    # make directory for each mol
    calc_space_name = Path(args.working_dir + f"/{keys}")
    if not calc_space_name.exists():
        os.makedirs(calc_space_name)
    
    print("Adding fragment")
    #Add anion and optimize interaction complex
    complex_struct = attach_fragment_most_polar_bond_along_CI_axis(
            opt_struct = mol_struct,
            frag_smiles = "N",
            distance = 3,
    )
    calculator_opt = get_opt_calculator(
        basename=f"opt",
        structure=complex_struct,
        cores=args.cores,
        working_dir=calc_space_name,
    )
    output_opt, final_struct = run_opt_calc(calc=calculator_opt)
    final_struct, iodine_index = align_ci_xaxis(final_struct, return_i_index=True)
    
    #SP calc on interaction complex
    final_struct.charge = complex_struct.charge
    final_struct.multiplicity = complex_struct.multiplicity
    calculator_sp_comp = get_sp_calculator(
        basename=f"sp_comp",
        structure=final_struct,
        cores=args.cores,
        working_dir=calc_space_name,
    )
    output_sp_comp = run_sp_calc(calc=calculator_sp_comp)
    prop_file_comp = calculator_sp_comp.working_dir / f"{calculator_sp_comp.basename}.property.json"
    with open(prop_file_comp, "r") as f:
        prop_data_comp = json.load(f)
    el_prop_comp = get_el_prop_from_prop_file(prop_data_comp)
    E_tot = float(prop_data_comp["Geometries"][0]["Single_Point_Data"]["FinalEnergy"])

    gbw_basename = f"sp_comp"
    subprocess.run(
        ["orca_2mkl", gbw_basename, "-molden"],
        cwd=calc_space_name,
        check=True,
    )
    molden_main = calc_space_name / f"{gbw_basename}.molden.input"

    # extract mbis properties from property files
    mbis_prop = get_mbis_from_prop_file(output_sp_comp.property_json_data)
    mbis_prop = add_orca_quad_sphm(mbis_prop)
    # extract atomic electrostatic properties for iodine
    mbis_iodine = get_iodine_mbis(mbis_props=mbis_prop, index=iodine_index)
    # populate dict
    iodine_mbis_dict[keys] = mbis_iodine
    # MWFN Calc
    mwfn_input = create_mwfn_input(
        directory=calc_space_name,
        atom_refs_base=args.atom_refs,
        elements=get_elem_list(final_struct),
        mbis=True,
        becke=False,
    )
    run_mwfn(mwfn_input, molden_main)
    mw = MwfnFuzzySpaceOut(calculator_sp_comp.working_dir / "mwfn_mbis.out")
    dump_mwfn_fuzzy_results(mw)

    # Create ASE Atom Object
    symbols = [atom.element for atom in final_struct.atoms]
    coords_angstrom = [
        [atom.element, atom.coordinates.x, atom.coordinates.y, atom.coordinates.z]
        for atom in final_struct.atoms
    ]
    positions = np.array([[x, y, z] for _, x, y, z in coords_angstrom])
    atoms = Atoms(symbols=symbols, positions=positions)

    # Add properties from ORCA
    atoms.energy = E_tot    # Total energy of complex in Hartree
    atoms.info["Molecule Energy"] = float(E_mol)
    atoms.info["ORCA_mol_dipole_sp_au"] = (
        el_prop_comp["molecular_dipole"].reshape(3).tolist()
    )
    atoms.info["ORCA_atomic_dipole"] = (
        el_prop_comp["atomic_dipole"].tolist()
    )
    atoms.info["ORCA_mol_quadrupole_sp_au"] = (
        el_prop_comp["molecular_quadrupole"].reshape(6).tolist()
    )

    atomic_quad_array = el_prop_comp["atomic_quadrupole_array"]  # np.array (natoms, 6)
    MBIS_quad_array = el_prop_comp["mbis_atomic_quadrupoles_array"]

    ORC_spherical_quads = []
    MBIS_spherical_quads = []

    for i in range(atomic_quad_array.shape[0]):
        quad_single = atomic_quad_array[i]  # (6,)
        sph = cartesian_to_spherical_quadrupole(quad_single)
        ORC_spherical_quads.append(sph)

    for i in range(MBIS_quad_array.shape[0]):
        quad_single = MBIS_quad_array[i]  # (6,)
        sph = cartesian_to_spherical_quadrupole(quad_single)
        MBIS_spherical_quads.append(sph)

    atoms.info["ORCA_atomic_quadrupole"] = (
        np.array(ORC_spherical_quads).tolist()
    )
    atoms.info["ORCA_iso_quadrupole_au"] = float(el_prop_comp["isotropic_quadrupole"])
    atoms.info["ORCA_MBIS_Atomic_Charges"] = el_prop_comp["mbis_atomic_charges"].tolist()
    atoms.info["ORCA_MBIS_r3"] = el_prop_comp["mbis_3rd_radial_moment"].tolist()
    atoms.info["ORCA_MBIS_Atomic_Dipoles"] = (el_prop_comp["mbis_atomic_dipoles"].tolist())
    atoms.info["ORCA_MBIS_Atomic_Quadrupoles"] = np.array(MBIS_spherical_quads).tolist()
    atoms.info["ORCA_MBIS_Atomic_Octupoles"] = (el_prop_comp["mbis_atomic_octupoles_array"].tolist())
    atoms.info["ORCA_mol_polarizability_iso"] = float(el_prop_comp["molecular_polarizability_iso"])
    atoms.info["ORCA_atomic_polarizability_iso"] = el_prop_comp["atomic_polarizability_iso"].tolist()

    # Add MBIS/Iodine-specific properties
    iodine_charge = float(mbis_iodine["mbis_monopole_array"][0])
    atoms.info["iodine_charge"] = iodine_charge

    # Add MWFN properties
    atoms.info["fuzzy_integrals"] = (
        mw.fuzzy_space_integral.tolist() # Convert array to list for serialization
    )

    atoms.info["charges"] = mw.charges.tolist()
    atoms.info["dipoles"] = mw.dipoles.tolist()
    atoms.info["quadrupoles"] = mw.quadrupoles.tolist()
    atoms.info["octupoles"] = mw.octopoles.tolist()
    atoms.info["r_2"] = mw.r_2.tolist()
    atoms.info["Overlap_matrix"] = mw.overlap_matrix.tolist()
    atoms.info["c6"] = mw.c6.tolist()
    atoms.info["polarisabilities"] = mw.polarisabilities.tolist()
    atoms.info["effective_volume"] = mw.effective_volume.tolist()
    atoms.info["iodine_vmin_vmax"] = mw.iodine_vmin_vmax.tolist()
    atoms.info["molecule_name"] = keys  # Metadata for easy querying

    charges_json = json.dumps(atoms.info["charges"])
    dipoles_json = json.dumps(atoms.info["dipoles"])
    quad_json = json.dumps(atoms.info["quadrupoles"])
    oct_json = json.dumps(atoms.info["octupoles"])
    r2_json = json.dumps(atoms.info["r_2"])
    overlap_json = json.dumps(atoms.info["Overlap_matrix"])
    c6_json = json.dumps(atoms.info["c6"])
    polar_json = json.dumps(atoms.info["polarisabilities"])
    eff_vol_json = json.dumps(atoms.info["effective_volume"])
    fuzzy_json = json.dumps(atoms.info["fuzzy_integrals"])
    iodine_json = json.dumps(atoms.info["iodine_vmin_vmax"])
    orca_mol_dip = json.dumps(atoms.info["ORCA_mol_dipole_sp_au"])
    orca_atom_dip = json.dumps(atoms.info["ORCA_atomic_dipole"])
    orca_mol_quad = json.dumps(atoms.info["ORCA_mol_quadrupole_sp_au"])
    orca_iso_quad = json.dumps(atoms.info["ORCA_iso_quadrupole_au"])
    orca_atom_quad = json.dumps(atoms.info["ORCA_atomic_quadrupole"])
    orca_mol_pol = json.dumps(atoms.info["ORCA_mol_polarizability_iso"])
    orca_atom_pol = json.dumps(atoms.info["ORCA_atomic_polarizability_iso"])
    orca_mbis_charge = json.dumps(atoms.info["ORCA_MBIS_Atomic_Charges"])
    orca_mbis_r3 = json.dumps(atoms.info["ORCA_MBIS_r3"])
    orca_mbis_dip = json.dumps(atoms.info["ORCA_MBIS_Atomic_Dipoles"])
    orca_mbis_quad = json.dumps(atoms.info["ORCA_MBIS_Atomic_Quadrupoles"])
    orca_mbis_oct = json.dumps(atoms.info["ORCA_MBIS_Atomic_Octupoles"])

    db_kwargs = dict(
        Aromtic_Scaffold=aromatic,
        Name=keys,
        SMILES=smi,
        E_tot=float(atoms.energy),
        E_mol=E_mol,
        Interaction_E = E_tot - (E_mol + -114.502732918681),
        MWFN_MBIS_Atomic_Charges=charges_json,
        MWFN_MBIS_Atom_Dipole=dipoles_json,
        MWFN_MBIS_Atom_Quadrupole=quad_json,
        MWFN_MBIS_Atom_Octupole=oct_json,
        MWFN_MBIS_r2=r2_json,
        MWFN_MBIS_Overlap_Matrix=overlap_json,
        MWFN_MBIS_c6=c6_json,
        MWFN_MBIS_Atomic_Polarizability=polar_json,
        MWFN_MBIS_Effective_Volume=eff_vol_json,
        MWFN_MBIS_Fuzzy_Int=fuzzy_json,
        MWFN_MBIS_Iodine_Vmin_Vmax=iodine_json,
        ORCA_Mol_Dipole=orca_mol_dip,
        ORCA_Atom_Dipole=orca_atom_dip,
        ORCA_Mol_Quadrupole=orca_mol_quad,
        ORCA_Iso_Quadrupole=orca_iso_quad,
        ORCA_Atom_Quadrupole=orca_atom_quad,
        ORCA_Mol_Polarizability=orca_mol_pol,
        ORCA_Atom_Polarizability=orca_atom_pol,
        ORCA_MBIS_Charges=orca_mbis_charge,
        ORCA_r3=orca_mbis_r3,
        ORCA_MBIS_Atom_Dipole=orca_mbis_dip,
        ORCA_MBIS_Atom_Quadrupole=orca_mbis_quad,
        ORCA_MBIS_Atom_Octupole=orca_mbis_oct,
    )

    # Debug: print type and value for everything
    for name, val in db_kwargs.items():
        print(name, type(val), val)

    # Optionally: auto-fix scalar numeric strings
    for name, val in list(db_kwargs.items()):
        if isinstance(val, str):
            try:
                db_kwargs[name] = float(val)
            except ValueError:
                pass  # leave non-numeric strings unchanged

    # Now write
    db.write(atoms, **db_kwargs)

# write dict with properties for iodine to dictionary
with open(f"{args.submission_dir}/iodine_mbis.json", "w") as f:
    json.dump(iodine_mbis_dict, f)

