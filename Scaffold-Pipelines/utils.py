import json
import os
import re
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np
from ase.data import chemical_symbols
from openbabel.pybel import readstring
from opi.core import Calculator

# from opi.execution.core import OrcaBinary, Runner
from opi.input.arbitrary_string import ArbitraryString
from opi.input.blocks.block_elprop import BlockElprop
from opi.input.simple_keywords import (
    AtomicCharge,
    BasisSet,
    Dft,
    RelativisticCorrection,
    DispersionCorrection,
    Task,
)
from opi.input.simple_keywords.grid import Grid
from opi.input.structures.structure import Atom, Structure
from opi.output.core import Output
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

def get_opt_calculator(
    basename: str,
    structure: Structure,
    cores: int,
    working_dir: Path,
    memory: int = 1024,
) -> Calculator:
    """
    Create an ORCA optimization calculator with r2scan-3c functional.

    Args:
        basename (str): Base name for calculation.
        structure (Structure): Molecular structure.
        cores (int): Number of CPU cores.
        working_dir (Path): Working directory.
        memory (int, optional): Memory in MB to allocate for calculation
        (default: 1024).

    Returns:
        Calculator: Configured ORCA calculator object.
    """
    calc = Calculator(basename=basename, working_dir=working_dir)
    calc.structure = structure
    sk_list = [Dft.R2SCAN_3C, Task.OPT]
    calc.input.add_simple_keywords(*sk_list)
    calc.input.ncores = cores
    calc.input.memory = memory
    return calc


def get_struct_from_smiles(smiles: str) -> Structure:
    """
        Obtain orca structure from smiles
        More robust implementation using obabel and GAFF

    Args:
        smiles (str): SMILES string of molecule

    Returns:
        Structure: ORCA structure
    """
    ob_mol = readstring(format="smi", string=smiles)
    ob_mol.addh()
    ob_mol.make3D(forcefield="gaff", steps=1000)
    ob_mol.write(format="xyz", filename="tmp.xyz", overwrite=True)
    orca_struct = Structure.from_xyz("tmp.xyz")
    os.remove("tmp.xyz")
    return orca_struct

def charge_mult_from_smiles(smiles: str) -> tuple[int, int]:
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)

    formal_charge = Chem.GetFormalCharge(mol)
    n_rad = sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms())

    # Very simple spin rule: improve as needed
    if n_rad == 0:
        mult = 1          # closed-shell singlet
    elif n_rad == 1:
        mult = 2          # doublet
    else:
        mult = n_rad + 1  # crude default (e.g. 2 radicals → triplet)

    return formal_charge, mult


def get_sp_calculator(
    basename: str,
    structure: Structure,
    cores: int,
    working_dir: Path,
    memory: int = 1024,
) -> Calculator:
    """Create ORCA single point calculator with wb97m-v functional
    and x2c Hamiltonian. x2ctzvppall basis set

    Args:
        basename (str): Base name for calculation.
        structure (Structure): Molecular structure
        cores (int): Number of CPU cores
        working_dir (Path): Working directory
        memory (int, optional): Max memory per core in MB. Defaults to 1024.

    Returns:
        Calculator: _description_
    """
    calc = Calculator(basename=basename, working_dir=working_dir)
    calc.structure = structure
    sk_list = [
        Dft.WB97M_V,
        DispersionCorrection.SCNL,
        BasisSet.X2C_TZVPPALL,
        RelativisticCorrection.X2C,
        Task.SP,
        AtomicCharge.MBIS,
        AtomicCharge.AIM,
        Grid.DEFGRID3,
    ]
    calc.input.add_simple_keywords(*sk_list)
    el_prop_block = BlockElprop(  # TODO Check these values, look at them, store them in ASE - ORCA partioning scheme
        dipole=True,
        quadrupole=True,
        dipoleatom=True,
        quadrupoleatom=True,
        polaratom=True,
        polar='analytic',
    )
    calc.input.add_blocks(el_prop_block)
    mbis_print = ArbitraryString("%method mbis_largeprint true end")
    chelpg_block = ArbitraryString("%chelpg dipole true end ")
    calc.input.arbitrary_strings.append(mbis_print)
    calc.input.arbitrary_strings.append(chelpg_block)
    calc.input.ncores = cores
    calc.input.memory = memory
    return calc


def write_tmp_plot_file(filename: str = "orca_plot.inp") -> str:
    """Write the input file for orca_plot to obtain density and esp cube file

    Args:
        filename (str): filename for tmp file

    Returns:
        str: filename string
    """
    # ESP is calculated first, then density
    string = """1\n43\n0\n11\n1\n2\ny\n11\n12\nEOF"""
    with open(file=filename, mode="w") as f:
        f.write(string)
    return filename


def run_opt_calc(calc: Calculator) -> tuple[Output, Structure]:
    """Run the geometry optimisation calculation

    Args:
        calc (Calculator): Opt calculator object

    Returns:
        tuple[Output, Structure]: tuple with the parsed output file
        and the final optimised structure
    """
    # > Write the ORCA input file
    calc.write_input()
    # > Run the ORCA calculation
    calc.run()
    # obtain output file
    output = calc.get_output()
    output.parse()
    # obtain coordinates of final geom from output file
    atoms = [
        Atom(element=i[0], coordinates=i[1:])
        for i in output.results_properties.geometries[
            -1
        ].geometry.coordinates.cartesians
    ]
    for atom in atoms:
        atom.coordinates.coordinates *= 0.529177  # convert to Å units!!!
    opt_struct = Structure(atoms)
    
    # propagate metadata from the input structure
    src = calc.structure
    if hasattr(src, "smiles"):
        opt_struct.smiles = src.smiles
    if hasattr(src, "charge"):
        opt_struct.charge = src.charge
    if hasattr(src, "multiplicity"):
        opt_struct.multiplicity = src.multiplicity

    return output, opt_struct

    return output, opt_struct


def run_sp_calc(calc: Calculator) -> Output:
    """Run single point calculation with single point calculator

    Args:
        calc (Calculator): Single point calculator

    Returns:
        Output: Parsed output file
    """
    # > Write the ORCA input file
    calc.write_input()
    # > Run the ORCA calculation
    calc.run()
    # TODO fix this
    # Ideally this should create the density and esp cube file
    # once the calculation is finished but it doesnt work for some reasom
    # orca_plot_runner = Runner(calc.working_dir)
    # plot_inp = calc.working_dir / "orca_plot.inp"
    # orca_plot_file = write_tmp_plot_file(filename=str(plot_inp))
    # gbw_file = calc.working_dir / f"{calc.basename}.gbw"
    # orca_plot_runner.run(OrcaBinary.ORCA_PLOT,
    # (f"{gbw_file}", f"-i < {orca_plot_file} > orca_plot.out"),
    # silent=True, cwd=calc.working_dir)
    # if Path("orca_plot.out").exists():
    # remove("orca_plot.out")
    # > Get the output object
    output = calc.get_output()
    output.parse()

    return output


# Some helper functions for handling multipole moments


def rearrange_octupole(oct_array: np.ndarray) -> np.ndarray:
    """
    Convert octupole tensor from flat rep to 3x3x3 symmetric tensors.

    Inp fmt: [XXX, YYY, ZZZ, XXY, XXZ, XYY, XYZ, XZZ, YYZ, YZZ] for each atom
    Output format: 3x3x3 symmetric tensor for each atom

    The octupole tensor is symmetric, so O_ijk = O_jik = O_ikj = O_kji = O_jki = O_kij

    Args:
        oct_array: Array of shape (n_atoms, 10) with octupole components

    Returns:
        Array of shape (n_atoms, 3, 3, 3) with symmetric octupole tensors
    """
    n_atoms = oct_array.shape[0]

    # Pre-allocate the result array
    result = np.zeros((n_atoms, 3, 3, 3))

    # Diagonal elements (XXX, YYY, ZZZ)
    result[:, 0, 0, 0] = oct_array[:, 0]  # XXX
    result[:, 1, 1, 1] = oct_array[:, 1]  # YYY
    result[:, 2, 2, 2] = oct_array[:, 2]  # ZZZ

    # Mixed terms with two identical indices
    # XXY and all its permutations
    result[:, 0, 0, 1] = result[:, 0, 1, 0] = result[:, 1, 0, 0] = oct_array[
        :, 3
    ]  # XXY

    # XXZ and all its permutations
    result[:, 0, 0, 2] = result[:, 0, 2, 0] = result[:, 2, 0, 0] = oct_array[
        :, 4
    ]  # XXZ

    # XYY and all its permutations
    result[:, 0, 1, 1] = result[:, 1, 0, 1] = result[:, 1, 1, 0] = oct_array[
        :, 5
    ]  # XYY

    # XZZ and all its permutations
    result[:, 0, 2, 2] = result[:, 2, 0, 2] = result[:, 2, 2, 0] = oct_array[
        :, 7
    ]  # XZZ

    # YYZ and all its permutations
    result[:, 1, 1, 2] = result[:, 1, 2, 1] = result[:, 2, 1, 1] = oct_array[
        :, 8
    ]  # YYZ

    # YZZ and all its permutations
    result[:, 1, 2, 2] = result[:, 2, 1, 2] = result[:, 2, 2, 1] = oct_array[
        :, 9
    ]  # YZZ

    # Mixed term with all different indices (XYZ)
    # All 6 permutations of XYZ
    result[:, 0, 1, 2] = result[:, 0, 2, 1] = result[:, 1, 0, 2] = oct_array[
        :, 6
    ]  # XYZ
    result[:, 1, 2, 0] = result[:, 2, 0, 1] = result[:, 2, 1, 0] = oct_array[
        :, 6
    ]  # XYZ

    return result


def rearrange_quadrupole(quad_array: np.ndarray) -> np.ndarray:
    """
    Convert quadrupole tensor from flat representation to 3x3 symmetric matrices.

    Input format: [Qxx, Qyy, Qzz, Qxy, Qxz, Qyz] for each atom
    Output format: 3x3 symmetric matrix for each atom

    Args:
        quad_array: Array of shape (n_atoms, 6) with quadrupole components

    Returns:
        Array of shape (n_atoms, 3, 3) with symmetric quadrupole matrices
    """
    n_atoms = quad_array.shape[0]

    # Pre-allocate the result array
    result = np.zeros((n_atoms, 3, 3))

    # Vectorized assignment for diagonal elements
    result[:, 0, 0] = quad_array[:, 0]  # Qxx
    result[:, 1, 1] = quad_array[:, 1]  # Qyy
    result[:, 2, 2] = quad_array[:, 2]  # Qzz

    # Vectorized assignment for off-diagonal elements (symmetric)
    result[:, 0, 1] = result[:, 1, 0] = quad_array[:, 3]  # Qxy
    result[:, 0, 2] = result[:, 2, 0] = quad_array[:, 4]  # Qxz
    result[:, 1, 2] = result[:, 2, 1] = quad_array[:, 5]  # Qyz

    return result


def get_mbis_from_prop_file(
    json_prop_file: dict, rearrange_tensors: bool = True
) -> dict:
    """
    Extract MBIS population analysis data from ORCA property JSON file.

    Args:
        json_prop_file: Dictionary containing parsed ORCA property JSON data
        rearrange_tensors: If True, rearrange quadrupole and octupole tensors
                          into proper tensor formats (default: True)

    Returns:
        Dictionary containing MBIS multipole moments and properties

    Raises:
        KeyError: If required keys are missing from the JSON structure
        IndexError: If geometry index is out of bounds
    """
    try:
        # Navigate to MBIS data with error checking
        geometries = json_prop_file["geometries"]
        if not geometries:
            raise ValueError("No geometries found in property file")

        geometry = geometries[0]
        mbis_analyses = geometry["mbis_population_analysis"]
        if not mbis_analyses:
            raise ValueError("No MBIS population analysis found")

        mbis_data = mbis_analyses[0]

        # Extract raw arrays
        mbis_mono_array = np.array(mbis_data["atomiccharges"])
        mbis_dip_array = np.array(mbis_data["atomicdipole"])
        mbis_quad_array = np.array(mbis_data["atomicquadrupole"])
        mbis_oct_array = np.array(mbis_data["atomicoctupole"])
        mbis_third_radial_moment = np.array(mbis_data["thirdradialmoment"])

        # Optionally rearrange tensors into proper formats
        if rearrange_tensors:
            mbis_quad_tensor = rearrange_quadrupole(mbis_quad_array)
            mbis_oct_tensor = rearrange_octupole(mbis_oct_array)
        else:
            mbis_quad_tensor = mbis_quad_array
            mbis_oct_tensor = mbis_oct_array

        # Build result dictionary with both raw and processed data
        mbis_prop = {
            # Raw data (flat arrays as from ORCA)
            "mbis_monopole_array": mbis_mono_array,
            "mbis_dipole_array": mbis_dip_array,
            "mbis_quadrupole_array": mbis_quad_array,
            "mbis_octupole_array": mbis_oct_array,
            "mbis_third_radial_moment": mbis_third_radial_moment,
            # Processed tensor data (if rearrange_tensors=True)
            "mbis_quadrupole_tensor": mbis_quad_tensor,
            "mbis_octupole_tensor": mbis_oct_tensor,
            # Metadata
            "n_atoms": len(mbis_mono_array),
            "tensors_rearranged": rearrange_tensors,
        }

        return mbis_prop

    except KeyError as e:
        raise KeyError(f"Missing required key in property file: {e}")
    except (IndexError, ValueError) as e:
        raise ValueError(f"Error processing property file structure: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error processing MBIS data: {e}")


def get_el_prop_from_prop_file(
    json_prop_file: dict,
    rearrange_tensors: bool = True
) -> dict:
    """
    Extract polarizability, dipole, quadrupole moment data, and
    MBIS atomic multipoles from ORCA property JSON file.

    Args:
        json_prop_file: Dictionary containing parsed ORCA property JSON data
        rearrange_tensors: If True, rearrange quadrupole tensors
            into proper tensor format (default: True)

    Returns:
        Dictionary containing:
            - Polarizability (atomic, molecular)
            - Dipole (atomic, molecular)
            - Quadrupole (atomic raw, atomic tensor, molecular, isotropic)
            - MBIS atomic charges, dipoles, quadrupoles, octupoles

    Raises:
        KeyError: If required keys are missing from the JSON structure
        IndexError: If geometry index is out of bounds
        ValueError: If no geometries are present
    """
    try:
        # Navigate to geometry data with error checking
        geometries = json_prop_file["Geometries"]
        if not geometries:
            raise ValueError("No geometries found in property file")

        geometry = geometries[0]

        # ===========
        # Polarizability
        # ===========
        polarizability_data = geometry.get("Polarizability", [{}])[0]
        polar_atom_iso = np.array(polarizability_data.get("atomicPolarIso", []))
        polar_mol = np.array(polarizability_data.get("isotropicPolar", []))

        # ===========
        # Dipole moment
        # ===========
        dipole_data = geometry.get("Dipole_Moment", [{}])[0]
        atomic_dipole = np.array(dipole_data.get("atomicDipole", []))
        molecular_dipole = np.array(dipole_data.get("dipoleTotal", []))

        # ===========
        # Quadrupole moment
        # ===========
        quadrupole_data = geometry.get("Quadrupole_Moment", [{}])[0]
        atomic_quadrupole = np.array(quadrupole_data.get("atomicQuad", []))
        molecular_quadrupole = np.array(quadrupole_data.get("quadTotal", []))
        isotropic_quadrupole = np.array(quadrupole_data.get("isotropicQuadMoment", []))

        # Optionally rearrange quadrupole tensors
        if rearrange_tensors and atomic_quadrupole.size > 0:
            atomic_quadrupole_tensor = rearrange_quadrupole(atomic_quadrupole)
        else:
            atomic_quadrupole_tensor = atomic_quadrupole

        # ===========
        # MBIS population analysis
        # ===========
        mbis_data = geometry.get("MBIS_Population_Analysis", [{}])[0]

        # MBIS atomic charges
        mbis_charges = np.array(mbis_data.get("AtomicCharges", []))

        # MBIS atomic dipoles (vector per atom, shape N_atoms x 3)
        mbis_atomic_dipoles = np.array(mbis_data.get("AtomicDipole", []))

        # MBIS atomic quadrupoles (raw array, typically 6 independent components per atom)
        mbis_atomic_quadrupoles = np.array(mbis_data.get("AtomicQuadrupole", []))

        # MBIS atomic octupoles (raw array, typically 10 independent components per atom)
        mbis_atomic_octupoles = np.array(mbis_data.get("AtomicOctupole", []))

        #Third radial moment
        mbis_r3 = np.array(mbis_data.get("ThirdRadialMoment", []))

        # If you also want rearranged MBIS quadrupole tensors, reuse the same helper
        if rearrange_tensors and mbis_atomic_quadrupoles.size > 0:
            mbis_atomic_quadrupole_tensor = rearrange_quadrupole(mbis_atomic_quadrupoles)
        else:
            mbis_atomic_quadrupole_tensor = mbis_atomic_quadrupoles

        # ===========
        # Build result dictionary
        # ===========
        polar_prop = {
            # Polarizability data
            "atomic_polarizability_iso": polar_atom_iso,
            "molecular_polarizability_iso": polar_mol,

            # Dipole moment data
            "atomic_dipole": atomic_dipole,
            "molecular_dipole": molecular_dipole,

            # Quadrupole moment data (raw + tensor)
            "atomic_quadrupole_array": atomic_quadrupole,
            "molecular_quadrupole": molecular_quadrupole,
            "isotropic_quadrupole": isotropic_quadrupole,
            "atomic_quadrupole_tensor": atomic_quadrupole_tensor,

            # MBIS multipoles
            "mbis_atomic_charges": mbis_charges,
            "mbis_atomic_dipoles": mbis_atomic_dipoles,
            "mbis_atomic_quadrupoles_array": mbis_atomic_quadrupoles,
            "mbis_atomic_quadrupoles_tensor": mbis_atomic_quadrupole_tensor,
            "mbis_atomic_octupoles_array": mbis_atomic_octupoles,
            "mbis_3rd_radial_moment": mbis_r3,
        }

        return polar_prop

    except KeyError as e:
        raise KeyError(f"Missing required key in property file: {e}")
    except (IndexError, ValueError) as e:
        raise ValueError(f"Error processing property file structure: {e}")
    except Exception as e:
        raise RuntimeError(
            f"Unexpected error processing polarizability/moments data: {e}"
        )


def cartesian_to_spherical_quadrupole(cartesian_quad: list[float]) -> np.ndarray:
    """
    Convert Cartesian quadrupole tensor to spherical harmonic coefficients
    Takes full traceless and non-traceless cartesian tensor
    Takes full tensor or just independent components


    Args:
        cartesian_quad: Array of shape (6,) with components [Qxx, Qyy, Qzz, Qxy, Qxz, Qyz]
                       or (3, 3) symmetric matrix

    Returns:
        spherical_quad: Array of shape (5,) with coefficients [q^-2, q^-1, q^0, q^+1 q^+2]
    """
    cartesian_quad_arr = np.array(cartesian_quad)

    if cartesian_quad_arr.shape == (6,):
        # Flat format: [Qxx, Qyy, Qzz, Qxy, Qxz, Qyz]
        Qxx, Qyy, Qzz, Qxy, Qxz, Qyz = cartesian_quad_arr
        # expand to form full tensor
        cartesian_quad_arr = rearrange_quadrupole(np.array([cartesian_quad_arr]))[0]
    elif cartesian_quad_arr.shape == (3, 3):
        # 3x3 matrix format - extract components
        Qxx = cartesian_quad_arr[0, 0]  # Q_xx
        Qyy = cartesian_quad_arr[1, 1]  # Q_yy
        Qzz = cartesian_quad_arr[2, 2]  # Q_zz
        Qxy = cartesian_quad_arr[0, 1]  # Q_xy (equal Q_yx due to symmetry)
        Qxz = cartesian_quad_arr[0, 2]  # Q_xz (equal Q_zx due to symmetry)
        Qyz = cartesian_quad_arr[1, 2]  # Q_yz (equal Q_zy due to symmetry)
    else:
        raise ValueError(
            f"Input shape {cartesian_quad_arr.shape} not supported. Expected (6,) or (3, 3)"
        )

    # rename to p+q+r=l convention
    Q_200 = Qxx
    Q_002 = Qzz
    Q_020 = Qyy
    Q_110 = Qxy
    Q_011 = Qyz
    Q_101 = Qxz
    trace = np.trace(cartesian_quad_arr)
    # Check if trace is close to zero
    if np.allclose(trace, 0):
        # follow conversion outlined here https://github.com/nils-schween/multipole-conv
        q_22 = np.sqrt(3) * (-2 / 3 * Q_020 - 1 / 3 * Q_002)
        q_21 = np.sqrt(3) * 2 / 3 * Q_101
        q_0 = Q_002
        q_2m1 = np.sqrt(3) * 2 / 3 * Q_011
        q_2m2 = np.sqrt(3) * 2 / 3 * Q_110

    else:
        # conversion from mwfn manual, for non-traceless tensor
        q_22 = np.sqrt(3) / 2 * (Q_200 - Q_020)
        q_21 = np.sqrt(3) * Q_101
        q_0 = (3 * Q_002 - trace) / 2
        q_2m1 = np.sqrt(3) * Q_011
        q_2m2 = np.sqrt(3) * Q_110

    return np.array([q_2m2, q_2m1, q_0, q_21, q_22])


# TODO MWFN returns spehcircal harmonics, but to get principal axes, need cartesian tensors
def spherical_quadrupole_to_cartesian(sph_quad: list[float]) -> np.ndarray:
    """./multipole-conv -d 2 -c real_solid_harmonics --include-addition-thm 1 --cartesian 1 --remove-csp 1 --include-l-factorial 0 --normalisation 1 --split-addition-thm 1 --dependent-components 1
        Cartesian multipole moments (independent components = multipole basis functions)

        Q_0,2,0 = -1.73205 q_2,2,0 -1 q_2,0,0
        Q_0,1,1 = 1.73205 q_2,1,1
        Q_0,0,2 = 2 q_2,0,0
        Q_1,0,1 = 1.73205 q_2,1,0
        Q_1,1,0 = 1.73205 q_2,2,1

    Cartesian multipole moments (dependent components)

    Q_2,0,0 = 1.73205 q_2,2,0 -1 q_2,0,0
        Args:
            sph_quad (list[float]): Irreducible representation of quadrupole moment
                                    [q^-2, q^-1, q^0, q^+1 q^+2]

        Returns:
            np.ndarray: irreducible components of cartesian quadrupole tensor
                        [Qxx, Qyy, Qzz, Qxy, Qxz, Qyz]
    """
    cart_quad = np.zeros((3, 3))
    return cart_quad


def cartesian_to_spherical_octupole(cartesian_oct: list[float]) -> np.ndarray:
    """
    Convert Cartesian octupole tensor to spherical harmonic coefficients
    using the exact equations provided.
    Takes full tensor or independent compononents
    Not implemented for traceless tensors

    Args:
        cartesian_oct: Array of shape (10,) with components [XXX, YYY, ZZZ, XXY, XXZ, XYY, XYZ, XZZ, YYZ, YZZ]
                      or (3, 3, 3) symmetric tensor

    Returns:
        spherical_oct: Array of shape (7,) with coefficients [q^-3, q^-2, q^-1, q^0, q^+1, q^+2, q^+3]
    """
    cartesian_oct_arr = np.array(cartesian_oct)

    if cartesian_oct_arr.shape == (10,):
        # Flat format: [XXX, YYY, ZZZ, XXY, XXZ, XYY, XYZ, XZZ, YYZ, YZZ]
        XXX, YYY, ZZZ, XXY, XXZ, XYY, XYZ, XZZ, YYZ, YZZ = cartesian_oct_arr
        cartesian_oct_arr = rearrange_octupole(np.array([cartesian_oct_arr]))[0]
    elif cartesian_oct_arr.shape == (3, 3, 3):
        # 3x3x3 tensor format - extract components
        XXX = cartesian_oct_arr[0, 0, 0]  # Q_xxx
        YYY = cartesian_oct_arr[1, 1, 1]  # Q_yyy
        ZZZ = cartesian_oct_arr[2, 2, 2]  # Q_zzz
        XXY = cartesian_oct_arr[0, 0, 1]  # Q_xxy
        XXZ = cartesian_oct_arr[0, 0, 2]  # Q_xxz
        XYY = cartesian_oct_arr[0, 1, 1]  # Q_xyy
        XYZ = cartesian_oct_arr[0, 1, 2]  # Q_xyz
        XZZ = cartesian_oct_arr[0, 2, 2]  # Q_xzz
        YYZ = cartesian_oct_arr[1, 1, 2]  # Q_yyz
        YZZ = cartesian_oct_arr[1, 2, 2]  # Q_yzz
    else:
        raise ValueError(
            f"Input shape {cartesian_oct_arr.shape} not supported. Expected (10,) or (3, 3, 3)"
        )

    # Map to p+q+r=3 convention
    Q_300 = XXX  # Q_3,0,0
    Q_030 = YYY  # Q_0,3,0
    Q_003 = ZZZ  # Q_0,0,3
    Q_210 = XXY  # Q_2,1,0
    Q_201 = XXZ  # Q_2,0,1
    Q_120 = XYY  # Q_1,2,0
    Q_111 = XYZ  # Q_1,1,1
    Q_102 = XZZ  # Q_1,0,2
    Q_021 = YYZ  # Q_0,2,1
    Q_012 = YZZ  # Q_0,1,2

    # get traces
    q_trx, q_try, q_trz = np.trace(cartesian_oct_arr)

    # conversion according to mwfn manual
    q_33 = np.sqrt(5 / 8) * (Q_300 - 3 * Q_120)
    q_32 = np.sqrt(15) / 2 * (Q_201 - Q_021)
    q_31 = np.sqrt(3 / 8) * (5 * Q_102 - q_trx)
    q_30 = 1 / 2 * (5 * Q_003 - 3 * q_trz)
    q_3m1 = np.sqrt(3 / 8) * (5 * Q_012 - q_try)
    q_3m2 = np.sqrt(15) * Q_111
    q_3m3 = np.sqrt(5 / 8) * (3 * Q_210 - Q_030)

    return np.array([q_3m3, q_3m2, q_3m1, q_30, q_31, q_32, q_33])


# TODO
def spherical_octupole_to_cartesian_octupole(sph_oct: list[float]) -> np.ndarray:
    """_summary_

    Args:
        sph_oct (list[float]): _description_

    Returns:
        np.ndarray: _description_
    """
    cart_oct = np.zeros((4, 4, 4))
    return cart_oct


def add_orca_quad_sphm(property_dict: dict) -> dict:
    """Add spherical harmonic tensors for mbis quadrupole and octupole

    Args:
        property_dict (dict): MBIS property dict

    Returns:
        dict: MBIS property dict with added spherical harmonic tensors
    """
    n_atoms = property_dict["n_atoms"]
    quad_array = np.zeros((n_atoms, 5))
    oct_array = np.zeros((n_atoms, 7))
    for atom in range(n_atoms):
        quad_cart_tensor = property_dict["mbis_quadrupole_array"][atom]
        quad_sph_tensor = cartesian_to_spherical_quadrupole(quad_cart_tensor)
        oct_cart_tensor = property_dict["mbis_octupole_array"][atom]
        oct_sph_tensor = cartesian_to_spherical_octupole(oct_cart_tensor)
        quad_array[atom] = quad_sph_tensor
        oct_array[atom] = oct_sph_tensor
    property_dict["mbis_quadrupole_sphm"] = quad_array
    property_dict["mbis_octupole_sphm"] = oct_array
    return property_dict


def extract_properties(orca_out, mwfn_out):
    """
    Extract all properties from ORCA and Multiwfn outputs.

    Parameters:
    -----------
    output : ORCA output object
    mwfn_inp_file : Path to Multiwfn input file

    Returns:
    --------
    dict : Dictionary containing all extracted properties
    """
    # Load ORCA properties
    json_file = orca_out.get_file(".property.json").__str__()
    with open(json_file, "r") as jfile:
        prop_file = json.load(jfile)

    # Load Multiwfn properties
    mwfn_out_dict = mwfn_out.get_array_dict()

    # Extract MBIS properties
    mbis_props = get_mbis_from_prop_file(prop_file, rearrange_tensors=False)
    mbis_props = add_orca_quad_sphm(mbis_props)
    el_props = get_el_prop_from_prop_file(prop_file, rearrange_tensors=False)

    return (
        prop_file,
        mwfn_out_dict["quadrupoles"],
        mwfn_out_dict["octopoles"],
        mbis_props["mbis_quadrupole_sphm"],
        mbis_props["mbis_octupole_sphm"],
        mbis_props,
        el_props,
    )


def smiles_to_mol(smiles: str) -> Chem.Mol:
    """Convert smiles to featurised PyGraph

    Args:
        smiles (str): SMILES string of molecule

    Returns:
        PyGraph: Featurised RustworkX graph
    """
    mol = Chem.MolFromSmiles(smiles)
    mol_3d = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol_3d)
    return mol_3d


def xyz_block_to_rd_mol(xyz_block: str) -> Chem.Mol:
    """Converts xyz string to rdkit Mol object

    Args:
        xyz_block (str): XYZ string (Angstrom)

    Returns:
        Chem.Mol: Rdkitmot object
    """
    pybel_mol = readstring(format="xyz", string=xyz_block)

    # Method 1: Auto-cleanup with context manager (RECOMMENDED)
    with tempfile.NamedTemporaryFile(suffix=".sdf", delete=True) as tmp_file:
        pybel_mol.write("sdf", tmp_file.name, overwrite=True)
        rdkit_mol = Chem.MolFromMolFile(tmp_file.name, removeHs=False)
        return rdkit_mol


def align_ci_xaxis(
    structure: Structure, return_i_index: bool = False
) -> Union[Structure, tuple[Structure, int]]:
    coords = np.array([atom.coordinates.coordinates for atom in structure.atoms])
    if hasattr(structure, "smiles"):
        # 1) Build RDKit mol from SMILES to preserve [N+]
        rd_mol = Chem.MolFromSmiles(structure.smiles)
        # If your Structure includes explicit Hs, mirror that:
        rd_mol = Chem.AddHs(rd_mol)

        # 2) Attach ORCA-optimised coordinates as a conformer
        conf = Chem.Conformer(rd_mol.GetNumAtoms())
        for i in range(rd_mol.GetNumAtoms()):
            x, y, z = coords[i]
            conf.SetAtomPosition(i, (float(x), float(y), float(z)))
        rd_mol.RemoveAllConformers()
        rd_mol.AddConformer(conf, assignId=True)

    else:
        # Fallback: use XYZ + total_charge if available
        xyz = structure.to_xyz_block()
        rd_mol = xyz_block_to_rd_mol(
            xyz,
            total_charge=getattr(structure, "charge", 0),
        )
    ci_smarts = Chem.MolFromSmarts("c-I")
    ch_smarts = Chem.MolFromSmarts("[c;H]", mergeHs=True)
    try:
        c_index, i_index = rd_mol.GetSubstructMatch(ci_smarts)
        c2_index = rd_mol.GetSubstructMatch(ch_smarts)[0]
    except ValueError:
        ci_smarts = Chem.MolFromSmarts("CI")
        c_index, i_index = rd_mol.GetSubstructMatch(ci_smarts)
        c2_index = max(c_index, i_index) + 1

    c_coords = coords[c_index]
    i_coords = coords[i_index]
    c2_coords = coords[c2_index]

    # Calculate rotation matrix to align ci_vec with z-axis
    rotation_matrix = get_alignment_matrix(i_coords, c_coords, c2_coords)

    # Apply rotation to all coordinates
    rotated_coords = apply_transformation(coords=coords, T=rotation_matrix)

    # Create new structure with rotated coordinates
    new_atoms = []
    for i, atom in enumerate(structure.atoms):
        new_atom = Atom(element=atom.element, coordinates=rotated_coords[i])
        new_atoms.append(new_atom)

    if return_i_index:
        return Structure(new_atoms), i_index
    else:
        return Structure(new_atoms)


def add_anion_on_ci_axis_structure(opt_struct: Structure, anion: str,
                                   distance: float = 3.0) -> Structure:
    """
    Take an aligned ORCA Structure (C–I on z-axis), add a monatomic anion
    along +z from iodine at a given distance, and return a new Structure.
    """
    coords = np.array([atom.coordinates.coordinates for atom in opt_struct.atoms])
    if hasattr(opt_struct, "smiles"):  #Converting ORCA structure object into RDkit molecule object
        # 1) Build RDKit mol from SMILES to preserve [N+]
        rd_mol = Chem.MolFromSmiles(opt_struct.smiles)
        # If your Structure includes explicit Hs, mirror that:
        rd_mol = Chem.AddHs(rd_mol)

        # 2) Attach ORCA-optimised coordinates as a conformer
        conf = Chem.Conformer(rd_mol.GetNumAtoms())
        for i in range(rd_mol.GetNumAtoms()):
            x, y, z = coords[i]
            conf.SetAtomPosition(i, (float(x), float(y), float(z)))
        rd_mol.RemoveAllConformers()
        rd_mol.AddConformer(conf, assignId=True)

    else:
        # Fallback: use XYZ + total_charge if available
        xyz = opt_struct.to_xyz_block()
        rd_mol = xyz_block_to_rd_mol(
            xyz,
            total_charge=getattr(structure, "charge", 0),
        )
    
    #Find I and it's coord
    conf = rd_mol.GetConformer()
    i_indices = [a.GetIdx() for a in rd_mol.GetAtoms()
                 if a.GetSymbol() == "I"]
    if len(i_indices) != 1:
        raise ValueError(f"Expected exactly one iodine, found {len(i_indices)}.")
    i_idx = i_indices[0]
    i_pos = conf.GetAtomPosition(i_idx)
    z_I = i_pos.z

    #Add anion to RDKit Mol
    em = Chem.EditableMol(rd_mol)
    new_idx = em.AddAtom(Chem.Atom(anion))
    rd_mol = em.GetMol()

    # Ensure conformer is still valid
    if rd_mol.GetNumConformers() == 0:
        rd_mol.AddConformer(conf, assignId=True)
    conf = rd_mol.GetConformer()

    # Place anion at same x,y as iodine, but z = z_I + distance
    conf.SetAtomPosition(
        new_idx,
        (float(i_pos.x), float(i_pos.y), float(z_I + distance))
    )

    #Convert back to ORCA Structure
    new_atoms = []
    AtomCls = opt_struct.atoms[0].__class__
    CoordCls = opt_struct.atoms[0].coordinates.__class__

    conf = rd_mol.GetConformer()
    for idx, atom in enumerate(rd_mol.GetAtoms()):
        pos = conf.GetAtomPosition(idx)
        x, y, z = pos.x, pos.y, pos.z
        elem = atom.GetSymbol()
        new_atom = AtomCls(
            element=elem,
            coordinates=CoordCls(x=x, y=y, z=z)
        )
        new_atoms.append(new_atom)

    new_struct = opt_struct.__class__(atoms=new_atoms)
    
    # Adjust charge for the anion; keep remaning metadata same
    new_struct.charge = opt_struct.charge - 1
    new_struct.smiles = opt_struct.smiles
    new_struct.multiplicity = opt_struct.multiplicity

    return new_struct

def get_elem_list(structure: Structure) -> list[str]:
    """Create list of elements in structure in ascending order

    Args:
        structure (Structure): structure as ORCA structure
    Returns:
        list[str]: List of elements sorted by atomic number
    """
    # Extract unique atomic numbers in one pass using set comprehension
    unique_atomic_numbers = sorted(
        {atom.element.atomic_number for atom in structure.atoms}
    )

    # Convert to chemical symbols
    return [chemical_symbols[atomic_num] for atomic_num in unique_atomic_numbers]


# TODO
# write get principal components of quadrupole tensor from its cartesian tensor!
# write functions spherical_quadrupole_to_cartesian first


def get_alignment_matrix(
    atom1_pos: np.ndarray, atom2_pos: np.ndarray, atom3_pos: np.ndarray
) -> np.ndarray:
    """
    Calculate transformation matrix to:
    1. Center atom2 at origin (0,0,0)
    2. Align atom1-atom2 bond along z-axis
    3. Place atom1-atom2-atom3 plane in yz-plane

    Parameters:
    -----------
    atom1_pos : array-like, shape (3,)
        Position of atom 1 (x, y, z)
    atom2_pos : array-like, shape (3,)
        Position of atom 2 (x, y, z) - will be centered at origin
    atom3_pos : array-like, shape (3,)
        Position of atom 3 (x, y, z)

    Returns:
    --------
    T : ndarray, shape (4, 4)
        Homogeneous transformation matrix
    """
    # Convert to numpy arrays
    a1 = np.array(atom1_pos, dtype=float)
    a2 = np.array(atom2_pos, dtype=float)
    a3 = np.array(atom3_pos, dtype=float)

    # Step 1: Translation to center atom2 at origin
    translation = -a2

    # Apply translation to all atoms
    a1_t = a1 + translation
    a2_t = np.zeros(3)  # a2 is now at origin
    a3_t = a3 + translation

    # Step 2: Define z-axis as atom1-atom2 direction
    z_axis = a1_t - a2_t
    z_axis = z_axis / np.linalg.norm(z_axis)

    # Step 3: Define x-axis perpendicular to plane containing atoms 1, 2, 3
    # The plane normal is perpendicular to both (a1-a2) and (a3-a2)
    v1 = a1_t - a2_t
    v2 = a3_t - a2_t
    x_axis = np.cross(v1, v2)
    x_axis = x_axis / np.linalg.norm(x_axis)

    # Step 4: Define y-axis to complete right-handed coordinate system
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    # Create rotation matrix (transpose because we're rotating the molecule, not the axes)
    # The columns of R^T are the new basis vectors
    R = np.column_stack([x_axis, y_axis, z_axis]).T

    # Create homogeneous transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = R @ translation

    return T


def apply_transformation(coords, T):
    """
    Apply transformation matrix to coordinates.

    Parameters:
    -----------
    coords : ndarray, shape (N, 3)
        Coordinates of N atoms
    T : ndarray, shape (4, 4)
        Transformation matrix

    Returns:
    --------
    transformed_coords : ndarray, shape (N, 3)
        Transformed coordinates
    """
    # Convert to homogeneous coordinates
    n_atoms = coords.shape[0]
    homogeneous = np.hstack([coords, np.ones((n_atoms, 1))])

    # Apply transformation
    transformed = (T @ homogeneous.T).T

    # Convert back to 3D coordinates
    return transformed[:, :3]


def get_iodine_mbis(mbis_props: dict, index: int) -> dict:
    iodine_mbis = dict()
    for key in mbis_props.keys():
        if "mbis" not in key:
            continue
        if "upole_array" in key or "upole_tensor" in key:
            continue
        iodine_mbis[key] = mbis_props[key][index].tolist()
    return iodine_mbis


# classes and functions for running and processing mwfn calculations
class MwfnFuzzySpaceOut:
    """Class to parse the output of mwfn fuzzy space integration calculations.

    This class reads an output file from a mwfn calculation and extracts several
    atomic properties including multipole moments, overlap matrix, C6 coefficients,
    polarizabilities, effective volumes, and fuzzy space integration values. The
    information is stored in NumPy arrays and exposed via read-only properties.

    Args:
        out_file (str): Path to the mwfn output file.

    Attributes:
        charges (np.ndarray): Array of atomic charges.
        dipoles (np.ndarray): Array of atomic dipole moments.
        quadrupoles (np.ndarray): Array of atomic quadrupole moments (spherical harmonics).
        octopoles (np.ndarray): Array of atomic octopole moments (spherical harmonics).
        r_2 (np.ndarray): Array of spatial extent values (<r^2>) for each atom.
        overlap_matrix (np.ndarray): Overlap matrix of the system.
        c6 (np.ndarray): Array of atomic C6 coefficients.
        polarisabilities (np.ndarray): Array of atomic polarizabilities and their contributions.
        effective_volume (np.ndarray): Array of effective and free volumes for each atom.
        fuzzy_space_integral (np.ndarray): Array of fuzzy space integration values.
        iodine_vmin_vmax (np.ndarray): (2,) Array of ESP Vmin/Vmax (kcal/mol) on the molecular surface for iodine

    Example:
        mwfn = MwfnFuzzySpaceOut("path/to/mwfn_output.out")
        print(mwfn.charges)
    """

    def __init__(self, out_file: str):
        self.out_file = out_file
        self.content = self._read_file()
        self._parse_content()

    def _read_file(self):
        with open(self.out_file, "r") as f:
            content = f.read()
        return content

    def _parse_content(self):
        (
            self._num_atoms,
            self._charge_array,
            self._dipole_array,
            self._quadrupole_array,
            self._octopole_array,
            self._r_2_array,
        ) = self._get_multipole()
        self._overlap_matrix = self._get_overlap_matrix()
        self._c6_array = self._get_atomic_c6_coefficients()
        self._polarisability_array = self._get_atomic_polarisabilities()
        self._effective_volume_array = self._get_atomic_effective_volume()
        self._fuzzy_space_array = self._get_fuzzy_space_integration()
        self._partitioning_scheme = self._get_partitioning_scheme()

    def __repr__(self):
        return f"""MwfnOut({self.out_file})\nFirst 3 atoms:\nCharges:\n{self.charges[:3]}\nDipoles:\n{self.dipoles[:3]}\nQuadrupoles:\n{self.quadrupoles[:3]}\nOctopoles:\n{self.octopoles[:3]}\n<r^2>:\n{self.r_2[:3]}"""

    def __str__(self):
        return self.__repr__()

    def _get_multipole(
        self,
    ) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extract multipole moments from the mwfn output file, in atomic units.
        Example for output file:
                                   *****  Atom     4(C )  *****
            Atomic charge:   -0.181999
            Atomic monopole moment (from electrons):   -6.181999
            Atomic dipole moments:
            X=   -0.151927  Y=    0.294188  Z=   -0.047095  Norm=    0.334434
            Contribution to molecular dipole moment:
            X=   -3.022647  Y=    0.134167  Z=    1.891539  Norm=    3.568237
            Atomic quadrupole moments (Traceless Cartesian form):
            XX=    0.594070  XY=    0.184042  XZ=    0.377291
            YX=    0.184042  YY=    0.387058  YZ=   -0.052244
            ZX=    0.377291  ZY=   -0.052244  ZZ=   -0.981128
            Magnitude of the traceless quadrupole moment tensor:    0.988380
            Atomic quadrupole moments (Spherical harmonic form):
            Q_2,0 =  -0.981128   Q_2,-1=  -0.060326   Q_2,1=   0.435658
            Q_2,-2=   0.212513   Q_2,2 =   0.119518
            Magnitude: |Q_2|=    1.102495
            Atomic electronic spatial extent <r^2>:     10.980388
            Components of <r^2>:  X=      3.264083  Y=      3.402091  Z=      4.314215
            Atomic octopole moments (Spherical harmonic form):
            Q_3,0 =  -0.172347  Q_3,-1=   0.288102  Q_3,1 =  -0.234485
            Q_3,-2=   0.161763  Q_3,2 =  -0.289436  Q_3,-3=   1.080882  Q_3,3 =  -0.954022
            Magnitude: |Q_3|=    1.534958

        Returns:
            charge_array (np.ndarray): Array of atomic charges.
            dipole_array (np.ndarray): Array of atomic dipole moments.
            quadrupole_array (np.ndarray): Array of atomic quadrupole moments in spherical harmonics: Q_2,0, Q_2,-1, Q_2,1, Q_2,-2, Q_2,2.
            octopole_array (np.ndarray): Array of atomic octopole moments in spherical harmonics: Q_3,0, Q_3,-1, Q_3,1, Q_3,-2, Q_3,2, Q_3,-3, Q_3,3.
            r_2_array (np.ndarray): Array of spatial extent <r^2> for each atom, in atomic units: X, Y, Z components.
            metadata (dict): Metadata dictionary containing information about the output file.
        Returns: MfwnFuzzySpaceOut
        """
        # regex patterns to extract data
        split_pattern = re.compile(r"\s*\*{5}\s*Atom\s*(\d+)\s*\(\w+\s*\)\s*\*{5}")
        charge_pattern = re.compile(r"Atomic\s*charge\s*:\s*(-?\d+.\d+)")
        dipole_pattern = re.compile(
            r"X=\s*(-?\d+.\d+)\s*Y=\s*(-?\d+.\d+)\s*Z=\s*(-?\d+.\d+)"
        )
        quadrupole_pattern = re.compile(
            r"\s*Q_2,0\s*=\s*(-?\d+.\d+)\s*Q_2,-1\s*=\s*(-?\d+.\d+)\s*Q_2,1\s*=\s*(-?\d+.\d+)\s*Q_2,-2\s*=\s*(-?\d+.\d+)\s*Q_2,2\s*=\s*(-?\d+.\d+)"
        )
        octopole_pattern = re.compile(
            r"\s*Q_3,0\s*=\s*(-?\d+.\d+)\s*Q_3,-1\s*=\s*(-?\d+.\d+)\s*Q_3,1\s*=\s*(-?\d+.\d+)\s*Q_3,-2\s*=\s*(-?\d+.\d+)\s*Q_3,2\s*=\s*(-?\d+.\d+)\s*Q_3,-3\s*=\s*(-?\d+.\d+)\s*Q_3,3\s*=\s*(-?\d+.\d+)"
        )
        r_2_pattern = re.compile(
            r"Components of <r\^2>:\s*X=\s*(-?\d+.\d+)\s*Y=\s*(-?\d+.\d+)\s*Z=\s*(-?\d+.\d+)"
        )
        # split by atom
        split_atoms = split_pattern.split(self.content)[1:]
        num_atoms = len(split_atoms) // 2
        # initialize arrays
        charge_array = np.zeros((num_atoms,))
        dipole_array = np.zeros((num_atoms, 3))
        r_2_array = np.zeros((num_atoms, 3))
        quadrupole_array = np.zeros((num_atoms, 5))
        octopole_array = np.zeros((num_atoms, 7))
        # iterate over atoms and extract data
        for i in range(0, len(split_atoms), 2):
            charge = float(charge_pattern.search(split_atoms[i + 1]).group(1))
            # quadrupole and octopole
            quad_vals = quadrupole_pattern.search(split_atoms[i + 1])
            oct_vals = octopole_pattern.search(split_atoms[i + 1])
            r_2_vals = r_2_pattern.search(split_atoms[i + 1])
            quad_array = np.array([float(quad_vals.group(i)) for i in range(1, 6)])
            # rearrange quadrupole
            q_20, q_2m1, q_21, q_2m2, q_22 = quad_array
            quad_array = np.array([q_2m2, q_2m1, q_20, q_21, q_22])
            # octupole
            oct_array = np.array([float(oct_vals.group(i)) for i in range(1, 8)])
            # rearrange octupole
            q_30, q_3m1, q_31, q_3m2, q_32, q_3m3, q_33 = oct_array
            oct_array = np.array([q_3m3, q_3m2, q_3m1, q_30, q_31, q_32, q_33])
            # radial spatial extent
            r_2_arr = np.array([float(r_2_vals.group(i)) for i in range(1, 4)])
            # dipole
            dip_vals = dipole_pattern.search(split_atoms[i + 1])
            dip_array = np.array([float(dip_vals.group(i)) for i in range(1, 4)])
            # put data into arrays
            charge_array[i // 2] = charge
            quadrupole_array[i // 2] = quad_array
            octopole_array[i // 2] = oct_array
            dipole_array[i // 2] = dip_array
            r_2_array[i // 2] = r_2_arr

        return (
            num_atoms,
            charge_array,
            dipole_array,
            quadrupole_array,
            octopole_array,
            r_2_array,
        )

    def _get_overlap_matrix(self) -> np.ndarray:
        """Extract the overlap matrix from the mwfn output file.
        Example for output file:
           ************* Integration of positive values in overlap region *************
                        1             2             3             4             5
                1  176.90407027    6.29147790    0.54491889    0.00001080    0.00000000
                2    6.29147790  109.65636682    4.81253976    0.21859500    0.00000479
                3    0.54491889    4.81253976  109.10536318    5.14978128    0.28519183
                4    0.00001080    0.21859500    5.14978128  119.44301555    6.95092816
                5    0.00000000    0.00000479    0.28519183    6.95092816  117.04038616
        Returns:
            overlap_matrix (np.ndarray): Overlap matrix of the system.
        """
        all_int_val_pattern = re.compile(
            r"\*{16}\s+Integration of all values in overlap region\s+\*{16}\n"
        )
        if not re.search(all_int_val_pattern, self.content):
            warnings.warn(
                "No overlap matrix found in the output file.", UserWarning, stacklevel=2
            )
            return np.zeros((self._num_atoms, self._num_atoms), dtype=np.float32)
        all_int_val = all_int_val_pattern.split(self.content)[1]
        header_pattern = re.compile(r"\s\d{1,3}[\s+\d{1,3}]{0,}\s+\n")
        blocks = header_pattern.split(all_int_val)[1:]
        # get number of columns in last block
        array_size = self._num_atoms
        array = np.zeros((array_size, array_size), dtype=np.float32)

        row_pattern = re.compile(r"(?<!\S)-?\d*\.\d+")
        for i, block in enumerate(blocks):
            # split lines
            lines = block.strip().split("\n")
            # get numbers from each line
            for j, line in enumerate(lines):
                # if line doesnt start with a number, break
                if not line.strip()[0].isdigit():
                    break
                numbers = row_pattern.findall(line)
                for k, number in enumerate(numbers):
                    # convert to float and assign to array
                    array[j, k + i * 5] = float(number)
        return array

    def _get_atomic_c6_coefficients(self) -> np.ndarray:
        """Extract C6 coefficients from the mwfn output file.
            Reference atomic WF obtained wtih r2scan-3c
            C6 coefficients estimatedd using tkatchenko-scheffler method.
        Example for outputfile:
        Atomic C6 coefficients estimated using Tkatchenko-Scheffler method:
            1(O ):   21.58 a.u. (Ref. data:    15.6 a.u.)
            2(C ):   18.74 a.u. (Ref. data:    46.6 a.u.)
            3(C ):   42.47 a.u. (Ref. data:    46.6 a.u.)
            4(C ):   41.67 a.u. (Ref. data:    46.6 a.u.)
            5(C ):   44.82 a.u. (Ref. data:    46.6 a.u.)
        Returns:
            c6_array (np.ndarray): Array of C6 coefficients.
        """
        c6_pattern = re.compile(
            r"Atomic C6 coefficients estimated using Tkatchenko-Scheffler method\:\n"
        )
        if not re.search(c6_pattern, self.content):
            warnings.warn(
                "No C6 coefficients found in the output file.",
                UserWarning,
                stacklevel=2,
            )
            return np.zeros((self._num_atoms), dtype=np.float64)
        c6_section = c6_pattern.split(self.content)[1]
        c6_values = re.findall(
            r"\s*\d+\([\w\s]{1,2}\)\:\s*(-?\d+\.\d{1,2})(?=\sa\.u\.\s*\(Ref\.\s*data\:\s*\d+\.\d+\s*a\.u\.\)\n)",
            c6_section,
        )
        c6_array = np.zeros((self._num_atoms), dtype=np.float64)
        for i, value in enumerate(c6_values):
            c6_array[i] = float(value)
        assert (
            c6_array.shape[0] == self._num_atoms
        ), "C6 coefficients length does not match number of atoms."
        return c6_array

    def _get_atomic_polarisabilities(self) -> np.ndarray:
        """Extract atomic polarizabilities and contributions to the total polarizability from the mwfn output file.
        The polarizabilities are estimated using the Tkatchenko-Scheffler method.
        The polarizabilities are given in atomic units (a.u.) and the contributions are given in percentage (%).
        Example for outputfile:
        Atomic polarizabilities estimated using Tkatchenko-Scheffler method:
            1(O ):   6.234 a.u.  Contribution:  2.74 %  (Ref. data:   5.300 a.u.)
            2(C ):   7.166 a.u.  Contribution:  3.14 %  (Ref. data:  11.300 a.u.)
            3(C ):  10.788 a.u.  Contribution:  4.73 %  (Ref. data:  11.300 a.u.)
            4(C ):  10.686 a.u.  Contribution:  4.69 %  (Ref. data:  11.300 a.u.)
            5(C ):  11.082 a.u.  Contribution:  4.86 %  (Ref. data:  11.300 a.u.)

        Returns:
            np.ndarray: Array of polarizabilities with two columns:
                - Column 0: Polarizability in a.u.
                - Column 1: Contribution to the total polarizability in percentage (%).
        """
        polarisability_pattern = re.compile(
            r"Atomic polarizabilities estimated using Tkatchenko-Scheffler method\:\n"
        )
        if not re.search(polarisability_pattern, self.content):
            warnings.warn(
                "No polarizabilities found in the output file.",
                UserWarning,
                stacklevel=2,
            )
            return np.zeros((self._num_atoms), dtype=np.float64)
        polarisability_section = polarisability_pattern.split(self.content)[1]
        polarisability_values = re.findall(
            r"\s*\d+\([\w\s]{1,2}\)\:\s*(-?\d+\.\d{3})\sa\.u\.\s*Contribution\:\s*(\d+\.\d+)(?=\s*\%\s*\(Ref\.\s*data\:\s*\d+\.\d+\s*a\.u\.\)\n)",
            polarisability_section,
        )
        polarisability_array = np.zeros((self._num_atoms, 2), dtype=np.float64)
        for i, value in enumerate(polarisability_values):
            polarisability_array[i, 0] = float(value[0])
            polarisability_array[i, 1] = float(value[1])
        assert (
            polarisability_array.shape[0] == self._num_atoms
        ), "Polarizabilities length does not match number of atoms."
        assert (
            polarisability_array.shape[1] == 2
        ), "Polarizabilities array should have two columns."
        return polarisability_array

    def _get_atomic_effective_volume(self) -> np.ndarray:
        """Extract atomic effective volumes from the mwfn output file.
        The effective volume is given in atomic units (a.u.).
        Example for output file:
            Running: /usr/local/orca_6_0_1/orca_2mkl "imine_1_conf_0" -molden > /dev/null
            Deleting imine_1_conf_0.molden.input
            Atom    1(O )  Effective V:    25.485  Free V:    21.667 a.u.  Ratio: 1.176
            Running: /usr/local/orca_6_0_1/orca_2mkl "imine_1_conf_0" -molden > /dev/null
            Deleting imine_1_conf_0.molden.input
            Atom    2(C )  Effective V:    21.835  Free V:    34.433 a.u.  Ratio: 0.634
            Running: /usr/local/orca_6_0_1/orca_2mkl "imine_1_conf_0" -molden > /dev/null
            Deleting imine_1_conf_0.molden.input
            Atom    3(C )  Effective V:    32.872  Free V:    34.433 a.u.  Ratio: 0.955
            Running: /usr/local/orca_6_0_1/orca_2mkl "imine_1_conf_0" -molden > /dev/null
            Deleting imine_1_conf_0.molden.input
            Atom    4(C )  Effective V:    32.563  Free V:    34.433 a.u.  Ratio: 0.946
            Running: /usr/local/orca_6_0_1/orca_2mkl "imine_1_conf_0" -molden > /dev/null
            Deleting imine_1_conf_0.molden.input
            Atom    5(C )  Effective V:    33.768  Free V:    34.433 a.u.  Ratio: 0.981
        Returns:
            np.ndarray: Array of effective volumes and free volumes for each atom.
        1. Column 0: Effective volume in a.u.
        2. Column 1: Free volume in a.u.
        """
        import warnings

        effective_volume_pattern = re.compile(
            r"Atom\s*\d+\s*\([\w\s]{1,2}\)\s*Effective\s*V\:\s*(-?\d+\.\d+)\s*Free\s*V\:\s*(-?\d+\.\d+)\sa\.u\.(?=\s*Ratio\:\s*\d+\.\d+\n)"
        )
        eff_vol_values = re.findall(effective_volume_pattern, self.content)
        if not eff_vol_values:
            warnings.warn(
                "No effective volumes found in the output file.",
                UserWarning,
                stacklevel=2,
            )
            return np.zeros((self._num_atoms, 2), dtype=np.float64)
        if len(eff_vol_values) < self._num_atoms:
            raise ValueError(
                f"Found only {len(eff_vol_values)} effective volume entries for "
                f"{self._num_atoms} atoms."
            )

        if len(eff_vol_values) > self._num_atoms:
            warnings.warn(
                f"Found {len(eff_vol_values)} effective volume entries; "
                f"expected {self._num_atoms}. Using the first {self._num_atoms}.",
                UserWarning,
                stacklevel=2,
            )

        effective_volume_array = np.zeros((self._num_atoms, 2), dtype=np.float64)

        for i, value in enumerate(eff_vol_values[: self._num_atoms]):
            effective_volume_array[i, 0] = float(value[0])
            effective_volume_array[i, 1] = float(value[1])

        assert effective_volume_array.shape[0] == self._num_atoms
        assert effective_volume_array.shape[1] == 2

        return effective_volume_array

    def _get_fuzzy_space_integration(self) -> np.ndarray:
        """Extract fuzzy space integration values from the mwfn output file.
        Integration function is userfunc: Here Weizsacker kinetic energy.
        Example for output file:
               Atomic space        Value                % of sum            % of sum abs
                1(O )          170.84183165             5.125328             5.125328
                2(C )          120.80265746             3.624131             3.624131
                3(C )          132.72326823             3.981755             3.981755
                4(C )          144.84255586             4.345339             4.345339
                5(C )          159.20888500             4.776335             4.776335

        Returns:
            np.ndarray: Array of fuzzy space integration values.
        """
        fuzzy_space_header_pattern = re.compile(
            r"\s*Atomic\s*space\s*Value\s*\% of sum\s*\% of sum abs\s*\n"
        )
        if not re.search(fuzzy_space_header_pattern, self.content):
            warnings.warn(
                "No fuzzy space integration values found in the output file.",
                UserWarning,
                stacklevel=2,
            )
            return np.zeros((self._num_atoms), dtype=np.float64)
        fuzzy_table = fuzzy_space_header_pattern.split(self.content)[1:]
        fuzzy_table = fuzzy_table[0].strip().split("\n")
        fuzzy_array = np.zeros(self._num_atoms, dtype=np.float64)
        row_pattern = re.compile(
            r"\s*\d+\([\w\s]{1,2}\)\s*(-?\d+\.\d+)\s*-?\d+\.\d+\s*-?\d+\.\d+\s*"
        )
        for i, line in enumerate(fuzzy_table):
            if not line.strip()[0].isdigit():
                break
            int_vals = row_pattern.findall(line)
            fuzzy_array[i] = float(int_vals[0])
        return fuzzy_array

    def _get_iodine_vmin_vmax(self) -> np.ndarray | None:
        """Extract ESP Vmin/Vmax (kcal/mol) on the molecular surface for iodine atoms.

        Returns
        -------
        np.ndarray | None
            If one iodine atom is present: array with shape (2,) -> [Vmin_kcal, Vmax_kcal].
            If multiple iodines: array with shape (n_I, 2) -> [[Vmin, Vmax], ...].
            If no iodine or table not found: None.
        """
        # 1) Find which atom indices are iodine from the Atom list
        atom_elem_pattern = re.compile(
          r"Atom\s+(\d+)\s*\(\s*([A-Z][a-z]?)\s*\)"
        )
        iodine_indices = sorted(
            {
                int(idx_str)
                for idx_str, sym in atom_elem_pattern.findall(self.content)
                if sym == "I"
            }
        )
        if not iodine_indices:
            return None

        # 2) Locate the per-atom ESP min/max table
        # Header in your file:
        # Note: Minimal and maximal value below are in kcal/mol
        #  Atom#    All/Positive/Negative area (Ang^2)  Minimal value   Maximal value
        marker = "Note: Minimal and maximal value below are in kcal/mol"
        start = self.content.find(marker)
        if start == -1:
            return None

        # Take the block from that marker onward and parse line by line
        block = self.content[start:].splitlines()

        # Find the line with "Atom#" (column header)
        header_idx = None
        for i, line in enumerate(block):
            if "Atom#" in line:
                header_idx = i
                break
        if header_idx is None:
            return None

        iodine_v = []

        # Data lines start just after column header
        for line in block[header_idx + 1 :]:
            stripped = line.strip()
            # Stop at the blank line before the next "Note:"
            if not stripped:
                break
            if not stripped[0].isdigit():
                # safety: if line doesn't start with an atom index, stop
                break

            # Example line:
            #  2     46.56182     6.65735    39.90447    -12.05835640     14.14485849
            parts = stripped.split()
            if len(parts) < 6:
                continue
            idx = int(parts[0])
            if idx in iodine_indices:
                vmin = float(parts[-2])  # second-to-last column
                vmax = float(parts[-1])  # last column
                iodine_v.append((vmin, vmax))

        if not iodine_v:
            return None

        iodine_v = np.array(iodine_v, dtype=np.float64)
        if iodine_v.shape[0] == 1:
            return iodine_v[0]
        return iodine_v

    def get_array_dict(self) -> dict:  # See this I can use to replace dump_mwfn_results
        """Create a dictionary with result arrays for the mwfn output file.

        Args:
            molecule_id (str): Unique identifier for the molecule.
            conformer_id (int, optional): Identifier for the conformer. Defaults to 0.

        Returns:
            array_dict (dict): dictionary with results
        """
        array_dict = {
            "partitioning_scheme": self._partitioning_scheme,
            "num_atoms": self._num_atoms,
            "charges": self._charge_array,
            "dipoles": self._dipole_array,
            "quadrupoles_sph": self._quadrupole_array,
            "octopoles_sph": self._octopole_array,
            "second_radial_moment": self._r_2_array,
            "overlap_matrix": self._overlap_matrix,
            "c6": self._c6_array,
            "polarisabilities": self._polarisability_array,
            "effective_volume": self._effective_volume_array,
            "fuzzy_space_integral": self._fuzzy_space_array,
            "iodine_vmin_vmax": self._iodine_vmin_vmax,
        }
        return array_dict

    def _get_partitioning_scheme(self):
        split_pattern = re.compile(
            r"Select method for partitioning atomic spaces, current\:\s*(\w+)"
        )
        partitioning_schemes = split_pattern.findall(self.content)
        return partitioning_schemes[1]

    @property
    def metadata(self) -> dict:
        """Metadata of the mwfn output file."""
        meta_dict = {
            "file_name": self.out_file,
            "num_atoms": self._num_atoms,
            "charges": "Atomic charges in atomic units",
            "dipoles": "Atomic dipole moments in atomic units (X, Y, Z components)",
            "quadrupoles": "Atomic quadrupole moments in spherical harmonics (Q_2,0, Q_2,-1, Q_2,1, Q_2,-2, Q_2,2)",
            "octopoles": "Atomic octopole moments in spherical harmonics (Q_3,0, Q_3,-1, Q_3,1, Q_3,-2, Q_3,2, Q_3,-3, Q_3,3)",
            "r_2": "Spatial extent <r^2> for each atom in atomic units (X, Y, Z components)",
            "overlap_matrix": "Overlap matrix of fuzzy space integration values, sum of columns give atomic fuzzy space integration values",
            "c6": "Atomic C6 coefficients in atomic units based on Tkatchenko-Scheffler method",
            "polarisabilities": "Atomic polarizabilities in atomic units and their contributions to the total polarizability in percentage",
            "effective_volume": "Atomic effective volumes and free volumes in atomic units, reference WF obtained with r2scan-3c",
            "fuzzy_space_integral": "Fuzzy space integration values for each atom, sum of columns give total fuzzy space integration value",
            "iodine_vmin_vmax": "ESP Vmin/Vmax (kcal/mol) on the molecular surface for iodine.",
        }
        return meta_dict

    @property
    def num_atoms(self) -> int:
        """Number of atoms in the system."""
        return self._num_atoms

    @property
    def charges(self) -> np.ndarray:
        return self._charge_array

    @property
    def dipoles(self) -> np.ndarray:
        return self._dipole_array

    @property
    def quadrupoles(self) -> np.ndarray:
        return self._quadrupole_array

    @property
    def octopoles(self) -> np.ndarray:
        return self._octopole_array

    @property
    def r_2(self) -> np.ndarray:
        return self._r_2_array

    @property
    def overlap_matrix(self) -> np.ndarray:
        return self._overlap_matrix

    @property
    def c6(self) -> np.ndarray:
        return self._c6_array

    @property
    def polarisabilities(self) -> np.ndarray:
        return self._polarisability_array

    @property
    def effective_volume(self) -> np.ndarray:
        return self._effective_volume_array

    @property
    def fuzzy_space_integral(self) -> np.ndarray:
        return self._fuzzy_space_array

    @property
    def iodine_vmin_vmax(self) -> np.ndarray | None:
        return self._get_iodine_vmin_vmax()


def dump_mwfn_fuzzy_results(
    mw: MwfnFuzzySpaceOut, out_dir: Path | str | None = None
):  # Pull this into MWFNFuzzySpace
    """
    Save all MwfnFuzzySpaceOut arrays to text files in out_dir.

    Files written (if available):
      charges.dat
      dipoles.dat
      quadrupoles.dat
      octopoles.dat
      r2.dat
      c6.dat
      polarisabilities.dat
      effective_volume.dat
      fuzzy_integrals.dat
    """
    if out_dir is None:
        out_dir = Path(mw.out_file).parent
    else:
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savetxt(out_dir / "charges.dat", mw.charges)
    np.savetxt(out_dir / "dipoles.dat", mw.dipoles)
    np.savetxt(out_dir / "quadrupoles.dat", mw.quadrupoles)
    np.savetxt(out_dir / "octopoles.dat", mw.octopoles)
    np.savetxt(out_dir / "r2.dat", mw.r_2)
    np.savetxt(out_dir / "c6.dat", mw.c6)
    np.savetxt(out_dir / "polarisabilities.dat", mw.polarisabilities)
    np.savetxt(out_dir / "effective_volume.dat", mw.effective_volume)
    np.savetxt(out_dir / "fuzzy_integrals.dat", mw.fuzzy_space_integral)


def create_mwfn_input(
    directory: Union[Path, str],
    atom_refs_base: Union[Path, str],
    mbis: bool = True,
    becke: bool = False,
    elements: list[str] = ["H", "C", "O", "S", "N", "I","Si"],
) -> Union[Path, str]:
    """Write mwfn input file for MBIS or Becke partitioning"""

    # Validate mutually exclusive parameters
    if mbis and becke:
        raise ValueError(
            "mbis and becke cannot both be True - they are mutually exclusive"
        )
    if not mbis and not becke:
        raise ValueError("Either mbis or becke must be True")

    atom_refs_base = Path(atom_refs_base)
    directory = Path(directory)

    # Build reference_densities from existing molden.input files
    reference_densities = "\n".join(
        str(atom_refs_base / element.lower() / f"{element.lower()}.molden.input")
        for element in elements
    )

    becke_lines = [
        "15",  # Enter function 15: Fuzzy Space analysis, Becke is default option
        "8",  # Select option 8: Integration over fuzzy overlap region for real space function
        "100",  # Selected real space function: 100, which is userdefined function, i.e. Weizsäcker kinetic energy
        "n",  # Outputting matrices in intovlp.txt?
        "13",  # Calculate atomic effective volume, free vol, polarisability, and C6 coeff
        reference_densities,
        "2",  # Calculate atomic and molecular multipole moments and r^2
        "1",  # Output on screen
        "0",  # Return to main menu
        "q",  # Quit program
        "EOF",  # End of file marker
    ]

    mbis_lines = [
        "15",  # Enter function 15: Fuzzy Space analysis, Becke is default option
        "-1",  # Select MBIS (Minimal Basis Iterative Stockholder) analysis
        "5",  # MBIS
        "3",  # choose convergence
        "0.000001",  # convergence criterion for atomic charges
        "1",  # start calculation
        "2",  # Calculate atomic and molecular multipole moments and r^2
        "1",  # print on screen
        "13",  # Calculate atomic effective volume, free vol, polarisability, and C6 coeff
        reference_densities,
        "1",  # perform integration in fuzzy atomic space for real space function
        "100",  # select userfunc, in this case Weizsäcker kinetic energy
        "0",  # return to main menu
        "5",  # output and plot specific property
        "1",  # Electron density (Maybe delete??)
        "4",  # grid spacing
        "0.3333",  # grid spacing in bohrs
        "2",  # export to gaussian cube file
        "0",  # return to main menu
        "5",  # output and plot specific property
        "12", # ESP (Maybe delete??)
        "4",  # grid spacing
        "0.3333",  # grid spacing in bohrs
        "0",  # return to main menu
        "12",  # Quantitative analysis of molecular surface
        "0",  # Start analysis
        "11",  # output surface properties of each atom
        "y",  # outputting the surface facets to locsurf.pqr in current folder
        "q",  # Quit program
        "EOF",  # End of file marker
    ]

    if mbis:
        lines = mbis_lines
        file_name = f"{directory}/mwfn_mbis.inp"
    else:  # becke is True
        lines = becke_lines
        file_name = f"{directory}/mwfn_becke.inp"

    input_string = "\n".join(lines) + "\n"

    with open(file_name, "w") as f:
        f.write(input_string)
    return Path(file_name)


# Read and understand how this function works
def run_mwfn(
    input_file: Union[str, Path],
    molden_file: Union[str, Path],
    mwfn_exec: str = "mwfn",
    num_thread: int = 12,
    ignore_fortran_errors: bool = True,
    working_dir: Optional[Union[str, Path]] = None,
    timeout: Optional[int] = None,
) -> subprocess.CompletedProcess:
    """Run mwfn using molden file obtained from orca gbw

    Args:
        input_file (Union[str, Path]): file name of input file generated using create_mwfn_input
        molden_file (Union[str, Path]): wavefunction file
        mwfn_exec (str, optional): how to call multiwfn. Defaults to "mwfn".
        num_thread (int, optional): Number of threads used by mwfn
        working_dir (Optional[Union[str, Path]], optional): path to where the molden and input file are stored Defaults to None.
        timeout (Optional[int], optional): Time after which the calculation is aborted. Defaults to None.

    Returns:
        subprocess.CompletedProcess: Completed mwfn job.
    """
    input_file = Path(input_file)
    molden_file = Path(molden_file)
    output_file = input_file.with_suffix(".out")

    # check if files exist
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    if not molden_file.exists():
        raise FileNotFoundError(f"Molden file not found: {molden_file}")

    # set working dir
    if working_dir is None:
        working_dir = molden_file.parent
    else:
        working_dir = Path(working_dir)

    # Set up environment variables for OpenMP and stack size
    env = os.environ.copy()
    env["OMP_STACKSIZE"] = "1000M"
    env["OMP_NUM_THREADS"] = str(num_thread)

    # prepare cmd - fix the thread argument format
    cmd = [mwfn_exec, str(molden_file), "-nt", str(num_thread)]

    try:
        # Set stack size limit using preexec_fn (Linux only)
        def set_limits():
            import resource

            # Set stack size to unlimited
            resource.setrlimit(
                resource.RLIMIT_STACK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY)
            )

        with open(input_file, "r") as inp, open(output_file, "w") as out:
            result = subprocess.run(
                cmd,
                stdin=inp,
                stdout=out,
                stderr=subprocess.PIPE,
                text=True,
                cwd=working_dir,
                timeout=timeout,
                check=False,
                env=env,
                preexec_fn=set_limits,
            )
        if result.returncode != 0:
            stderr_text = result.stderr.lower()

            # Check for common Fortran I/O errors that indicate completion
            fortran_io_errors = [
                "list-directed i/o syntax error",
                "forrtl: severe (59)",
                "end of file",
                "invalid input",
            ]

            is_fortran_io_error = any(
                error in stderr_text for error in fortran_io_errors
            )

            if ignore_fortran_errors and is_fortran_io_error:
                print(
                    "Multiwfn completed with Fortran I/O error (likely normal termination)"
                )
                print(f"Return code: {result.returncode}")
                print(f"Output written to: {output_file}")
                # Don't raise exception, just return the result
                return result
            else:
                # For other types of errors, still raise exception
                print(f"Multiwfn failed with return code {result.returncode}")
                print(f"STDERR: {result.stderr}")
                raise subprocess.CalledProcessError(
                    result.returncode, cmd, result.stdout, result.stderr
                )
        else:
            print("Multiwfn completed successfully")
            print(f"Output written to: {output_file}")

        return result
    except subprocess.CalledProcessError as e:
        print(f"Multiwfn failed with return code {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        raise
    except subprocess.TimeoutExpired:
        print(f"Multiwfn execution timed out after {timeout} seconds")
        raise
    except FileNotFoundError:
        print(f"Multiwfn executable ot found: {mwfn_exec}")
        print("Make sure Multiwfn is installed and in your PATH")
        raise

def align_ci_xaxis_no_rdkit(
    structure: Structure, return_i_index: bool = False
) -> Union[Structure, tuple[Structure, int]]:
    """Align structure so the C-I bond defines the axis without RDKit.
    This uses covalent radii to find the iodine-bonded atom and a second
    neighbor to define the plane for alignment.
    """
    def _symbol(atom_obj) -> str:
        elem = atom_obj.element
        if isinstance(elem, str):
            return elem
        if hasattr(elem, "symbol"):
            return elem.symbol
        return str(elem)
    atoms = structure.atoms
    if not atoms:
        raise ValueError("Structure has no atoms.")
    symbols = [_symbol(atom) for atom in atoms]
    coords = np.array([atom.coordinates.coordinates for atom in atoms], dtype=float)
    i_indices = [idx for idx, sym in enumerate(symbols) if sym == "I"]
    if len(i_indices) != 1:
        raise ValueError(f"Expected exactly one iodine, found {len(i_indices)}.")
    i_index = i_indices[0]
    i_radius = covalent_radii[atomic_numbers[symbols[i_index]]]
    scale = 1.25
    bonded_candidates = []
    for idx, sym in enumerate(symbols):
        if idx == i_index:
            continue
        d_ij = float(np.linalg.norm(coords[i_index] - coords[idx]))
        cutoff = scale * (i_radius + covalent_radii[atomic_numbers[sym]])
        if d_ij <= cutoff:
            bonded_candidates.append((d_ij, idx))
    if bonded_candidates:
        c_index = min(bonded_candidates, key=lambda x: x[0])[1]
    else:
        # Fallback: nearest neighbor to iodine.
        all_distances = [
            (float(np.linalg.norm(coords[i_index] - coords[idx])), idx)
            for idx in range(len(atoms))
            if idx != i_index
        ]
        if not all_distances:
            raise ValueError("Could not find any atom bonded to iodine.")
        c_index = min(all_distances, key=lambda x: x[0])[1]
    # Find a second neighbor of the carbon to define the plane.
    c_radius = covalent_radii[atomic_numbers[symbols[c_index]]]
    neighbor_candidates = []
    for idx, sym in enumerate(symbols):
        if idx in (i_index, c_index):
            continue
        d_c = float(np.linalg.norm(coords[c_index] - coords[idx]))
        cutoff = scale * (c_radius + covalent_radii[atomic_numbers[sym]])
        if d_c <= cutoff:
            neighbor_candidates.append((d_c, idx))
    if neighbor_candidates:
        c2_index = min(neighbor_candidates, key=lambda x: x[0])[1]
    else:
        # Fallback: choose nearest non-iodine atom to the carbon.
        all_distances = [
            (float(np.linalg.norm(coords[c_index] - coords[idx])), idx)
            for idx in range(len(atoms))
            if idx not in (i_index, c_index)
        ]
        if not all_distances:
            raise ValueError("Could not find a second atom to define the plane.")
        c2_index = min(all_distances, key=lambda x: x[0])[1]
    c_coords = coords[c_index]
    i_coords = coords[i_index]
    c2_coords = coords[c2_index]
    rotation_matrix = get_alignment_matrix(i_coords, c_coords, c2_coords)
    rotated_coords = apply_transformation(coords=coords, T=rotation_matrix)
    new_atoms = []
    for i, atom in enumerate(atoms):
        new_atom = Atom(element=atom.element, coordinates=rotated_coords[i])
        new_atoms.append(new_atom)
    if return_i_index:
        return Structure(new_atoms), i_index
    return Structure(new_atoms)

def attach_fragment_most_polar_bond_along_CI_axis(opt_struct, frag_smiles, distance):
    """
    opt_struct: ORCA Structure-like object with .atoms, .charge, .multiplicity, .smiles
    frag_smiles: SMILES of fragment to place
    distance: distance (in Å) from iodine along +z where the more negative end
              of the most polar fragment bond will be placed

    Assumes the host C–I bond is already aligned along +z such that I has
    the larger z-coordinate.
    """

    # ========== 1) Build RDKit mol for host ==========
    coords = np.array([atom.coordinates.coordinates for atom in opt_struct.atoms])
    if hasattr(opt_struct, "smiles"):
        host = Chem.MolFromSmiles(opt_struct.smiles)
        host = Chem.AddHs(host)
        if host.GetNumAtoms() != len(coords):
            raise ValueError(
                "Number of atoms in host SMILES does not match opt_struct.atoms."
            )
        conf = Chem.Conformer(host.GetNumAtoms())
        for i in range(host.GetNumAtoms()):
            x, y, z = coords[i]
            conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
        host.RemoveAllConformers()
        host.AddConformer(conf, assignId=True)
    else:
        xyz = opt_struct.to_xyz_block()
        host = xyzblock_to_rd_mol(
            xyz,
            total_charge=getattr(opt_struct, "charge", 0),
        )

    conf_host = host.GetConformer()

    # ========== 2) Find iodine (C–I already along +z) ==========
    i_indices = [a.GetIdx() for a in host.GetAtoms() if a.GetSymbol() == "I"]
    if len(i_indices) != 1:
        raise ValueError(f"Expected exactly one iodine, found {len(i_indices)}.")
    i_idx = i_indices[0]
    i_pos = conf_host.GetAtomPosition(i_idx)
    z_I = i_pos.z

    # ========== 3) Build fragment and compute Gasteiger charges ==========
    frag = Chem.MolFromSmiles(frag_smiles)
    frag = Chem.AddHs(frag)
    AllChem.EmbedMolecule(frag)
    conf_frag = frag.GetConformer()

    AllChem.ComputeGasteigerCharges(frag)

    # ========== 4) Find most polar bond (max |Δq|) ==========
    bond_polarities = []
    for bond in frag.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        qi = float(frag.GetAtomWithIdx(i).GetProp("_GasteigerCharge"))
        qj = float(frag.GetAtomWithIdx(j).GetProp("_GasteigerCharge"))
        dq = abs(qi - qj)
        bond_polarities.append((bond.GetIdx(), i, j, dq))

    if not bond_polarities:
        raise ValueError("Fragment has no bonds to define a polar bond.")

    _, i_pol, j_pol, _ = max(bond_polarities, key=lambda x: x[3])

    # choose more negative atom as anchor, other as partner
    qi = float(frag.GetAtomWithIdx(i_pol).GetProp("_GasteigerCharge"))
    qj = float(frag.GetAtomWithIdx(j_pol).GetProp("_GasteigerCharge"))
    if qi < qj:
        anchor_idx = i_pol   # more negative
        other_idx  = j_pol
    else:
        anchor_idx = j_pol
        other_idx  = i_pol

    anchor_pos = conf_frag.GetAtomPosition(anchor_idx)
    other_pos  = conf_frag.GetAtomPosition(other_idx)

    v_frag = np.array([other_pos.x - anchor_pos.x,
                       other_pos.y - anchor_pos.y,
                       other_pos.z - anchor_pos.z])
    v_norm = np.linalg.norm(v_frag)
    if v_norm == 0.0:
        raise ValueError("Zero-length polar bond vector in fragment.")
    u_frag = v_frag / v_norm          # current bond direction
    u_z = np.array([0.0, 0.0, 1.0])   # target direction (C–I already along +z)

    # ========== 5) Build rotation to align u_frag -> u_z ==========
    c = np.dot(u_frag, u_z)
    if np.isclose(c, 1.0):
        R = np.eye(3)  # already along +z
    elif np.isclose(c, -1.0):
        # 180° rotation around axis ⟂ u_frag
        axis = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(axis, u_frag)) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        axis = axis / np.linalg.norm(axis)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R = np.eye(3) + 2 * K @ K
    else:
        v = np.cross(u_frag, u_z)
        s = np.linalg.norm(v)
        v = v / s
        K = np.array([[0, -v[2], v[1]],
                      [v[2], 0, -v[0]],
                      [-v[1], v[0], 0]])
        R = np.eye(3) + K * s + K @ K * ((1 - c) / (s**2))

    # rotate fragment around anchor
    frag_coords = []
    for i in range(frag.GetNumAtoms()):
        p = conf_frag.GetAtomPosition(i)
        frag_coords.append([p.x, p.y, p.z])
    frag_coords = np.array(frag_coords)

    anchor_vec = np.array([anchor_pos.x, anchor_pos.y, anchor_pos.z])
    frag_coords_rot = (frag_coords - anchor_vec) @ R.T + anchor_vec

    for i in range(frag.GetNumAtoms()):
        x, y, z = frag_coords_rot[i]
        conf_frag.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))

    # ========== 6) Translate so anchor is at z_I + distance on the C–I axis ==========
    target = np.array([i_pos.x, i_pos.y, z_I + distance])
    anchor_pos_new = conf_frag.GetAtomPosition(anchor_idx)
    trans = target - np.array([anchor_pos_new.x,
                               anchor_pos_new.y,
                               anchor_pos_new.z])

    for i in range(frag.GetNumAtoms()):
        p = conf_frag.GetAtomPosition(i)
        new_p = Point3D(
            float(p.x + trans[0]),
            float(p.y + trans[1]),
            float(p.z + trans[2]),
        )
        conf_frag.SetAtomPosition(i, new_p)

    # ========== 7) Merge host and fragment, copy coords ==========
    combo = Chem.CombineMols(host, frag)
    conf_combo = Chem.Conformer(combo.GetNumAtoms())

    # host atoms
    for i in range(host.GetNumAtoms()):
        p = conf_host.GetAtomPosition(i)
        conf_combo.SetAtomPosition(i, p)

    # fragment atoms (offset)
    offset = host.GetNumAtoms()
    for i in range(frag.GetNumAtoms()):
        p = conf_frag.GetAtomPosition(i)
        conf_combo.SetAtomPosition(offset + i, p)

    combo.RemoveAllConformers()
    combo.AddConformer(conf_combo, assignId=True)

    # ========== 8) Convert back to ORCA Structure ==========
    AtomCls = opt_struct.atoms[0].__class__
    CoordCls = opt_struct.atoms[0].coordinates.__class__

    new_atoms = []
    for idx, atom in enumerate(combo.GetAtoms()):
        p = conf_combo.GetAtomPosition(idx)
        elem = atom.GetSymbol()
        coord_vec = np.array([p.x, p.y, p.z], dtype=float)
        coord_obj = CoordCls(coord_vec)
        new_atom = AtomCls(element=elem, coordinates=coord_obj)
        new_atoms.append(new_atom)

    new_struct = opt_struct.__class__(atoms=new_atoms)
    # adjust charge if fragment is charged; here assumed neutral
    new_struct.charge = getattr(opt_struct, "charge", 0)
    new_struct.smiles = getattr(opt_struct, "smiles", "")
    new_struct.multiplicity = getattr(opt_struct, "multiplicity", 1)

    return new_struct
