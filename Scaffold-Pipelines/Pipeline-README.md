# Scripts

#### `utils.py` - Function library

**Few Key Functions**
- **Calculator Setup**:
  - `get_opt_calculator()`: Creates ORCA optimization calculator with r2SCAN-3c
  - `get_sp_calculator()`: Creates single-point calculator with wb97m-v and X2C relativistic treatment
  - `create_mwfn_input()`: Sets up input file for Multiwfn calculation

- **Calculation Execution**:
  - `run_opt_calc()`: Executes geometry optimization
  - `run_sp_calc()`: Executes single-point energy calculation
  - `run_mwfn()`: Runs Multiwfn calculation

- **Data Processing**:
  - `get_mbis_from_prop_file()`: Extracts MBIS properties from ORCA output
  - `get_iodine_mbis()`: Specifically extracts iodine atomic properties
  - `align_ci_xaxis()`: Aligns molecules with C-I bond along z-axis for consistent analysis
  - `add_anion_on_ci_axis_structure`: Adds monoatomic anion along +z from iodine at a given distance
  - `attach_fragment_most_polar_bond_along_CI_axis`: Aligns the most polar bond on the halogen acceptor with the C-I bond along the z-axis
  - MwfnFuzzySpaceOut class: Parses Multiwfn output and holds properties
 
#### `run.sh` - SLURM Job Submission Script

**Features**:

- Configured for SLURM workload manager
- Sets up ORCA 6.1.0 and Anaconda environment
- Manages scratch disk usage for temporary files
- Automatically copies results back to submission directory
- Optimized for CSF3 cluster architecture

**Key Settings**:

- 12 cores, single node
- 7-day time limit
- Loads required modules (ORCA, Anaconda)
- Activates conda environment named "opi"

#### `opi_wf.ipynb` - Interactive Analysis Notebook

- **Data Loading**: Reads the ase database file with calculated properties
- **Property Extraction**: Organizes atomic multipole moments:
   - Monopole (atomic charge)
   - Dipole moments (3 components)
   - Quadrupole moments (5 spherical components)
   - ESP (Electrostatic Potential) maxima and minima
   - Interaction Energy
- **Visualization**: Creates interactive plots using Plotly:
   - Scatter plots of atomic properties vs ESP maxima
   - Scatter plots of atomic properties vs Interaction energies
   - Heatmaps showing correlations between different properties
   - Component-wise analysis of multipole moments

### Atom-refs - Free atom reference densities

- Contains free atom reference densities for all the atoms found in the molecules/complexes
- Read by the opi pipeline scripts and used to build reference densities for the molecule/complex

### OPI Pipeline scripts

#### `opi_wf_isolated_scaffold.py` - Primary OPI workflow script used for isolated scaffolds

**What it does**:
- Reads SMILES.csv, which contains the SMILES and names of 96 different aromatic iodine compounds
- For each molecule:
  1. Generates 3D structure from SMILES strings
  2. Runs geometry optimization using r2SCAN-3c functional
  3. Aligns the molecule with the C-I bond along the z-axis
  4. Performs single-point calculation with wb97m-v functional and relativistic corrections
  5. Extracts MBIS (Minimal Basis Iterative Stockholder) atomic properties for iodine
  6. Adds properites from ORCA and MWFN to an ASE database
 
#### `opi_wf_PC.py` and `opi_wf_non_PC.py`- Script used for interacting complexes

**What it does**:
- Reads the database written by `opi_wf_isolated_scaffold.py`
- For each entry in the database:
  1. Pulls the optimized structure
  2. Adds halogen bond acceptor along the z-axis in line with the C-I bond
  3. Runs geometry optimization on the interacting complex using r2SCAN-3c functional
  4. Performs single-point calculation with wb97m-v functional and relativistic corrections
 
#### Usage

```bash
python opi_wf.py -c <cores> -wrkd <working_directory> -subd <submission_directory> --stom-refs <Atom-ref location>
```
