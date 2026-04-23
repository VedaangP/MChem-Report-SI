# MChem-Report-SI
Supporting Information for the MChem Report written by Vedaang Sunil Parammal, student ID:11106372

## Folder Overview

Scaffold-Pipelines:
Opi workflow used to generate the ASE databases for the isolated molecules and halogen bonding complexes

ASE-databases:
Contains the ASE databases which have been calculated as described in section 2.1 and 2.2 in the report.

Input-parameter-filtering:
Python scripts used to check possible input parameters and filter them to optimize the ML model input parameters

ML-scripts: 
Python scripts used to train and test the multi-linear regression and gradient boosting models

## Setting up 
Certain programmes and modules are needed to run the code in the repository

### Prerequisites
- ORCA 6.1.1
- Python enviroment with the following packages:
  - opi
  - numpy
  - pandas
  - rdkit
  - ase
  - sklearn
  - plotly
  - mathplotlib
  - json
  - bayes_opt
  - scipy
  - openbabel
