## `feature-test.py` : Collects the best feature combinations for each interaction database

**Steps:**
- Extracts acceptor features from the acceptor database
  - Polarizability of the acceptor
  - $r^2$ of the acceptor
- Extracts input features from the isolated scaffold database
  - Monopole
  - Dipole_0
  - Q_0
  - Q_2
  - $C_6$
  - $r^2$
  - Polarizability
- All 9 input parameters assembled into a single dataframe
- 80/20 test/train split
- MLR model ran on every each combination of all the input parameters => 511 combinations

**Output**
- Top 10 models ranked by cross validation $R^2$
- A csv file with every model tested and the following metrics for each model:
  - Train $R^2$ CV
  - Val $R^2$ CV
  - Gap (Train $R^2$ CV - Val $R^2$ CV)
  - CV RMSE
  - Adjusted $R^2$
  - AIC
  - BIC
  - Test $R^2$
  - Test RMSE

## `collective-feature-test.py` : Decides the best input parameters across the databases

**Prerequsities:**

Needs the .csv files generated from running `feature-test.py` on different interaction databases

**Usage:**

```bash
python pareto_analysis.py Cl_all_combinations.csv Br_all_combinations.csv
```
```bash
python pareto_analysis.py results/*_all_combinations.csv
```

**Steps:**
- Loops through all the csv files builds a dictionary keyed on the canonical feature string. Each entry stores the composite score, Val R², Test R², Gap, Test RMSE, and BIC from every database where that combination appears.
- Build a cross dataset dataframe and for each feature set, it computes
  - Coverage
  - Mean and minimum of Val $R^2$
  - Mean and minimum of Test $R^2$
- Computes a universal score for each feature set
- Holdout evaluation of the feature set

**Output:**
- `cross_dataset_rankings.csv` : All feature sets ranked by universal score
- `holdout_evaluation.txt` : Clean test set report for the best feature set 
