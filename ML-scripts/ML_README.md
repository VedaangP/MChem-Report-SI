# ML scripts

The ML scripts are broadly split into 2 categories, the Multi-Linear Regression (MLR) and the Gradient Boosting (GB) scripts. Since there is very little difference between the individual MLR and GB scripts, first the general script structure for MLR and GB will be explained, followed by individual script-specific details.

## MLR 

### Steps: 
- Reads the isolated scaffold database and extracts the chosen input parameters and scaffold factors are assigned
- Extracts the interactions energies from the interacting complex database
- Builds a combined dataframe with the input features and interaction energies
- 80/20 holdout split for model training
- Numericals scaled using scikitlearn's Standard Scaler function
- K fold cross validation on training data
- Model trained on full training set

## GB

### Steps:
- Reads the isolated scaffold database and extracts the chosen input parameters and scaffold factors are assigned
- Extracts the interactions energies from the interacting complex database
- Builds a combined dataframe with the input features and interaction energies
- 80/20 holdout split for model training
- Numericals scaled using scikitlearn's Standard Scaler function
- Bayesian optimization of the model hyperparameters
- 10 5-fold cross validations
- Model trained on full training set

## Breakdown of individual scripts

### `GB_individual` and `MLR_individual` 

These are the scripts for training the model on individual databases
Output: Parity Plot

### `GB_Rem_BX_6` and `MLR_Rem_BX_6`

The BX-6 scaffolds are excluded from the training set and heldout for the test set.
Output: Parity Plot 

### `GB_Rem_Br` and `MLR_Rem_Br` 

The models are trained on the $\mathrm{F}^-$, $\mathrm{Cl}^-$ and $\mathrm{I}^-$ databases and the $\mathrm{Br}^-$ database is heldout as the test set. The acceptor input parameters are also added here, they are read from acceptor.db
Output: Parity Plot

### `MLR_Vmax_vs_Unified.py`

A single script to compare the Vmax and Unified MLR models. Both models are trained on acceptor parameters on top of the initial input parameters.
Output: 
- Parity Plot of Vmax vs Unified model
- Residual Plot of Vmax vs Unified model

### `GB_Vmax` and `GB_individual`

Trains GB Vmax and Unified models. Both models are trained on acceptor parameters on top of the initial input parameters.
Output: Parity Plot
