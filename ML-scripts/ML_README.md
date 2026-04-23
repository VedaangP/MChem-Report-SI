## Multi-Linear Regression - `MLR.py`

### Steps: 
- Reads the isolated scaffold database and extracts the chosen input parameters and scaffold factors are assigned
- Extracts the interactions energies from the interacting complex database
- Builds a combined dataframe with the input features and interaction energies
- 80/20 holdout split for model training
- Numericals scaled using scikitlearn's Standard Scaler function
- K fold cross validation on training data
- Model trained on full training set

### Output:
- Training and test set $R^2$ and RMSE
- Parity Plot

## Gradient Boosting Model - `gb.py`

### Steps:
- Reads the isolated scaffold database and extracts the chosen input parameters and scaffold factors are assigned
- Reads halogen acceptor databases and extracts acceptor input features
- Extracts the interactions energies from the interacting complex database
- Builds a combined dataframe with the input features and interaction energies
- 80/20 holdout split for model training
- Numericals scaled using scikitlearn's Standard Scaler function
- Bayesian optimization of the model hyperparameters
- 10 5-fold cross validations
- Model trained on full training set

### Output:
- Training and test set $R^2$ and RMSE
- Parity Plot
- Residual Plot
- KDE plot of residuals
- Bayesian optimization convergence trace
