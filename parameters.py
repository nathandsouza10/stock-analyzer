# Random Forest
param_grid_rf = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],

}

# Gradient Boosting
param_grid_gb = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.001, 0.01, 0.1],
    'max_depth': [3, 4, 5]
}

# Linear Regression
param_grid_lr = {
    'fit_intercept': [True, False],
}

# Ridge Regression
param_grid_ridge = {
    'alpha': [0.001, 0.1, 1, 10],
    'fit_intercept': [True, False],
}

# Lasso Regression
param_grid_lasso = {
    'alpha': [0.001, 0.1, 1, 10],
    'fit_intercept': [True, False],
}

# Elastic Net
param_grid_en = {
    'alpha': [0.001, 0.1, 1],
    'l1_ratio': [0.2, 0.5, 0.7],
    'fit_intercept': [True, False],
}

# Support Vector Regression
param_grid_svr = {
    'C': [0.2, 0.5, 2],
    'epsilon': [0.05, 0.1, 0.5],
    'kernel': ['linear', 'poly', 'rbf']
}

# K-Nearest Neighbors
param_grid_knn = {
    'n_neighbors': [2, 7, 9],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

# Decision Tree
param_grid_dt = {
    'criterion': ['squared_error', 'absolute_error', 'friedman_mse'],
    'splitter': ['best', 'random'],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

param_grid_lstm = {
    'hidden_dim': [32, 64, 128],
    'num_layers': [1, 2, 3],
    'output_dim': [1],  # assuming you are doing regression with a single output
    'epochs': [50, 100, 1000],
    'lr': [0.01, 0.001, 0.0001]
}