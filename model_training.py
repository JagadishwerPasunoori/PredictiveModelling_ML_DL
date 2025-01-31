from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np

def train_models(X_train, y_train):
    """Train multiple machine learning models and return the best one."""
    models = {
        'RandomForest': RandomForestRegressor(),
        'GradientBoosting': GradientBoostingRegressor(),
        'LinearRegression': LinearRegression()
    }

    best_model = None
    best_score = np.inf

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_train)
        mse = mean_squared_error(y_train, y_pred)
        print(f"{name} MSE: {mse}")

        if mse < best_score:
            best_score = mse
            best_model = model

    return best_model

def hyperparameter_tuning(model, X_train, y_train):
    """Perform hyperparameter tuning using GridSearchCV."""
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }

    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_