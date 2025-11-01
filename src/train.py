from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import BaseCrossValidator
import json
import joblib
from src import preprocess
from src import config
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

DATA_DIR = config.DATA_DIR
TARGET_VARIABLE = config.TARGET_VARIABLE
MODELS_DIR = config.MODELS_DIR


class RollingWindowCV(BaseCrossValidator):
    """
    Custom cross-validator for rolling window time series split based on years.
    Parameters:
        train_size (int): Number of years to include in the training set.
        n_splits (int): Number of splits to create.
    Yields:
        train indices, validation indices for each split.
    """
    def __init__(self, train_size=5, n_splits=3):
        self.train_size = train_size
        self.n_splits = n_splits

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        current_year = datetime.today().year
        years = np.sort([year for year in range(current_year - 2 - self.train_size - self.n_splits,
                                                current_year - 1)])

        for i in range(self.n_splits):
            train_start = i
            train_end = i + self.train_size
            valid_year = years[train_end]

            train_mask = X['year'].isin(years[train_start:train_end])
            valid_mask = X['year'] == valid_year

            train_idx = np.where(train_mask)[0]
            valid_idx = np.where(valid_mask)[0]

            yield train_idx, valid_idx


def load_data():
    """
    Load the cleaned anime data from a Parquet file.
    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    root = Path(__file__).resolve().parent.parent
    file_path = root / DATA_DIR / config.RUN_DATE / 'anime_released_cleaned.parquet'
    anime = pd.read_parquet(file_path)
    return anime


def build_pipeline(params=None):
    """
        Build a machine learning pipeline with preprocessing and LightGBM regressor.
        Args:
            params (dict, optional): Hyperparameters for the LightGBM regressor. Defaults to None.
    """
    default_hyperparams = {
        "subsample": 0.5,
        "num_leaves": 255,
        "n_estimators": 1500,
        "min_split_gain": 0.0,
        "min_child_samples": 5,
        "max_depth": 40,
        "max_bin": 63,
        "learning_rate": 0.01,
        "lambda_l2": 1.0,
        "lambda_l1": 1.0,
        "feature_fraction": 0.5,
        "colsample_bytree": 0.5,
        "boosting_type": "gbdt",
        "bagging_freq": 5,
        "bagging_fraction": 0.8
    }
    model = lgb.LGBMRegressor(objective='regression_l2', n_jobs=-1, **default_hyperparams)
    if params:
        model = lgb.LGBMRegressor(objective='regression_l2', n_jobs=-1, **params)
    return Pipeline([
        ('preprocessor', preprocess.preprocessor),
        ('lgb', model)
    ])

def get_param_grid():
    """
        Define the hyperparameter grid for RandomizedSearchCV.
        Returns:
            dict: Hyperparameter grid.
    """
    param_dist = {
        "lgb__n_estimators": [100, 200, 500, 750, 1000, 1500],
        "lgb__learning_rate": [0.01, 0.05, 0.1],
        "lgb__num_leaves": [15, 31, 63, 127, 255],
        "lgb__max_depth": [-1, 5, 10, 20, 40],
        "lgb__min_child_samples": [5, 10, 20, 50, 100],
        "lgb__subsample": [0.5, 0.6, 0.8, 1.0],
        "lgb__colsample_bytree": [0.5, 0.6, 0.8, 1.0],
        "lgb__feature_fraction": [0.5, 0.7, 0.8, 1.0],
        "lgb__bagging_fraction": [0.5, 0.7, 0.8, 1.0],
        "lgb__lambda_l1": [0.0, 0.1, 0.5, 1.0],
        "lgb__lambda_l2": [0.0, 0.1, 0.5, 1.0],
        "lgb__min_split_gain": [0.0, 0.01, 0.05],
        "lgb__max_bin": [63, 127, 255],
        "lgb__bagging_freq": [0, 1, 5],
        "lgb__boosting_type": ["gbdt", "dart"]
    }
    return param_dist

def run_grid_search(pipeline, param_grid, X, y, cv_training_window=5, cv_folds=3):
    """
    Perform hyperparameter tuning using RandomizedSearchCV with rolling window cross-validation.
    Args:
        pipeline (Pipeline): The machine learning pipeline.
        param_grid (dict): Hyperparameter grid for RandomizedSearchCV.
        X (pd.DataFrame): Feature DataFrame.
        y (pd.Series): Target variable Series.
        cv_training_window (int): Number of years to include in the training set for each split.
    Returns:
        RandomizedSearchCV: The fitted RandomizedSearchCV object.   
    """
    scoring = {
        "MAE": make_scorer(mean_absolute_error, greater_is_better=False),
        "RMSE": make_scorer(root_mean_squared_error, greater_is_better=False),
        "R2": make_scorer(r2_score)
    }

    grid_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        refit="RMSE",
        n_iter=300,
        scoring=scoring,
        cv=RollingWindowCV(train_size=cv_training_window, n_splits=cv_folds),
        verbose=2,
        n_jobs=-1
    )
    grid_search.fit(X, y)
    return grid_search


def create_directory():
    """ Create directory for saving models and results if it doesn't exist. """
    root = Path(__file__).resolve().parent.parent
    (root / MODELS_DIR).mkdir(exist_ok=True)
    (root / MODELS_DIR / config.RUN_DATE).mkdir(parents=True, exist_ok=True)


def save_hyperparams(hyperparams):
    """ Save the best hyperparameters to a JSON file. """
    root = Path(__file__).resolve().parent.parent
    file_path = root / MODELS_DIR / config.RUN_DATE / 'hyperparams.json'
    with open(file_path, 'w') as f:
        json.dump(hyperparams, f, indent=4)


def save_performance(performance):
    """ Save the model performance metrics to a JSON file. """
    root = Path(__file__).resolve().parent.parent
    file_path = root / MODELS_DIR / config.RUN_DATE / 'performance.json'
    with open(file_path, 'w') as f:
        json.dump(performance, f, indent=4)


def save_model(model):
    """ Save the trained model to a file using joblib. """
    root = Path(__file__).resolve().parent.parent
    file_path = root / MODELS_DIR / config.RUN_DATE / 'model.pkl'
    joblib.dump(model, file_path)


def evaluate_performance(data, params, training_window=5):
    """ Evaluate model performance on a hold-out test set. """
    current_year = datetime.today().year
    train = data[data['year'].between(current_year - training_window - 1, current_year - 2)]
    X_train = train.drop(TARGET_VARIABLE, axis=1)
    y_train = train[TARGET_VARIABLE]

    test = data[data['year'] == current_year - 1]
    X_test = test.drop(TARGET_VARIABLE, axis=1)
    y_test = test[TARGET_VARIABLE]

    pipeline = build_pipeline(params)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mae = round(mean_absolute_error(y_test, y_pred), 4)
    rmse = round(root_mean_squared_error(y_test, y_pred), 4)
    r2 = round(r2_score(y_test, y_pred), 4)

    return {'MAE': mae, 'RMSE': rmse, 'R2': r2}


def train_final_model(pipeline, data, training_window=5):
    """ Train the final model on the most recent year. """
    current_year = datetime.today().year
    train = data[data['year'].between(current_year - training_window, current_year)]

    X_train = train.drop(TARGET_VARIABLE, axis=1)
    y_train = train[TARGET_VARIABLE]
    model = pipeline.fit(X_train, y_train)
    print("model trained.")

    return model


def find_best_training_window(X, y, folds=3):
    training_windows = [i for i in range(16, 21)]
    best_window = None
    lowest_rmse = float("inf")
    best_hyperparams = None
    results = {}

    param_grid = get_param_grid()
    for window in training_windows:
        cv_pipeline = build_pipeline()
        grid_search = run_grid_search(cv_pipeline,
                                      param_grid,
                                      X, y,
                                      cv_training_window=window,
                                      cv_folds=folds)
        rmse = -grid_search.best_score_
        results[window] = round(float(rmse), 4)

        # Track best window
        if rmse < lowest_rmse:
            lowest_rmse = rmse
            best_window = window
            best_hyperparams = grid_search.best_params_

    return results, best_window, best_hyperparams


def run(param_search=False, window_search=False):
    """ Main function to execute the training pipeline. """
    create_directory()

    data = load_data()

    X = data.drop(TARGET_VARIABLE, axis=1)
    y = data[TARGET_VARIABLE]
    best_window = 10

    if window_search:
        results, best_window, best_hyperparams = find_best_training_window(X, y, folds=3)
        print(results)
        print(f"Best window = {best_window}")

    elif param_search:
        folds = 3
        cv_pipeline = build_pipeline()
        param_grid = get_param_grid()
        grid_search = run_grid_search(cv_pipeline,
                                      param_grid,
                                      X, y,
                                      cv_training_window=best_window,
                                      cv_folds=folds)

        best_hyperparams = grid_search.best_params_

        model_params = {k.split("__")[1]: v for k, v in best_hyperparams.items()}
        metrics = evaluate_performance(data, model_params, training_window=best_window)

        save_hyperparams(model_params)
        save_performance(metrics)
        print(metrics)

    final_pipeline = build_pipeline()
    final_model = train_final_model(final_pipeline, data, training_window=best_window)
    save_model(final_model)


if __name__ == '__main__':
    run()