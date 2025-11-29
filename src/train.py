from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
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
        window_size (int): Number of years to include in the training set.
        n_splits (int): Number of splits to create.
    Yields:
        train indices, validation indices for each split.
    """
    def __init__(self, window_size=2, n_splits=3):
        self.window_size = window_size
        self.n_splits = n_splits

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        current_year = datetime.today().year
        years = np.sort([year for year in range(current_year - self.window_size - self.n_splits - 1,
                                                current_year - 1)])

        for i in range(self.n_splits):
            train_start = i
            train_end = i + self.window_size
            valid_year = years[train_end]

            train_mask = X['year'].isin(years[train_start:train_end])
            valid_mask = X['year'] == valid_year

            train_idx = np.where(train_mask)[0]
            valid_idx = np.where(valid_mask)[0]
            
            # print(f'CV {i+1}: {years[train_start:train_end]} -> {valid_year}')

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


def build_pipeline(model):
    return Pipeline([
        ('preprocessor', preprocess.preprocessor),
        ('model', model)
    ])


def get_models():
    return {
        "lgb": LGBMRegressor(
            subsample=0.7,
            reg_lambda=0.5,
            reg_alpha=0.3,
            num_leaves=255,
            n_estimators=400,
            min_child_samples=5,
            max_depth=6,
            learning_rate=0.03,
            colsample_bytree=0.7,
            verbose=-1
        ),
        "xgb": XGBRegressor(
            subsample=0.9,
            reg_lambda=0.3,
            reg_alpha=1.0,
            n_estimators=1200,
            min_child_weight=4,
            max_depth=5,
            learning_rate=0.01,
            gamma=0,
            colsample_bytree=0.6
        ),
        "cat": CatBoostRegressor(
            random_strength=5,
            learning_rate=0.03,
            l2_leaf_reg=1,
            iterations=500,
            depth=8,
            bagging_temperature=0.25,
            verbose=False
        )
    }


def get_hyperparams():
    return {
        "lgb":
            {
                "model__n_estimators": [200, 400, 700, 1000, 1500],
                "model__learning_rate": [0.003, 0.01, 0.03, 0.05, 0.1],
                "model__num_leaves": [31, 63, 127, 255],
                "model__max_depth": [-1, 4, 6, 8, 12],
                "model__min_child_samples": [5, 10, 20, 40, 80],
                "model__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
                "model__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
                "model__reg_alpha": [0, 0.1, 0.3, 0.5, 1.0],
                "model__reg_lambda": [0, 0.1, 0.3, 0.5, 1.0],

            },
        "xgb":
            {
                "model__n_estimators": [300, 600, 900, 1200],
                "model__learning_rate": [0.003, 0.01, 0.03, 0.05, 0.1],
                "model__max_depth": [3, 4, 5, 6, 8],
                "model__min_child_weight": [1, 2, 4, 6, 10],
                "model__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
                "model__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
                "model__gamma": [0, 0.1, 0.3, 0.5, 1.0],
                "model__reg_alpha": [0, 0.1, 0.3, 1.0],
                "model__reg_lambda": [0.1, 0.3, 1.0, 3.0],
            },
        "cat":
            {
                "model__depth": [4, 5, 6, 8, 10],
                "model__learning_rate": [0.003, 0.01, 0.03, 0.05],
                "model__iterations": [500, 800, 1200, 1600],
                "model__l2_leaf_reg": [1, 3, 5, 7, 9, 15],
                "model__bagging_temperature": [0, 0.25, 0.5, 1.0],
                "model__random_strength": [1, 5, 10, 20],
            }
    }


def run_grid_search(X, y):
    models = get_models()
    hyperparams = get_hyperparams()

    scoring = {
        "MAE": make_scorer(mean_absolute_error, greater_is_better=False),
        "RMSE": make_scorer(root_mean_squared_error, greater_is_better=False),
        "R2": make_scorer(r2_score)
    }

    models_grid_search = {}

    for name, model in models.items():
        pipeline = build_pipeline(model)

        print(f"Grid Search on {name}...")
        grid_search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=hyperparams[name],
            refit="RMSE",
            n_iter=100,
            scoring=scoring,
            cv=RollingWindowCV(),
            verbose=0,
            n_jobs=-1
        )
        grid_search.fit(X, y)
        models_grid_search[name] = grid_search

    return models_grid_search


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


def save_model_meta(name, model, metrics):
    """ Save the trained model to a file using joblib. """
    root = Path(__file__).resolve().parent.parent
    file_path = root / MODELS_DIR / config.RUN_DATE / f'{name}_with_rmse.pkl'

    bundle = {
        'model': model,
        'rmse': metrics[name]['RMSE']
    }

    joblib.dump(bundle, file_path)


def evaluate_performance(data, training_window=6):
    """ Evaluate model performance on last year's data. """
    current_year = datetime.today().year
    train = data[data['year'].between(current_year - training_window - 1, current_year - 2)]
    X_train = train.drop(TARGET_VARIABLE, axis=1)
    y_train = train[TARGET_VARIABLE]

    test = data[data['year'] == current_year - 1]
    X_test = test.drop(TARGET_VARIABLE, axis=1)
    y_test = test[TARGET_VARIABLE]

    models = get_models()
    models_performance = {}
    models_pred = []
    models_inv_rmse = []
    for name, model in models.items():
        pipeline = build_pipeline(model)
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        models_pred.append(y_pred)

        rmse = round(root_mean_squared_error(y_test, y_pred), 4)
        mae = round(mean_absolute_error(y_test, y_pred), 4)
        r2 = round(r2_score(y_test, y_pred), 4)
        models_performance[name] = {'RMSE': rmse, 'MAE': mae, 'R2': r2}

        models_inv_rmse.append(1/rmse)

    weights = models_inv_rmse / np.sum(models_inv_rmse)
    pred_matrix = np.column_stack(models_pred)
    y_pred_blend = np.dot(pred_matrix, weights)

    mae = round(mean_absolute_error(y_test, y_pred_blend), 4)
    rmse = round(root_mean_squared_error(y_test, y_pred_blend), 4)
    r2 = round(r2_score(y_test, y_pred_blend), 4)

    models_performance["blend"] = {'RMSE': rmse, 'MAE': mae, 'R2': r2}

    return models_performance


def train_final_models(data, training_window=6):
    """ Train the final model on the most recent year. """
    current_year = datetime.today().year
    train = data[data['year'].between(current_year - training_window, current_year)]

    X_train = train.drop(TARGET_VARIABLE, axis=1)
    y_train = train[TARGET_VARIABLE]

    models = get_models()
    for name, model in models.items():
        pipeline = build_pipeline(model)
        pipeline.fit(X_train, y_train)

        yield name, pipeline


def run(grid_search=False):
    """ Main function to execute the training pipeline. """
    create_directory()

    data = load_data()

    X = data.drop(TARGET_VARIABLE, axis=1)
    y = data[TARGET_VARIABLE]

    if grid_search:
        grid_search_results = run_grid_search(X, y)
        hyperparams = {}
        for model_name, grid_search in grid_search_results.items():
            best_hyperparams = grid_search.best_params_
            hyperparams[model_name] = {k.split("__")[1]: v for k, v in best_hyperparams.items()}

        save_hyperparams(hyperparams)

    metrics = evaluate_performance(data)
    save_performance(metrics)

    for name, model in train_final_models(data):
        save_model_meta(name, model, metrics)


if __name__ == '__main__':
    run()