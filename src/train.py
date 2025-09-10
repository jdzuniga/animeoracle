from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_absolute_error, root_mean_squared_error, r2_score
import json
import joblib
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from src import preprocess
from src.config import DATA_DIR, MODELS_DIR, RUN_DATE, TARGET_VARIABLE


def load_data():
    """Load the cleaned released anime parquet file."""
    root = Path(__file__).resolve().parent.parent
    file_path = root / DATA_DIR / RUN_DATE / 'anime_released_cleaned.parquet'
    anime = pd.read_parquet(file_path)
    return anime


def make_train_valid_split(anime_released, training_year_window=4, valid_year_window=1):
    current_year = datetime.strptime(RUN_DATE, "%Y-%m-%d").date().year
    anime_released_years = pd.to_datetime(anime_released['datetime']).dt.year

    valid = anime_released[(anime_released_years < current_year) & (anime_released_years >= current_year - valid_year_window)]
    train = anime_released[(anime_released_years < current_year - valid_year_window) 
                            & (anime_released_years >= current_year - valid_year_window - training_year_window)]

    X_valid, y_valid = valid.drop(TARGET_VARIABLE, axis=1), valid[TARGET_VARIABLE]
    X_train, y_train = train.drop(TARGET_VARIABLE, axis=1), train[TARGET_VARIABLE]
    
    print(X_valid['datetime'].max(), X_valid['datetime'].min())
    print(f'Valid size: {X_valid.shape[0]}')
    
    return X_train, y_train, X_valid, y_valid


def make_predefined_split(X_train, X_valid):
    """Create PredefinedSplit for GridSearchCV."""
    split_index = np.concatenate([
        -1 * np.ones(len(X_train), dtype=int),
        np.zeros(len(X_valid), dtype=int)
    ])
    return PredefinedSplit(test_fold=split_index)


def build_pipeline():
    """Build ML pipeline with preprocessing + LGBM."""
    model = lgb.LGBMRegressor(objective='regression_l2', random_state=42, n_jobs=-1)
    return Pipeline([
        ('preprocessor', preprocess.preprocessor),
        ('lgb', model)
    ])

def get_param_grid():
    """Return hyperparameter grid for tuning."""
    param_grid = {
        "lgb__bagging_fraction": [1.0],
        "lgb__colsample_bytree": [0.6],
        "lgb__feature_fraction": [1.0],
        "lgb__learning_rate": [0.05, 0.1],
        "lgb__max_depth": [20],
        "lgb__min_child_samples": [20],
        "lgb__n_estimators": [200, 500],
        "lgb__num_leaves": [31],
        "lgb__subsample": [0.6]
    }
    param_grid_extended = {
        'lgb__n_estimators': [200, 500, 750],
        'lgb__learning_rate': [0.01, 0.05, 0.1],
        'lgb__num_leaves': [31, 63],
        'lgb__max_depth': [10, 20],
        'lgb__min_child_samples': [10, 20],
        'lgb__subsample': [0.6, 0.8],
        'lgb__colsample_bytree': [0.6, 1.0],
        "lgb__feature_fraction": [0.8, 1.0],
        "lgb__bagging_fraction": [0.8, 1.0],
    }
    return param_grid

def run_grid_search(pipeline, param_grid, X, y, ps):
    scoring = {
        "MAE": make_scorer(mean_absolute_error, greater_is_better=False),
        "RMSE": make_scorer(root_mean_squared_error, greater_is_better=False),
        "R2": make_scorer(r2_score)
    }

    """Run grid search with PredefinedSplit."""
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scoring,
        refit="RMSE",
        cv=ps,
        verbose=2,
        n_jobs=-1
    )
    grid_search.fit(X, y)
    return grid_search


def create_directory() -> None:
    root = Path(__file__).resolve().parent.parent
    (root / MODELS_DIR / RUN_DATE).mkdir(parents=True, exist_ok=True)


def save_model(model):
    root = Path(__file__).resolve().parent.parent
    file_path = root / MODELS_DIR / RUN_DATE / 'model.pkl'
    joblib.dump(model, file_path)


def save_hyperparameters(hyperparams):
    root = Path(__file__).resolve().parent.parent
    file_path = root / MODELS_DIR / RUN_DATE / 'hyperparams.json'
    with open(file_path, 'w') as f:
        json.dump(hyperparams, f, indent=4)


def save_performance(performance):
    root = Path(__file__).resolve().parent.parent
    file_path = root / MODELS_DIR / RUN_DATE / 'performance.json'
    with open(file_path, 'w') as f:
        json.dump(performance, f, indent=4)


def run():
    anime_released = load_data()

    X_train, y_train, X_valid, y_valid = make_train_valid_split(anime_released)

    X = pd.concat([X_train, X_valid])
    y = pd.concat([y_train, y_valid])
    ps = make_predefined_split(X_train, X_valid)

    pipeline = build_pipeline()
    param_grid = get_param_grid()

    grid_search = run_grid_search(pipeline, param_grid, X, y, ps)

    create_directory()

    best_params = grid_search.best_params_
    save_hyperparameters(best_params)

    save_model(grid_search.best_estimator_)

    mae = -grid_search.cv_results_["mean_test_MAE"].max()
    rmse = -grid_search.cv_results_["mean_test_RMSE"].max()
    r2 = grid_search.cv_results_["mean_test_R2"].max()

    print("Best MAE: ", round(mae, 2))
    print("Best RMSE: ", round(rmse, 2))
    print("Best R2: ", round(r2, 2))

    performance = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }
    save_performance(performance)


if __name__ == '__main__':
    run()