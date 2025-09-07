from os import makedirs
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_absolute_error
import json
import joblib
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from src import preprocess
from src.config import DATA_DIR, MODELS_DIR, RUN_DATE, TARGET_VARIABLE


def load_data():
    """Load the cleaned released anime parquet file."""
    path = f"../{DATA_DIR}/{RUN_DATE}/anime_cleaned_released.parquet"
    return pd.read_parquet(path)

def make_train_valid_split(anime_released, training_year_window=4, valid_year_window=1):
    current_year = RUN_DATE.year
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
    model = lgb.LGBMRegressor(random_state=42, n_jobs=-1)
    return Pipeline([
        ('preprocessor', preprocess.preprocessor),
        ('lgb', model)
    ])

def get_param_grid():
    """Return hyperparameter grid for tuning."""
    param_grid = {
        'lgb__n_estimators': [200, 500],
        'lgb__learning_rate': [0.01, 0.05, 0.1],
        'lgb__num_leaves': [31, 63],          # default 31, higher = more complex
        'lgb__max_depth': [10, 20],       
        'lgb__min_child_samples': [10, 20, 50],
        'lgb__subsample': [0.6, 0.8, 1.0],         # row sampling
        'lgb__colsample_bytree': [0.6, 1.0],  # feature sampling
    }
    return param_grid

def run_grid_search(pipeline, param_grid, X, y, ps):
    """Run grid search with PredefinedSplit."""
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=make_scorer(mean_absolute_error, greater_is_better=False),
        cv=ps,
        verbose=2,
        n_jobs=-1
    )
    grid_search.fit(X, y)
    return grid_search

def create_model_directory() -> None:
    path = f'../{MODELS_DIR}/{RUN_DATE}'
    makedirs(path, exist_ok=True)

def save_model(model):
    joblib.dump(model, f'../{MODELS_DIR}/{RUN_DATE}/model.pkl')

def save_parameters(hyperparams):
    with open(f'../{MODELS_DIR}/{RUN_DATE}/hyperparams.json', 'w') as f:
        json.dump(hyperparams, f, indent=4)

def save_performance(performance):
    with open(f'../{MODELS_DIR}/{RUN_DATE}/performance.json', 'w') as f:
        json.dump({'mae': performance}, f, indent=4)


def run():
    anime_released = load_data()

    X_train, y_train, X_valid, y_valid = make_train_valid_split(anime_released)

    X = pd.concat([X_train, X_valid])
    y = pd.concat([y_train, y_valid])
    ps = make_predefined_split(X_train, X_valid)

    pipeline = build_pipeline()
    param_grid = get_param_grid()

    grid_search = run_grid_search(pipeline, param_grid, X, y, ps)

    create_model_directory()

    best_params = grid_search.best_params_
    save_parameters(best_params)

    save_model(grid_search.best_estimator_)

    print("Best MAE:", -grid_search.best_score_)
    save_performance(-grid_search.best_score_)


if __name__ == '__main__':
    run()