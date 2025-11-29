from pathlib import Path
import joblib
import pandas as pd
import numpy as np

from src import config

PREDICTIONS_DIR = config.PREDICTIONS_DIR
TARGET_VARIABLE = config.TARGET_VARIABLE


def create_directory():
    """ Create directory for saving predictions if it doesn't exist. """
    root = Path(__file__).resolve().parent.parent
    (root / PREDICTIONS_DIR).mkdir(exist_ok=True)
    (root / PREDICTIONS_DIR / config.RUN_DATE).mkdir(exist_ok=True)


def load_model_meta(name):
    """ Load the pre-trained model from a pickle file. """
    file_path = Path(__file__).resolve().parent.parent / config.MODELS_DIR / config.RUN_DATE / f'{name}_with_rmse.pkl'
    return joblib.load(file_path)


def get_weighted_prediction(X):
    models_pred = []
    models_inv_rmse = []
    for model_name in ['lgb', 'xgb', 'cat']:
        meta = load_model_meta(model_name)
        model = meta['model']
        models_inv_rmse.append(1 / meta['rmse'])

        y_pred = model.predict(X)
        models_pred.append(y_pred)

    weights = models_inv_rmse / np.sum(models_inv_rmse)
    pred_matrix = np.column_stack(models_pred)

    y_pred_blend = np.dot(pred_matrix, weights)
    y_pred_blend = [round(x, 2) for x in y_pred_blend]

    return y_pred_blend

def predict_airing():
    """ Predict scores for currently airing anime using the provided model. """
    file_path = Path(__file__).resolve().parent.parent / config.DATA_DIR / config.RUN_DATE / 'anime_airing_cleaned.parquet'
    anime_airing = pd.read_parquet(file_path)

    X = anime_airing.drop(TARGET_VARIABLE, axis=1)
    y = anime_airing[TARGET_VARIABLE]

    y_pred = get_weighted_prediction(X)

    airing_pred = pd.DataFrame(y_pred, index=X.index, columns=['predicted_score'])

    df = airing_pred.join(y, how='inner')
    df = df.join(X[['title', 'members', 'image_url']], how='inner')

    return df


def predict_unreleased():
    """ Predict scores for unreleased anime using the provided model. """
    file_path = Path(__file__).resolve().parent.parent / config.DATA_DIR / config.RUN_DATE / 'anime_unreleased_cleaned.parquet'
    anime_unreleased = pd.read_parquet(file_path)
    X = anime_unreleased.drop(TARGET_VARIABLE, axis=1)

    y_pred = get_weighted_prediction(X)

    unreleased = pd.DataFrame(y_pred, index=X.index, columns=['predicted_score'])

    anime_released_years = pd.DataFrame({'year': X['year']}, index=X.index)

    df = unreleased.join(X[['title', 'members', 'image_url']], how='inner').join(anime_released_years, how='inner')
    return df


def save_airing_predictions(predictions):
    """ Save predictions for currently airing anime to a CSV file. """
    root = Path(__file__).resolve().parent.parent
    file_path = root / PREDICTIONS_DIR / config.RUN_DATE / 'predictions_airing.csv'
    predictions.to_csv(file_path, index=True)


def save_airing_unreleased(predictions):
    """ Save predictions for unreleased anime to a CSV file. """
    root = Path(__file__).resolve().parent.parent
    file_path = root / PREDICTIONS_DIR / config.RUN_DATE / 'predictions_unreleased.csv'
    predictions.to_csv(file_path, index=True)


def keep_most_popular_anime(df, n):
    """ Keep only the top N most popular anime based on the number of members. """
    return df.nlargest(n, 'members')


def run():
    """ Main function to execute the prediction pipeline. """
    create_directory()

    airing_predictions = predict_airing()
    unreleased_predictions = predict_unreleased()

    airing_predictions = keep_most_popular_anime(airing_predictions, 50)
    unreleased_predictions = keep_most_popular_anime(unreleased_predictions, 50)

    save_airing_predictions(airing_predictions)
    save_airing_unreleased(unreleased_predictions)


if __name__ == '__main__':
    run()

