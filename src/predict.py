from pathlib import Path
import joblib
import pandas as pd

from src import config

PREDICTIONS_DIR = config.PREDICTIONS_DIR
TARGET_VARIABLE = config.TARGET_VARIABLE


def create_predictions_directory():
    """ Create directory for saving predictions if it doesn't exist. """
    root = Path(__file__).resolve().parent.parent
    (root / PREDICTIONS_DIR / config.RUN_DATE).mkdir(parents=True, exist_ok=True)


def load_model():
    """ Load the pre-trained model from a pickle file. """
    file_path = Path(__file__).resolve().parent.parent / config.MODELS_DIR / config.RUN_DATE / 'model.pkl'
    return joblib.load(file_path)


def predict_airing(model):
    """ Predict scores for currently airing anime using the provided model. """
    file_path = Path(__file__).resolve().parent.parent / config.DATA_DIR / config.RUN_DATE / 'anime_airing_cleaned.parquet'
    anime_airing = pd.read_parquet(file_path)
    X = anime_airing.drop(TARGET_VARIABLE, axis=1)
    y = anime_airing[TARGET_VARIABLE]
    predictions = model.predict(X)
    predictions = [round(x, 2) for x in predictions]

    airing_pred = pd.DataFrame(predictions, index=X.index, columns=['predicted_score'])

    df = airing_pred.join(y, how='inner')
    df = df.join(X[['title', 'members', 'image_url']], how='inner')

    return df


def predict_unreleased(model):
    """ Predict scores for unreleased anime using the provided model. """
    file_path = Path(__file__).resolve().parent.parent / config.DATA_DIR / config.RUN_DATE / 'anime_unreleased_cleaned.parquet'
    anime_unreleased = pd.read_parquet(file_path)
    X = anime_unreleased.drop(TARGET_VARIABLE, axis=1)
    predictions = model.predict(X)
    predictions = [round(x, 2) for x in predictions]

    unreleased = pd.DataFrame(predictions, index=X.index, columns=['predicted_score'])

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


def keep_most_popular_anime(df, members):
    """ Keep only the top N most popular anime based on the number of members. """
    return df.nlargest(members, 'members')


def run():
    """ Main function to execute the prediction pipeline. """
    create_predictions_directory()

    model = load_model()

    airing_predictions = predict_airing(model)
    unreleased_predictions = predict_unreleased(model)

    airing_predictions = keep_most_popular_anime(airing_predictions, 100)
    unreleased_predictions = keep_most_popular_anime(unreleased_predictions, 100)

    save_airing_predictions(airing_predictions)
    save_airing_unreleased(unreleased_predictions)


if __name__ == '__main__':
    run()

