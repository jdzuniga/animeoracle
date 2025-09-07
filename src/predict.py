from pathlib import Path
import joblib
import pandas as pd

from src.config import DATA_DIR, MODELS_DIR, RUN_DATE, PREDICTIONS_DIR, TARGET_VARIABLE


def create_directory() -> None:
    root = Path(__file__).resolve().parent.parent
    (root / PREDICTIONS_DIR / RUN_DATE).mkdir(parents=True, exist_ok=True)


def load_model():
    file_path = Path(__file__).resolve().parent.parent / MODELS_DIR / RUN_DATE / 'model.pkl'
    return joblib.load(file_path)


def predict_airing(model):
    file_path = Path(__file__).resolve().parent.parent / DATA_DIR / RUN_DATE / 'anime_airing_cleaned.parquet'
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
    file_path = Path(__file__).resolve().parent.parent / DATA_DIR / RUN_DATE / 'anime_unreleased_cleaned.parquet'
    anime_unreleased = pd.read_parquet(file_path)
    X = anime_unreleased.drop(TARGET_VARIABLE, axis=1)
    predictions = model.predict(X)
    predictions = [round(x, 2) for x in predictions]

    unreleased = pd.DataFrame(predictions, index=X.index, columns=['predicted_score'])

    anime_released_years = pd.DataFrame({'year': pd.to_datetime(X['datetime']).dt.year}, index=X.index)

    df = unreleased.join(X[['title', 'members', 'image_url']], how='inner').join(anime_released_years, how='inner')
    return df


def save_airing_predictions(predictions):
    root = Path(__file__).resolve().parent.parent
    file_path = root / PREDICTIONS_DIR / RUN_DATE / 'predictions_airing.csv'
    predictions.to_csv(file_path, index=True)


def save_airing_unreleased(predictions):
    root = Path(__file__).resolve().parent.parent
    file_path = root / PREDICTIONS_DIR / RUN_DATE / 'predictions_unreleased.csv'
    predictions.to_csv(file_path, index=True)


def keep_most_popular_anime(df, members):
    return df.nlargest(members, 'members')


def run():
    create_directory()

    model = load_model()

    airing_predictions = predict_airing(model)
    unreleased_predictions = predict_unreleased(model)

    airing_predictions = keep_most_popular_anime(airing_predictions, 50)
    unreleased_predictions = keep_most_popular_anime(unreleased_predictions, 50)

    save_airing_predictions(airing_predictions)
    save_airing_unreleased(unreleased_predictions)


if __name__ == '__main__':
    run()

