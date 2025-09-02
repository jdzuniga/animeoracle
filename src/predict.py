from os import makedirs
import joblib
import pandas as pd

from src.config import DATA_DIR, MODELS_DIR, RUN_DATE, PREDICTIONS_DIR, TARGET_VARIABLE



def create_predictions_directory() -> None:
    path = f'{PREDICTIONS_DIR}/{RUN_DATE}'
    makedirs(path, exist_ok=True)

def predict_airing(model):
    anime_airing = pd.read_parquet(f'../{DATA_DIR}/{RUN_DATE}/anime_cleaned_airing.parquet')
    X = anime_airing.drop(TARGET_VARIABLE, axis=1)
    predictions = pd.DataFrame(model.predict(X), index=X.index, columns=['predicted_score'])
    df = predictions.join(X['members'], how='inner')
    return df

def predict_unreleased(model):
    anime_unreleased = pd.read_parquet(f'../{DATA_DIR}/{RUN_DATE}/anime_cleaned_unreleased.parquet')
    X = anime_unreleased.drop(TARGET_VARIABLE, axis=1)
    predictions = pd.DataFrame(model.predict(X),index=X.index, columns=['predicted_score'])

    anime_released_years = pd.DataFrame({'year': pd.to_datetime(X['datetime']).dt.year},
                                        index=X.index)
    df = predictions.join(X['members'], how='inner').join(anime_released_years, how='inner')
    return df

def save_airing_predictions(predictions):
    predictions.to_csv(f'{PREDICTIONS_DIR}/{RUN_DATE}/predictions_airing.csv', index=True)

def save_airing_unreleased(predictions):
    predictions.to_csv(f'{PREDICTIONS_DIR}/{RUN_DATE}/predictions_unreleased.csv', index=True)

def keep_most_popular_anime(df, members):
    return df.nlargest(members, 'members')


def main():
    create_predictions_directory()

    model = joblib.load(f'{MODELS_DIR}/{RUN_DATE}/model.pkl')

    airing_predictions = predict_airing(model)
    unreleased_predictions = predict_unreleased(model)

    airing_predictions = keep_most_popular_anime(airing_predictions, 50)
    unreleased_predictions = keep_most_popular_anime(unreleased_predictions, 50)

    save_airing_predictions(airing_predictions)
    save_airing_unreleased(unreleased_predictions)


if __name__ == '__main__':
    main()

