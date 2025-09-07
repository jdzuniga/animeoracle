import logging
from pathlib import Path
import pandas as pd
from src.config import TARGET_VARIABLE, DATA_DIR, RUN_DATE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fix_trailer_format(anime: pd.DataFrame) -> pd.DataFrame:
    anime['trailer'] = anime['trailer'].apply(lambda x: False if x['url'] is None else True)
    return anime


def remove_duplicates(anime: pd.DataFrame) -> pd.DataFrame:
    duplicated_mask = anime.duplicated(subset=['title'], keep='first') | anime.index.duplicated(keep='first')
    anime = anime[~duplicated_mask]
    logger.info(f"{duplicated_mask.sum()} duplicated rows dropped.\n")
    return anime


def remove_unsafe_ratings(anime: pd.DataFrame) -> pd.DataFrame:
    unsafe_ratings = anime['rating'] == "Rx - Hentai"
    anime = anime[~unsafe_ratings]
    logger.info(f"{unsafe_ratings.sum()} unsafe ratings dropped.\n")
    return anime


def remove_low_members_anime(anime: pd.DataFrame, quantile: float=0.2) -> pd.DataFrame:
    members_threshold = anime['members'].quantile(quantile)
    logger.info(f"{anime[anime['members'] <= members_threshold].shape[0]} anime below {members_threshold} members.")
    anime = anime[anime['members'] > members_threshold]
    return anime


def get_released_anime(anime: pd.DataFrame) -> pd.DataFrame:
    return anime[anime['status'] == 'Finished Airing']


def get_airing_anime(anime: pd.DataFrame) -> pd.DataFrame:
    return anime[anime['status'] == 'Currently Airing']


def get_unreleased_anime(anime: pd.DataFrame) -> pd.DataFrame:
    return anime[anime['status'] == 'Not yet aired']


def remove_unlabeled_anime(anime_released: pd.DataFrame) -> pd.DataFrame:
    logger.info(f'{anime_released['score'].isnull().sum()} missing labels removed from sample.')
    return anime_released.dropna(subset=[TARGET_VARIABLE])


def load_data():
    root = Path(__file__).resolve().parent.parent
    file_path = root / DATA_DIR / RUN_DATE / f'anime_raw.parquet'
    anime = pd.read_parquet(file_path)
    return anime


def create_parquet(anime: pd.DataFrame, file_name: str) -> None:
    root = Path(__file__).resolve().parent.parent
    file_path = root / DATA_DIR / RUN_DATE / f'{file_name}_cleaned.parquet'
    anime.to_parquet(file_path, engine="pyarrow", compression='snappy')
    logger.info(f'Cleaned data saved at {file_path}.')


def run() -> None:
    anime = load_data()
    anime.set_index('mal_id', inplace=True)
    anime = fix_trailer_format(anime)
    anime = remove_duplicates(anime)
    anime = remove_unsafe_ratings(anime)
    anime = remove_low_members_anime(anime)

    anime_released = get_released_anime(anime)
    anime_released = remove_unlabeled_anime(anime_released)

    anime_airing = get_airing_anime(anime)
    anime_unreleased = get_unreleased_anime(anime)

    create_parquet(anime_released, 'anime_released')
    create_parquet(anime_airing, 'anime_airing')
    create_parquet(anime_unreleased, 'anime_unreleased')


if __name__ == "__main__":
    run()
