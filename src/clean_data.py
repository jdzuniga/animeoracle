import logging
from pathlib import Path
import pandas as pd
from src import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def transform_trailer_format(data: pd.DataFrame) -> pd.DataFrame:
    """ Transform the 'trailer' column to a boolean indicating presence of a trailer.
    Args:
        data (pd.DataFrame): DataFrame containing the 'trailer' column.

    Returns:
        pd.DataFrame: The transformed DataFrame with the 'trailer' column updated.
    """
    data['trailer'] = data['trailer'].notna()
    return data


def remove_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    """ Remove duplicate entries from the anime DataFrame.
    Args:
        data (pd.DataFrame): DataFrame containing anime data.

    Returns:
        pd.DataFrame: The DataFrame with duplicates removed.
    """
    duplicated_mask = data.duplicated(subset=['title'], keep='first') | data.index.duplicated(keep='first')
    data = data[~duplicated_mask]
    logger.info(f"{duplicated_mask.sum()} duplicated rows dropped.\n")
    return data


def remove_unsafe_ratings(data: pd.DataFrame) -> pd.DataFrame:
    """
    Remove entries with unsafe ratings from the anime DataFrame.
    Args:
        data (pd.DataFrame): DataFrame containing anime data.
    Returns:
        pd.DataFrame: The DataFrame with unsafe ratings removed.
    """
    unsafe_ratings = data['rating'] == "Rx - Hentai"
    data = data[~unsafe_ratings]
    logger.info(f"{unsafe_ratings.sum()} unsafe ratings dropped.\n")
    return data


def remove_low_members_anime(data: pd.DataFrame, quantile: float=0.25) -> pd.DataFrame:
    """
    Remove anime entries with a number of members below a specified quantile threshold.
    Args:
        data (pd.DataFrame): DataFrame containing anime data.
        quantile (float): Quantile threshold to filter members. Default is 0.25 (25th percentile).
    Returns:
        pd.DataFrame: The DataFrame with low-members anime removed.
    """
    members_threshold = data['members'].quantile(quantile)
    logger.info(f"{data[data['members'] <= members_threshold].shape[0]} anime removed below {members_threshold} members.")
    data = data[data['members'] > members_threshold]
    return data


def remove_unlabeled_anime(data: pd.DataFrame) -> pd.DataFrame:
    """
    Remove anime entries with missing target variable labels.
    Args:
        data (pd.DataFrame): DataFrame containing anime data.
    Returns:
        pd.DataFrame: The DataFrame with unlabeled anime removed.
    """
    logger.info(f'{data["score"].isnull().sum()} missing labels removed from sample.')
    return data.dropna(subset=[config.TARGET_VARIABLE])


def extract_year_month(data: pd.DataFrame) -> pd.DataFrame:
    """
    Extract year and month from the 'datetime' column and drop the original column.
    Args:
        data (pd.DataFrame): DataFrame containing the 'datetime' column.
    Returns:
        pd.DataFrame: The DataFrame with 'year' and 'month' columns added and 'datetime' column removed.
    """
    data['year'] = pd.to_datetime(data['datetime']).dt.year
    data['month'] = pd.to_datetime(data['datetime']).dt.month
    return data.drop('datetime', axis=1)


def load_data() -> pd.DataFrame:
    """
    Load the raw anime data from a Parquet file.
    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    root = Path(__file__).resolve().parent.parent
    file_path = root / config.DATA_DIR / config.RUN_DATE / f'anime_raw.parquet'
    anime = pd.read_parquet(file_path)
    return anime


def create_parquet(data: pd.DataFrame, file_name: str) -> None:
    """
    Save the cleaned DataFrame to a Parquet file.
    Args:
        data (pd.DataFrame): The cleaned DataFrame to save.
        file_name (str): The name of the output Parquet file (without extension).
    Returns:
        None
    """
    root = Path(__file__).resolve().parent.parent
    file_path = root / config.DATA_DIR / config.RUN_DATE / f'{file_name}_cleaned.parquet'
    data.to_parquet(file_path, engine="pyarrow", compression='snappy')
    logger.info(f'Cleaned data saved at {file_path}.')


def run() -> None:
    """ Main function to execute the data cleaning pipeline. """
    data = load_data()
    data.set_index('mal_id', inplace=True)
    data = transform_trailer_format(data)
    data = remove_duplicates(data)
    data = remove_unsafe_ratings(data)
    data = remove_low_members_anime(data)
    data = extract_year_month(data)

    anime_released = data[data['status'] == 'Finished Airing']
    anime_airing = data[data['status'] == 'Currently Airing']
    anime_unreleased = data[data['status'] == 'Not yet aired']

    anime_released = remove_unlabeled_anime(anime_released)

    create_parquet(anime_released, 'anime_released')
    create_parquet(anime_airing, 'anime_airing')
    create_parquet(anime_unreleased, 'anime_unreleased')


if __name__ == "__main__":
    run()
