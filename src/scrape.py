from pathlib import Path
from datetime import date
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
import requests
from src.config import (SINGLE_VALUE_FIELDS, MULTI_VALUE_FIELDS, TARGET_VARIABLE, DATA_DIR)


RUN_DATE = date.today().strftime("%Y-%m-%d")

FROM_YEAR = 2000
TO_YEAR = date.today().year + 2

COLUMNS = ['title', *SINGLE_VALUE_FIELDS, *MULTI_VALUE_FIELDS, 'image_url', 'datetime', 'trailer', TARGET_VARIABLE]


def is_jikan_online(timeout: int=5) -> bool:
    """
    Check if the Jikan API is online by sending a GET request to its base URL.
    Args:
        timeout (int): The timeout duration in seconds for the request.

    Returns:
        bool: True if the API is online (status code 200), False otherwise.
    """
    url = 'https://api.jikan.moe/v4'
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            print('Jikan API is online.')
            return True
        else:
            print(f'Jikan API responded with status code {response.status_code}')
            return False
    except requests.exceptions.RequestException as e:
        print(f'Failed to reach Jikan API: {e}')
        return False


def get_api_response(url: str) -> requests.Response | None:
    """ Send a GET request to the specified URL and handle rate limiting."""
    while True:
        response = requests.get(url)
        if response.status_code == 200:
            return response
        elif response.status_code == 429:
            # print('Rate limit exceeded. Waiting for 1 second...')
            time.sleep(1)
        else:
            print(f'Error {response.status_code}: {response.text}')
            return None
    

def scrape_anime(from_year: int, to_year: int) -> list[dict]:
    """
    Scrape anime data from Jikan API for the specified year range.
    Args:
        from_year (int): The starting year for scraping (inclusive).
        to_year (int): The ending year for scraping (inclusive).

    Returns:
        list[dict]: A list of dictionaries containing scraped anime data.
    """
    years = [year for year in range(from_year, to_year + 1)]
    scraped_anime = []
    with tqdm(total=to_year-from_year+1, desc='Scraping anime', dynamic_ncols=True) as pbar:
        for year in years:
            pbar.set_postfix(year=year)
            for season in ['winter', 'spring', 'summer', 'fall']:
                page = 1
                while True:
                    url = f'https://api.jikan.moe/v4/seasons/{year}/{season}?page={page}'
                    response = get_api_response(url)
                    if response is None:
                        return []

                    anime_json = response.json()
                    anime_data = anime_json.get('data', [])
                    if not anime_data:
                        break

                    for anime in anime_data:
                        anime_dict = dict.fromkeys(COLUMNS, np.nan)

                        for variable in [*SINGLE_VALUE_FIELDS, TARGET_VARIABLE]:
                            anime_dict[variable] = anime.get(variable, np.nan)

                        for variable in MULTI_VALUE_FIELDS:
                            anime_dict[variable] = [x['name'] for x in anime.get(variable, [])]

                        title_english = anime.get('title_english')
                        anime_dict['title'] = title_english if title_english else anime.get('title')

                        anime_dict['datetime'] = anime.get('aired').get('from')

                        anime_dict['trailer'] = anime.get('trailer').get('embed_url')

                        anime_dict['image_url'] = anime.get('images').get('webp').get('large_image_url')

                        scraped_anime.append(anime_dict)

                    if not anime_json.get('pagination', {}).get('has_next_page'):
                        break
                    page += 1
                    time.sleep(0.6)
            pbar.update(1)

    return scraped_anime


def create_directory() -> None:
    """
    Create the data directory for storing scraped data if it doesn't exist.
    Returns:
        None
    """
    root = Path(__file__).resolve().parent.parent
    (root / DATA_DIR).mkdir(exist_ok=True)
    (root / DATA_DIR / RUN_DATE).mkdir(parents=True, exist_ok=True)


def save_data(anime: pd.DataFrame) -> None:
    """
    Save the scraped anime DataFrame as a Parquet file.
    Args:
        anime (pd.DataFrame): The DataFrame containing scraped anime data.
    Returns:
        None
    """
    root = Path(__file__).resolve().parent.parent
    file_path = root / DATA_DIR / RUN_DATE / 'anime_raw.parquet'
    anime.to_parquet(file_path, engine="pyarrow", compression='snappy')


def run() -> None:
    """
    Main function to run the scraping process and save data.
    Returns:
        None
    """
    data = scrape_anime(FROM_YEAR, TO_YEAR) if is_jikan_online() else []
    anime = pd.DataFrame(data, columns=COLUMNS)

    if not anime.empty:
        create_directory()
        save_data(anime)
    else:
        print('No data scraped because Jikan API is offline.')
    

if __name__ == '__main__':
    run()
