from pathlib import Path
import logging
import time
import numpy as np
import pandas as pd
import requests
from src.config import (SINGLE_VALUE_FIELDS, MULTI_VALUE_FIELDS, TARGET_VARIABLE, DATA_DIR, RUN_DATE)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SCRAPE_FROM_YEAR = 2020
SCRAPE_TO_YEAR = 2027

COLUMNS = ['title', *SINGLE_VALUE_FIELDS, *MULTI_VALUE_FIELDS, 'image_url', 'datetime', TARGET_VARIABLE]

def is_jikan_online(timeout: int=5) -> bool:
    url = 'https://api.jikan.moe/v4'
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            logger.info('Jikan API is online.')
            return True
        else:
            logger.info(f'Jikan API responded with status code {response.status_code}')
            return False
    except requests.exceptions.RequestException as e:
        logger.error(f'Failed to reach Jikan API: {e}')
        return False


def get_api_response(url: str) -> requests.Response | None:
    while True:
        response = requests.get(url)
        if response.status_code == 200:
            return response
        elif response.status_code == 429:
            logger.warning('Rate limit exceeded. Waiting for 1 second...')
            time.sleep(1)
        else:
            logger.error(f'Error {response.status_code}: {response.text}')
            return None
    

def scrape_anime(from_year: int, to_year: int) -> list[dict]:
    years = [year for year in range(from_year, to_year + 1)]
    logger.info('Scraping data...')
    scraped_anime = []
    time_start = time.time()
    for year in years:
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

                    title_english = anime.get('title_english')
                    anime_dict['title'] = title_english if title_english else anime.get('title')

                    for variable in [*SINGLE_VALUE_FIELDS, TARGET_VARIABLE]:
                        anime_dict[variable] = anime.get(variable, np.nan)

                    for variable in MULTI_VALUE_FIELDS:
                        anime_dict[variable] = [x['name'] for x in anime.get(variable, [])]

                    anime_dict['datetime'] = anime.get('aired', {}).get('from', np.nan)

                    anime_dict['image_url'] = anime.get('images').get('webp').get('large_image_url')

                    scraped_anime.append(anime_dict)

                if not anime_json.get('pagination', {}).get('has_next_page'):
                    break
                    
                page += 1
                time.sleep(0.6)

        logger.debug(f'Year {year} completed.')

    time_end = time.time()
    logger.info(f'{len(scraped_anime)} animes scraped in {time_end - time_start:.2f} seconds.')
    return scraped_anime


def create_directory() -> None:
    root = Path(__file__).resolve().parent.parent
    (root / DATA_DIR / RUN_DATE).mkdir(parents=True, exist_ok=True)


def create_parquet(anime: pd.DataFrame) -> None:
    root = Path(__file__).resolve().parent.parent
    file_path = root / DATA_DIR / RUN_DATE / 'anime_raw.parquet'
    anime.to_parquet(file_path, engine="pyarrow", compression='snappy')
    logger.info(f'Scraped data saved at {file_path}.')


def run() -> None:
    data = scrape_anime(SCRAPE_FROM_YEAR, SCRAPE_TO_YEAR) if is_jikan_online() else []
    anime = pd.DataFrame(data, columns=COLUMNS)

    if not anime.empty:
        create_directory()
        create_parquet(anime)
    else:
        logger.error('No data scraped because Jikan API is offline.')
    

if __name__ == '__main__':
    run()
