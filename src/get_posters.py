from pathlib import Path
import logging
import pandas as pd
import requests
from src import config

PREDICTIONS_DIR = config.PREDICTIONS_DIR
POSTERS_DIR = config.POSTERS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

root = Path(__file__).resolve().parent.parent


def create_posters_directory():
    """ Create directory for saving posters if it doesn't exist. """
    (root / POSTERS_DIR / config.RUN_DATE).mkdir(parents=True, exist_ok=True)


def get_images_info():
    """ Retrieve image URLs and IDs from the predictions CSV files. """
    airing_path = root / PREDICTIONS_DIR / config.RUN_DATE / 'predictions_airing.csv'
    unreleased_path = root / PREDICTIONS_DIR / config.RUN_DATE / 'predictions_unreleased.csv'

    airing = pd.read_csv(airing_path)
    unreleased = pd.read_csv(unreleased_path)

    info = pd.concat([airing[['mal_id', 'image_url']], unreleased[['mal_id', 'image_url']]], ignore_index=True)
    return info 


def download_posters(info: pd.DataFrame):
    """ Download posters from URLs and save them locally. """
    logger.info('Downloading anime posters...')
    for _, anime in info.iterrows():
        file_name = f'{anime["mal_id"]}.webp'
        url = anime['image_url']
        response = requests.get(url)
        if response.status_code == 200:
            file_path = root / POSTERS_DIR / config.RUN_DATE / file_name
            with open(file_path, 'wb') as f:
                f.write(response.content)
        else:
            logger.error(f'Error fetching {url}.')


def run():
    """ Main function to execute the poster downloading pipeline. """
    create_posters_directory()
    info = get_images_info()
    download_posters(info)


if __name__ == '__main__':
    run()


