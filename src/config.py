from pathlib import Path
from datetime import date

RUN_DATE = date.today().strftime("%Y-%m-%d")

SINGLE_VALUE_FIELDS = ['mal_id', 'favorites', 'members', 'type', 'source', 'rating', 'status']
MULTI_VALUE_FIELDS = ['studios', 'producers', 'genres', 'themes', 'demographics']
TARGET_VARIABLE = 'score'

# Base data folder
DATA_DIR = "data"
MODELS_DIR = "models"
PREDICTIONS_DIR = "predictions"
POSTERS_DIR = 'posters'


def create_directories():
    root = Path(__file__).resolve().parent.parent
    for folder in ["data", "models", "posters", "predictions"]:
        (root / folder).mkdir(parents=True, exist_ok=True)