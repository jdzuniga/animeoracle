from datetime import date

RUN_DATE = date.today()
RUN_DATE = date.fromisoformat('2025-08-31')

SINGLE_VALUE_FIELDS = ['mal_id', 'favorites', 'members', 'type', 'source', 'rating', 'trailer', 'status']
MULTI_VALUE_FIELDS = ['studios', 'producers', 'genres', 'themes', 'demographics']
TARGET_VARIABLE = 'score'

# Base data folder
DATA_DIR = "./data"
MODELS_DIR = "./models"
PREDICTIONS_DIR = "./predictions"