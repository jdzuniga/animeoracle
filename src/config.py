RUN_DATE = None
# date.today().strftime("%Y-%m-%d"))

SINGLE_VALUE_FIELDS = ['mal_id', 'favorites', 'members', 'type', 'source', 'rating', 'status']
MULTI_VALUE_FIELDS = ['studios', 'producers', 'genres', 'themes', 'demographics']
TARGET_VARIABLE = 'score'

# Base data folder
DATA_DIR = "data"
MODELS_DIR = "models"
PREDICTIONS_DIR = "predictions"
POSTERS_DIR = 'posters'


