RUN_DATE = None

SINGLE_VALUE_FIELDS = ['mal_id', 'favorites', 'members', 'type', 'source', 'rating', 'status']
MULTI_VALUE_FIELDS = ['studios', 'producers', 'genres', 'themes', 'demographics']
TARGET_VARIABLE = 'score'

dtypes = {
    "mal_id": "int32",
    "title": "string",
    "favorites": "int32",
    "members": "int32",
    "type": "category",
    "source": "category",
    "rating": "category",
    "status": "category",
    "studios": "object",
    "producers": "object",
    "genres": "object",
    "themes": "object",
    "demographics": "object",
    "image_url": "string",
    "datetime": "object",
    "trailer": "bool",
    "score": "float32"
}


# Base data folder
DATA_DIR = "data"
MODELS_DIR = "models"
PREDICTIONS_DIR = "predictions"
POSTERS_DIR = 'posters'
