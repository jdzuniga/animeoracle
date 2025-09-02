import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

def clean_text(text):
    text = re.sub(r'[^a-z0-9\s]', '', text, flags=re.IGNORECASE)
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    # Remove leading and trailing spaces
    text = re.sub(r'^\s+|\s+$', '', text)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    # Remove any remaining non-alphanumeric characters except spaces
    text = re.sub(r'[^\w\s]', '', text)

    return text


class CleanText(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.apply(clean_text)

class Tfidf(BaseEstimator, TransformerMixin):
    def __init__(self, max_features=500):
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(max_features=self.max_features, ngram_range=(1, 2), stop_words='english')

    def fit(self, X, y=None):
        self.vectorizer.fit(X)
        return self

    def transform(self, X):
        return self.vectorizer.transform(X)
    
class MultiLabelBinarizerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mlb = MultiLabelBinarizer(sparse_output=True)
        self.classes_ = None

    def fit(self, X, y=None):
        self.mlb.fit(X)
        self.classes_ = self.mlb.classes_
        return self

    def transform(self, X):
        return self.mlb.transform(X)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.classes_)
    
class MultiLabelImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X  = X.apply(lambda x: np.array(['Unknown']) if isinstance(x, np.ndarray) and len(x) == 0 else x)
        return X
    
    def get_feature_names_out(self, input_features=None):
        return np.array(input_features) if input_features is not None else np.array(["multilabel"])


class DimensionalityReducer(BaseEstimator, TransformerMixin):

    def __init__(self, n_components=10):
        self.n_components = n_components
        self.pca = TruncatedSVD(n_components=self.n_components)

    def fit(self, X, y=None):
        self.pca.fit(X.toarray() if hasattr(X, "toarray") else X)
        return self

    def transform(self, X):
        return self.pca.transform(X.toarray() if hasattr(X, "toarray") else X)
    
    def get_feature_names_out(self, input_features=None):
        # If n_components is a fraction, PCA chooses actual n_components after fit
        n_out = self.pca.n_components_
        return np.array([f"pca{i}" for i in range(n_out)])
    
class AddIsSequel(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X = X.copy()
        X = X.apply(lambda title: 1 if (
            bool(re.search(r'sequel|next|return|part|chapter|movie', title, flags=re.IGNORECASE)) |
            bool(re.search(r'season \d+|s\d+', title, flags=re.IGNORECASE)) |
            bool(re.search(r'remake|reboot|^Re:', title, flags=re.IGNORECASE))
        ) else 0)
        return pd.DataFrame(X)
    
    def get_feature_names_out(self, input_features=None):
        return np.array(['is_sequel']) 
    
class AddYear(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X = X.copy()
        X = pd.to_datetime(X).dt.year
        return pd.DataFrame(X)
    
    def get_feature_names_out(self, input_features=None):
        return np.array(["year"])
    
    
class AddSeason(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X = X.copy()
        X = pd.to_datetime(X).dt.month
        seasons = X.apply(self.month_to_season)
        seasons_one_hot = pd.get_dummies(seasons)
        return seasons_one_hot
    
    def month_to_season(self, month: int) -> str:
        if month in [12, 1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Spring"
        elif month in [6, 7, 8]:
            return "Summer"
        elif month in [9, 10, 11]:
            return "Fall"
        else:
            return None
    
    def get_feature_names_out(self, input_features=None):
        return np.array(['Winter', 'Spring', 'Summer', 'Fall']) 
    


title_pipeline = Pipeline([
    ('clean_text', CleanText()),
    ('tfidf', Tfidf(max_features=10))
])

is_sequel_pipeline = Pipeline([
    ('sequel_feature', AddIsSequel())
])

year_pipeline = Pipeline([
    ('year', AddYear())
])

season_pipeline = Pipeline([
    ('season', AddSeason()),
])

single_label_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', min_frequency=0.05, sparse_output=True)),
])

multi_label_pipeline = Pipeline([
    ('mlb_imputer', MultiLabelImputer()),
    ('mlb', MultiLabelBinarizerTransformer()),
    ('dim_reducer', DimensionalityReducer(n_components=10))
])

preprocessor = ColumnTransformer(transformers=[
    # ('process_title', title_pipeline, 'title'),
    ('sequel', is_sequel_pipeline, 'title'),
    ('year', year_pipeline, 'datetime'),
    ('season', season_pipeline, 'datetime'),
    ('cat', single_label_pipeline, ['type', 'source', 'rating', 'trailer']),
    ('studios_label', multi_label_pipeline, 'studios'),
    ('producers_label', multi_label_pipeline, 'producers'),
    # ('genres_label', multi_label_pipeline, 'genres'),
    ('themes_label', multi_label_pipeline, 'themes'),
    # ('demographics_label', multi_label_pipeline, 'demographics')
    ],
    sparse_threshold=1.0)