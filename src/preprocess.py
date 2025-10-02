import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """
    Groups rare categories in a categorical feature into a single 'Rare_Group' category.
    Categories that appear in less than `min_pct` proportion of the data are considered rare.
    """
    def __init__(self, min_pct=0.01):
        """
        Args:
            min_pct (float): Minimum proportion threshold to retain a category. Default is 0.01 (1%).
        """
        self.min_pct = min_pct
        self.top_categories_ = None

    def fit(self, X, y=None):
        """
        Fit the transformer to identify top categories.
        Args:
            X (pd.Series): The input categorical feature.
            y (pd.Series, optional): The target variable (not used).
        """
        X = X.copy()
        counts = X.value_counts(normalize=True)
        self.top_categories_ = counts[counts >= self.min_pct].index.tolist()
        return self

    def transform(self, X):
        """
        Transform the input feature by grouping rare categories.
        Args:
            X (pd.Series): The input categorical feature.
        Returns:
            pd.DataFrame: The transformed feature with rare categories grouped.
        """
        X = X.copy()
        X = X.apply(lambda x: x if x in self.top_categories_ else 'Rare_Group')
        return pd.DataFrame(X)


def clean_text(text):
    """
    Clean and preprocess text data by removing special characters, converting to lowercase,
    and normalizing whitespace.
    Args:
        text (str): The input text string.
    Returns:
        str: The cleaned text string.
    """
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
    """
    Clean and preprocess text data in a pandas Series.
    """
    def fit(self, X, y=None):
        """
            Fit method (no operation).
            Args:
                X (pd.Series): The input text data.
                y (pd.Series, optional): The target variable (not used).
            Returns:
                self
        """
        return self

    def transform(self, X):
        """
        Transform the input text data by applying cleaning.
        Args:
            X (pd.Series): The input text data.
        Returns:
            pd.Series: The cleaned text data.
        """
        return X.apply(clean_text)


class Tfidf(BaseEstimator, TransformerMixin):
    """
    Convert a collection of raw documents to a matrix of TF-IDF features.
    """
    def __init__(self, max_features=100):
        """
        Args:
            max_features (int): Maximum number of features to extract. Default is 100.
        """
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(max_features=self.max_features, ngram_range=(1, 2), stop_words='english')

    def fit(self, X, y=None):
        """
        Fit the TF-IDF vectorizer to the input text data.
        Args:
            X (pd.Series): The input text data.
            y (pd.Series, optional): The target variable (not used).    
        Returns:
            self
        """
        self.vectorizer.fit(X)
        return self

    def transform(self, X):
        """
        Transform the input text data to TF-IDF features.
        Args:
            X (pd.Series): The input text data.
        Returns:
            sparse matrix: The TF-IDF feature matrix.
        """
        return self.vectorizer.transform(X)


class MultiLabelBinarizerTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to apply MultiLabelBinarizer with rare category grouping.
    Categories that appear in less than `min_pct` proportion of the data are grouped into 'Rare_Group'.
    """
    def __init__(self, min_pct=0.01):
        self.min_pct = min_pct
        self.mlb_ = None
        self.classes_ = None
        self.top_labels_ = None

    def fit(self, X, y=None):
        """
        Fit the transformer to identify top labels.
        Args:
            X (pd.Series): The input multi-label feature (list of labels).
            y (pd.Series, optional): The target variable (not used).
        Returns:
            self
        """
        all_labels = pd.Series([label for sublist in X for label in sublist])
        freqs = all_labels.value_counts(normalize=True)
        self.top_labels_ = freqs[freqs >= self.min_pct].index.tolist()

        self.mlb_ = MultiLabelBinarizer()

        if freqs[freqs <= self.min_pct].shape[0] > 0:
            self.mlb_.fit([self.top_labels_ + ["Rare_Group"]])
            self.top_labels_.append("Rare_Group")
        else:
            self.mlb_.fit([self.top_labels_])
        return self

    def transform(self, X):
        """
        Transform the input multi-label feature to a binary matrix with rare labels grouped.
        Args:
            X (pd.Series): The input multi-label feature (list of labels).
        Returns:
            sparse matrix: The binary feature matrix.
        """
        X_grouped = []
        for sublist in X:
            new_labels = [cat if cat in self.top_labels_ else "Rare_Group" for cat in sublist]
            X_grouped.append(list(set(new_labels)))

        return self.mlb_.transform(X_grouped)

    def get_feature_names_out(self, input_features=None):
        """
        Get output feature names.
        Args:
            input_features (list, optional): Input feature names (not used).
        Returns:
            np.array: The output feature names.
        """
        return np.array(self.top_labels_)


class MultiLabelImputer(BaseEstimator, TransformerMixin):
    """
    Impute empty lists in multi-label features with ['Unknown'].
    """
    def fit(self, X, y=None):
        """
         Fit method (no operation).
         Args:
            X (pd.Series): The input multi-label feature.
            y (pd.Series, optional): The target variable (not used).
         Returns:
            self
        """
        return self

    def transform(self, X):
        """
        Transform the input multi-label feature by imputing empty lists.
        Args:
            X (pd.Series): The input multi-label feature.
        Returns:
            pd.Series: The transformed multi-label feature with empty lists imputed.
        """
        X = X.copy()
        X = X.apply(lambda x: np.array(['Unknown']) if isinstance(x, np.ndarray) and len(x) == 0 else x)
        return X
    
    def get_feature_names_out(self, input_features=None):
        """
        Get output feature names.
        Args:
            input_features (list, optional): Input feature names (not used).
        Returns:
            np.array: The output feature names.
        """
        return np.array(input_features) if input_features is not None else np.array(["multilabel"])


    
class AddIsSequel(BaseEstimator, TransformerMixin):
    """
    Add a binary feature indicating whether the title suggests it is a sequel, remake, or part of a series.
    """
    def fit(self, X, y=None):
        """
        Fit method (no operation).
        Args:
            X (pd.Series): The input title data.
            y (pd.Series, optional): The target variable (not used).
        Returns:
            self
        """
        return self
    
    def transform(self, X, y=None):
        """
        Transform the input title data to add the 'is_sequel' feature.
        Args:
            X (pd.Series): The input title data.
            y (pd.Series, optional): The target variable (not used).
        Returns:
            pd.DataFrame: The transformed feature with 'is_sequel' column.
        """
        X = X.copy()
        X = X.apply(lambda title: 1 if (
            bool(re.search(r'sequel|next|return|part|chapter|movie', title, flags=re.IGNORECASE)) |
            bool(re.search(r'season \d+|s\d+', title, flags=re.IGNORECASE)) |
            bool(re.search(r'remake|reboot|^Re:', title, flags=re.IGNORECASE))
        ) else 0)
        return pd.DataFrame(X)
    
    def get_feature_names_out(self, input_features=None):
        """
        Get output feature names.
        Args:
            input_features (list, optional): Input feature names (not used).
        Returns:
            np.array: The output feature names.
        """
        return np.array(['is_sequel'])
    

title_pipeline = Pipeline([
    ('clean_text', CleanText()),
    ('tfidf', Tfidf(max_features=10))
])

is_sequel_pipeline = Pipeline([
    ('sequel_feature', AddIsSequel())
])

single_label_pipeline = Pipeline([
    ('group_sparse', RareCategoryGrouper()),
    ('imputer', SimpleImputer(strategy='constant', fill_value='Rare_Group')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True)),
])

multi_label_pipeline = Pipeline([
    ('mlb_imputer', MultiLabelImputer()),
    ('mlb', MultiLabelBinarizerTransformer()),
])

preprocessor = ColumnTransformer(transformers=[
    ('process_title', title_pipeline, 'title'),
    ('sequel', is_sequel_pipeline, 'title'),
    ('type', single_label_pipeline, 'type'),
    ('source', single_label_pipeline, 'source'),
    ('rating', single_label_pipeline, 'rating'),
    ('passthrough', 'passthrough', ['trailer', 'year', 'month']),
    ('studios_label', multi_label_pipeline, 'studios'),
    ('producers_label', multi_label_pipeline, 'producers'),
    ('genres_label', multi_label_pipeline, 'genres'),
    ('themes_label', multi_label_pipeline, 'themes'),
    ('demographics_label', multi_label_pipeline, 'demographics')
    ],
    sparse_threshold=1.0)