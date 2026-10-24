"""
Feature Extraction Module using TF-IDF (Term Frequency - Inverse Document Frequency).
Extracts unigrams and bigrams with sublinear TF scaling.
"""

import os
from typing import Tuple, Optional
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import spmatrix

DEFAULT_VECTORIZER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "tfidf_vectorizer.joblib",
)


def create_vectorizer(
    ngram_range: Tuple[int, int] = (1, 2),
    max_features: int = 10000,
    min_df: int = 3,
    max_df: float = 0.9,
    sublinear_tf: bool = True,
) -> TfidfVectorizer:
    """
    Constructs an optimized TF-IDF Vectorizer with unigrams and bigrams.
    """
    return TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
    )


def fit_and_save_vectorizer(
    texts,
    save_path: str = DEFAULT_VECTORIZER_PATH,
    **kwargs,
) -> Tuple[TfidfVectorizer, spmatrix]:
    """
    Fits TF-IDF vectorizer on input texts and saves the fitted artifact to disk.
    """
    vectorizer = create_vectorizer(**kwargs)
    X_tfidf = vectorizer.fit_transform(texts)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(vectorizer, save_path)
    print(f"TF-IDF Vectorizer fitted and saved to {save_path} (Vocabulary size: {len(vectorizer.vocabulary_)})")

    return vectorizer, X_tfidf


def load_vectorizer(vectorizer_path: str = DEFAULT_VECTORIZER_PATH) -> TfidfVectorizer:
    """Loads saved TF-IDF vectorizer artifact."""
    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(f"TF-IDF Vectorizer not found at {vectorizer_path}. Run training first.")
    return joblib.load(vectorizer_path)


def transform_texts(texts, vectorizer: Optional[TfidfVectorizer] = None) -> spmatrix:
    """Transforms raw or preprocessed texts into TF-IDF feature matrix."""
    if vectorizer is None:
        vectorizer = load_vectorizer()
    return vectorizer.transform(texts)


if __name__ == "__main__":
    sample_texts = [
        "brilliant acting superb plot masterpiece",
        "terrible boring awful waste time",
        "great movie loved the direction",
    ]
    vec, X = fit_and_save_vectorizer(sample_texts, max_features=100, min_df=1)
    print("Feature matrix shape:", X.shape)
