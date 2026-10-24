"""
Text Preprocessing Pipeline for Movie Review Sentiment Classification.
Includes HTML cleaning, URL removal, contraction expansion, tokenization,
negation-aware stopword removal, and optional lemmatization.
"""

import re
from typing import List, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# English contractions map to preserve semantic intent
CONTRACTIONS = {
    "won't": "will not",
    "can't": "cannot",
    "n't": " not",
    "'re": " are",
    "'s": " is",
    "'d": " would",
    "'ll": " will",
    "'t": " not",
    "'ve": " have",
    "'m": " am",
}

# Negation words that must NOT be removed as stopwords
NEGATION_WORDS = {
    "not", "no", "never", "nor", "neither", "none", "nobody", "nowhere",
    "cannot", "hardly", "scarcely", "barely"
}

# Base stopwords excluding negations
STOP_WORDS = set(ENGLISH_STOP_WORDS) - NEGATION_WORDS

# Try loading NLTK lemmatizer with fallback
try:
    import nltk
    from nltk.stem import WordNetLemmatizer
    try:
        _lemmatizer = WordNetLemmatizer()
        _lemmatizer.lemmatize("testing")
    except Exception:
        _lemmatizer = None
except ImportError:
    _lemmatizer = None


def expand_contractions(text: str) -> str:
    """Expand common English contractions."""
    for contraction, expansion in CONTRACTIONS.items():
        text = re.sub(re.escape(contraction), expansion, text, flags=re.IGNORECASE)
    return text


def clean_html(text: str) -> str:
    """Strip HTML tags and entities."""
    clean_re = re.compile(r"<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")
    return clean_re.sub(" ", text)


def clean_urls(text: str) -> str:
    """Remove web URLs and mail links."""
    url_pattern = re.compile(r"https?://\S+|www\.\S+|ftp://\S+")
    return url_pattern.sub(" ", text)


def tokenize(text: str) -> List[str]:
    """Tokenize text into alphabetic tokens of length >= 2."""
    return re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())


def remove_stopwords(tokens: List[str]) -> List[str]:
    """Remove stopwords while strictly retaining negations."""
    return [token for token in tokens if token not in STOP_WORDS]


def lemmatize_tokens(tokens: List[str]) -> List[str]:
    """Lemmatize tokens if lemmatizer is available; otherwise return as-is."""
    if _lemmatizer:
        try:
            return [_lemmatizer.lemmatize(token) for token in tokens]
        except Exception:
            return tokens
    return tokens


def preprocess_text(
    text: str,
    remove_stops: bool = True,
    lemmatize: bool = True,
) -> str:
    """
    Complete end-to-end NLP preprocessing pipeline for a single raw review string.

    Pipeline:
      1. Lowercase text
      2. Strip HTML tags and URLs
      3. Expand contractions (e.g. didn't -> did not)
      4. Tokenize into words
      5. Filter stopwords (preserving critical negation cues)
      6. Lemmatize tokens
      7. Reconstruct cleaned text string
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. Clean HTML & URLs
    text = clean_html(text)
    text = clean_urls(text)

    # 2. Contraction expansion
    text = expand_contractions(text)

    # 3. Tokenization
    tokens = tokenize(text)

    # 4. Stopword filtering
    if remove_stops:
        tokens = remove_stopwords(tokens)

    # 5. Lemmatization
    if lemmatize:
        tokens = lemmatize_tokens(tokens)

    return " ".join(tokens)


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer for text preprocessing.
    """

    def __init__(self, remove_stops: bool = True, lemmatize: bool = True):
        self.remove_stops = remove_stops
        self.lemmatize = lemmatize

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [
            preprocess_text(
                doc, remove_stops=self.remove_stops, lemmatize=self.lemmatize
            )
            for doc in X
        ]


if __name__ == "__main__":
    sample = (
        "<b>A wonderful little production!</b> <br />I didn't expect it to be this good. "
        "Check http://imdb.com for details. Truly didn't disappoint!"
    )
    print("Original:\n", sample)
    print("Preprocessed:\n", preprocess_text(sample))
