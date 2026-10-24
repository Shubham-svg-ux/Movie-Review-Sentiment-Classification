"""
Inference Module for Sentiment Classification.
Provides SentimentPredictor class for single-text and batch inference,
confidence scoring, and feature attribution (word importance explanation).
"""

import os
from typing import Dict, Any, List, Optional
import numpy as np
import joblib

import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import preprocess_text
from src.features import load_vectorizer

MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
)


class SentimentPredictor:
    """
    Production inference engine for movie review sentiment classification.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        vectorizer_path: Optional[str] = None,
    ):
        if model_path is None:
            model_path = os.path.join(MODELS_DIR, "best_model.joblib")
        if vectorizer_path is None:
            vectorizer_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model artifact not found at {model_path}. Run 'python src/train.py' first."
            )
        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(
                f"Vectorizer artifact not found at {vectorizer_path}. Run 'python src/train.py' first."
            )

        self.model = joblib.load(model_path)
        self.vectorizer = load_vectorizer(vectorizer_path)
        self.feature_names = np.array(self.vectorizer.get_feature_names_out())

        # Extract weights if available for feature explanation
        self.coefficients = self._extract_coefficients()

    def _extract_coefficients(self) -> Optional[np.ndarray]:
        """Extract linear weights for explainability if model is linear or calibrated."""
        try:
            if hasattr(self.model, "coef_"):
                return self.model.coef_[0]
            # If wrapped in CalibratedClassifierCV
            if hasattr(self.model, "calibrated_classifiers_"):
                coefs = []
                for base_clf in self.model.calibrated_classifiers_:
                    if hasattr(base_clf, "estimator") and hasattr(base_clf.estimator, "coef_"):
                        coefs.append(base_clf.estimator.coef_[0])
                if coefs:
                    return np.mean(coefs, axis=0)
            if hasattr(self.model, "feature_log_prob_"):
                # For Naive Bayes, log prob difference
                return self.model.feature_log_prob_[1] - self.model.feature_log_prob_[0]
        except Exception:
            pass
        return None

    def explain_text(self, clean_text: str, top_k: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """Identifies key words contributing towards positive and negative sentiment."""
        if self.coefficients is None:
            return {"positive_indicators": [], "negative_indicators": []}

        tfidf_vec = self.vectorizer.transform([clean_text])
        non_zero_indices = tfidf_vec.nonzero()[1]

        if len(non_zero_indices) == 0:
            return {"positive_indicators": [], "negative_indicators": []}

        words = self.feature_names[non_zero_indices]
        weights = self.coefficients[non_zero_indices] * tfidf_vec.data

        word_weight_pairs = sorted(zip(words, weights), key=lambda x: x[1], reverse=True)

        pos_indicators = [
            {"word": w, "weight": round(float(wt), 4)}
            for w, wt in word_weight_pairs if wt > 0
        ][:top_k]

        neg_indicators = [
            {"word": w, "weight": round(float(wt), 4)}
            for w, wt in reversed(word_weight_pairs) if wt < 0
        ][:top_k]

        return {
            "positive_indicators": pos_indicators,
            "negative_indicators": neg_indicators,
        }

    def predict(self, raw_text: str) -> Dict[str, Any]:
        """
        Classifies sentiment of a single review text.

        Returns:
          Dictionary with sentiment ('Positive'/'Negative'), confidence,
          preprocessed text, and keyword explanation.
        """
        cleaned = preprocess_text(raw_text)
        if not cleaned:
            return {
                "sentiment": "Neutral",
                "label": -1,
                "confidence": 0.5,
                "probability_positive": 0.5,
                "probability_negative": 0.5,
                "cleaned_text": "",
                "explanation": {"positive_indicators": [], "negative_indicators": []},
            }

        X_tfidf = self.vectorizer.transform([cleaned])
        pred_label = int(self.model.predict(X_tfidf)[0])

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_tfidf)[0]
            prob_neg = float(probs[0])
            prob_pos = float(probs[1])
        else:
            prob_pos = float(pred_label)
            prob_neg = 1.0 - prob_pos

        confidence = prob_pos if pred_label == 1 else prob_neg
        sentiment = "Positive" if pred_label == 1 else "Negative"

        explanation = self.explain_text(cleaned)

        return {
            "sentiment": sentiment,
            "label": pred_label,
            "confidence": round(confidence, 4),
            "probability_positive": round(prob_pos, 4),
            "probability_negative": round(prob_neg, 4),
            "cleaned_text": cleaned,
            "explanation": explanation,
        }

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Classifies a batch of reviews efficiently."""
        cleaned_list = [preprocess_text(t) for t in texts]
        X_tfidf = self.vectorizer.transform(cleaned_list)
        preds = self.model.predict(X_tfidf)

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_tfidf)
        else:
            probs = [[1.0 - p, p] for p in preds]

        results = []
        for raw, clean, pred, prob in zip(texts, cleaned_list, preds, probs):
            p_pos = float(prob[1])
            p_neg = float(prob[0])
            label = int(pred)
            conf = p_pos if label == 1 else p_neg
            results.append(
                {
                    "raw_text": raw[:100] + ("..." if len(raw) > 100 else ""),
                    "sentiment": "Positive" if label == 1 else "Negative",
                    "label": label,
                    "confidence": round(conf, 4),
                    "probability_positive": round(p_pos, 4),
                    "probability_negative": round(p_neg, 4),
                }
            )
        return results


if __name__ == "__main__":
    predictor = SentimentPredictor()
    test_review = "An extraordinary cinematic masterpiece! Superb performances and unforgettable screenplay."
    res = predictor.predict(test_review)
    print("Test Review:", test_review)
    print("Sentiment:", res["sentiment"], "Confidence:", res["confidence"])
    print("Explanation:", res["explanation"])
