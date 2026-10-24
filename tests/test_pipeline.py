"""
Unit tests for inference pipeline and predictor.
"""

import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.predict import SentimentPredictor


class TestSentimentPredictor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = SentimentPredictor()

    def test_positive_prediction(self):
        review = "A breathtaking masterpiece! Outstanding performances and phenomenal direction."
        result = self.predictor.predict(review)
        self.assertEqual(result["sentiment"], "Positive")
        self.assertEqual(result["label"], 1)
        self.assertGreater(result["confidence"], 0.5)
        self.assertGreater(result["probability_positive"], result["probability_negative"])
        self.assertIn("explanation", result)

    def test_negative_prediction(self):
        review = "Horrible waste of time. Completely awful, boring, and terrible acting throughout."
        result = self.predictor.predict(review)
        self.assertEqual(result["sentiment"], "Negative")
        self.assertEqual(result["label"], 0)
        self.assertGreater(result["confidence"], 0.5)
        self.assertGreater(result["probability_negative"], result["probability_positive"])

    def test_batch_prediction(self):
        reviews = [
            "Loved this brilliant movie, super entertaining!",
            "Utter disaster, painful to watch and poorly directed.",
        ]
        batch_res = self.predictor.predict_batch(reviews)
        self.assertEqual(len(batch_res), 2)
        self.assertEqual(batch_res[0]["sentiment"], "Positive")
        self.assertEqual(batch_res[1]["sentiment"], "Negative")

    def test_empty_string_handling(self):
        result = self.predictor.predict("")
        self.assertEqual(result["sentiment"], "Neutral")
        self.assertEqual(result["confidence"], 0.5)


if __name__ == "__main__":
    unittest.main()
