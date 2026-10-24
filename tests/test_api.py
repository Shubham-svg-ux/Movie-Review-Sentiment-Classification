"""
Integration tests for FastAPI REST API endpoints.
"""

import os
import sys
import unittest
from starlette.testclient import TestClient

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from api.main import app


class TestFastAPIEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_root_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "online")

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["model_loaded"])

    def test_predict_single_endpoint(self):
        payload = {"review": "An absolute masterpiece with terrific acting and brilliant visuals!"}
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["sentiment"], "Positive")
        self.assertGreater(data["confidence"], 0.5)
        self.assertIn("explanation", data)

    def test_predict_batch_endpoint(self):
        payload = {
            "reviews": [
                "Brilliant film, loved every moment of it!",
                "Terrible and boring, completely unwatchable.",
            ]
        }
        response = self.client.post("/predict-batch", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["total_reviews"], 2)
        self.assertGreaterEqual(data["positive_count"], 1)

    def test_metrics_endpoint(self):
        response = self.client.get("/metrics")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("champion_model", data)
        self.assertIn("all_models", data)


if __name__ == "__main__":
    unittest.main()
