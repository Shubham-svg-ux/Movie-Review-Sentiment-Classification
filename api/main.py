"""
FastAPI REST API for Movie Review Sentiment Classification.
Exposes endpoints for single inference, batch inference, health checks, and model metrics.
"""

import os
import sys
import json
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.predict import SentimentPredictor

app = FastAPI(
    title="Movie Review Sentiment Classification API",
    description="High-performance NLP sentiment classification service powered by Scikit-Learn TF-IDF.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

# Initialize global predictor
try:
    predictor = SentimentPredictor()
except Exception as e:
    predictor = None
    print(f"Warning: Predictor could not be loaded at startup: {e}")


class ReviewRequest(BaseModel):
    review: str = Field(
        ...,
        min_length=1,
        example="An extraordinary cinematic masterpiece! Superb performances and unforgettable screenplay.",
        description="The raw movie review text to analyze.",
    )


class BatchReviewRequest(BaseModel):
    reviews: List[str] = Field(
        ...,
        min_items=1,
        max_items=500,
        example=[
            "A wonderful production, highly recommended!",
            "Terrible and boring movie. Waste of time.",
        ],
        description="List of movie reviews to classify in batch.",
    )


class WordImpact(BaseModel):
    word: str
    weight: float


class Explanation(BaseModel):
    positive_indicators: List[WordImpact]
    negative_indicators: List[WordImpact]


class PredictionResponse(BaseModel):
    sentiment: str = Field(..., example="Positive")
    label: int = Field(..., example=1)
    confidence: float = Field(..., example=0.8524)
    probability_positive: float = Field(..., example=0.8524)
    probability_negative: float = Field(..., example=0.1476)
    cleaned_text: str
    explanation: Explanation


class BatchPredictionItem(BaseModel):
    raw_text: str
    sentiment: str
    label: int
    confidence: float
    probability_positive: float
    probability_negative: float


class BatchPredictionResponse(BaseModel):
    total_reviews: int
    positive_count: int
    negative_count: int
    predictions: List[BatchPredictionItem]


@app.get("/", tags=["General"])
def root():
    """Root endpoint returning API service status and docs URL."""
    return {
        "service": "Movie Review Sentiment Classification API",
        "version": "1.0.0",
        "status": "online",
        "documentation": "/docs",
    }


@app.get("/health", tags=["General"])
def health_check():
    """Health check endpoint confirming model availability."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor model is not loaded.")
    return {
        "status": "healthy",
        "model_loaded": True,
        "vocab_size": len(predictor.vectorizer.vocabulary_),
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_sentiment(payload: ReviewRequest):
    """
    Classifies a single movie review into Positive or Negative sentiment,
    returning probability scores and NLP word attribution.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model artifact unavailable. Train model first.")
    
    result = predictor.predict(payload.review)
    return result


@app.post("/predict-batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch_sentiment(payload: BatchReviewRequest):
    """
    Classifies a batch of movie reviews efficiently.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model artifact unavailable. Train model first.")

    batch_res = predictor.predict_batch(payload.reviews)
    pos_count = sum(1 for r in batch_res if r["sentiment"] == "Positive")
    neg_count = len(batch_res) - pos_count

    return {
        "total_reviews": len(batch_res),
        "positive_count": pos_count,
        "negative_count": neg_count,
        "predictions": batch_res,
    }


@app.get("/metrics", tags=["Benchmarks"])
def get_model_metrics():
    """
    Retrieves comparative benchmark metrics for all evaluated ML models.
    """
    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
    if not os.path.exists(meta_path):
        raise HTTPException(status_code=404, detail="Model metadata benchmark not found.")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return meta


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
