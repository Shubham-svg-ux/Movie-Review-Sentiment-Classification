# 🎬 Movie Review Sentiment Classification

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-v1.8.0-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end, production-ready Natural Language Processing (NLP) and Machine Learning system that classifies movie reviews into **Positive** or **Negative** sentiment with over **87% accuracy** and **0.948 ROC-AUC**. 

Built with **TF-IDF n-gram feature extraction**, a **negation-preserving preprocessing pipeline**, and **multi-model benchmarking** across 5 distinct machine learning architectures. Features an interactive **Streamlit web application** and a high-performance **FastAPI REST API**.

---

## 📌 Project Highlights & Resume Summary

- **Binary Sentiment Classification:** Categorizes natural language movie reviews into positive or negative polarity with calibrated confidence probabilities.
- **Robust Preprocessing & Feature Engineering:** Implemented HTML stripping, contraction expansion, tokenization, lemmatization, and negation-aware stopword removal (retaining critical polarity words like *not*, *no*, *never*). Extracted 10,000 sublinear TF-IDF unigram and bigram features.
- **Multi-Model Benchmark:** Trained, tuned, and evaluated 5 ML classifiers (Logistic Regression, Calibrated LinearSVC, Multinomial Naive Bayes, Complement Naive Bayes, and Random Forest) to identify the optimal classification approach.
- **Explainable AI (XAI):** Built feature-attribution transparency displaying the exact words and weights driving positive or negative classifications.
- **Dual Deployment Interfaces:** Includes a Streamlit web app with live sentiment gauge and batch CSV processing, plus a FastAPI backend with Swagger docs.

---

## 📊 Model Comparison & Benchmark Results

All models were evaluated on a held-out test set of 1,000 balanced reviews using identical TF-IDF features:

| Model Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Train Time (s) | Inference Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🏆 **Logistic Regression** | **0.8680** | **0.8525** | **0.8900** | **0.8708** | **0.9482** | 0.117s | < 0.01 ms |
| 🥈 **Linear SVC (Calibrated)** | 0.8680 | 0.8580 | 0.8820 | 0.8698 | 0.9475 | 0.212s | 0.006 ms |
| 🥉 **Multinomial Naive Bayes** | 0.8700 | 0.8715 | 0.8680 | 0.8697 | 0.9448 | **0.013s** | **0.001 ms** |
| 🔹 **Complement Naive Bayes** | 0.8700 | 0.8715 | 0.8680 | 0.8697 | 0.9448 | **0.012s** | **0.001 ms** |
| 🌲 **Random Forest** | 0.8200 | 0.8113 | 0.8340 | 0.8225 | 0.9080 | 2.252s | 0.065 ms |

> **Champion Model:** **Logistic Regression** achieved the highest overall balance of F1-Score (0.8708) and ROC-AUC (0.9482) while delivering sub-millisecond inference latency.

---

## 📈 Visual Performance Reports

All evaluation figures are generated automatically by `src/evaluate.py`:

- **Metric Comparison Bar Chart:** `reports/figures/model_comparison_metrics.png`
- **Confusion Matrices:** `reports/figures/confusion_matrices.png`
- **Receiver Operating Characteristic (ROC) Curves:** `reports/figures/roc_curves.png`

---

## 🏗️ System Architecture

```
Raw Review Text (HTML / Slang / Contractions)
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│              NLP Preprocessing Pipeline                 │
│  - Strip HTML tags (<br />, <p>) & Web URLs             │
│  - Expand contractions (didn't -> did not, can't)       │
│  - Tokenize into alphabetic words                       │
│  - Stopword filtering (Negation-aware: keep not/never)   │
│  - Lemmatization & normalization                        │
└─────────────────────────────┬───────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────┐
│              TF-IDF Feature Extraction                  │
│  - N-Gram range: (1, 2) unigrams & bigrams              │
│  - 10,000 max features, min_df=3, max_df=0.9           │
│  - Sublinear TF scaling: 1 + log(tf)                    │
└─────────────────────────────┬───────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────┐
│             Machine Learning Classifiers                │
│  - Logistic Regression (L2 regularization) [Champion]   │
│  - LinearSVC with CalibratedClassifierCV probabilities  │
│  - Multinomial & Complement Naive Bayes                 │
│  - Random Forest Ensemble                               │
└─────────────────────────────┬───────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
    ┌──────────────────┐            ┌──────────────────┐
    │  Streamlit UI    │            │   FastAPI Server │
    │  Interactive App │            │   REST Endpoints │
    └──────────────────┘            └──────────────────┘
```

---

## 📁 Repository Structure

```
Movie-Review-Sentiment-Classification/
├── .github/
│   └── workflows/
│       └── ci.yml               # Automated GitHub Actions test pipeline
├── .gitignore                   # Standard Python gitignore
├── README.md                    # Project documentation & results
├── requirements.txt             # Project dependencies
├── run_app.bat                  # One-click launcher for Streamlit UI
├── run_api.bat                  # One-click launcher for FastAPI REST server
├── data/
│   └── movie_reviews.csv        # Balanced 5,000 IMDb benchmark dataset
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # Dataset loader, remote fetcher & split
│   ├── preprocessing.py         # Text cleaning, tokenization & stopwords
│   ├── features.py              # TF-IDF vectorizer configuration
│   ├── train.py                 # Multi-model training and benchmarking
│   ├── evaluate.py              # Publication-grade chart generation
│   └── predict.py               # Production inference & explainability engine
├── models/
│   ├── best_model.joblib        # Champion model artifact
│   ├── tfidf_vectorizer.joblib  # Fitted TF-IDF Vectorizer
│   ├── model_metadata.json      # Benchmark metadata and metrics
│   └── *.joblib                 # Individual model artifacts
├── reports/
│   ├── model_comparison.csv     # Numerical benchmark scores
│   └── figures/
│       ├── model_comparison_metrics.png
│       ├── confusion_matrices.png
│       └── roc_curves.png
├── app/
│   └── streamlit_app.py         # Interactive multi-tab web application
├── api/
│   ├── __init__.py
│   └── main.py                  # FastAPI REST API with Swagger docs
├── notebooks/
│   └── Movie_Review_Sentiment_Classification.ipynb
└── tests/
    ├── test_preprocessing.py    # Unit tests for text cleaning
    ├── test_pipeline.py         # Unit tests for inference pipeline
    └── test_api.py              # Integration tests for REST API
```

---

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/Shubham-svg-ux/Movie-Review-Sentiment-Classification.git
cd Movie-Review-Sentiment-Classification
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Pre-trained Inference Directly
```bash
python src/predict.py
```

### 4. Retrain Models (Optional)
```bash
python src/train.py
python src/evaluate.py
```

---

## 🖥️ Running the Applications

### 🌐 Streamlit Interactive Web Application
Launch the interactive web UI:
```bash
streamlit run app/streamlit_app.py
```
Or double-click `run_app.bat` on Windows.
- **Single Review Tab:** Type any custom review, view real-time confidence scores and feature word attributions.
- **Batch Processing Tab:** Upload CSV files and download annotated sentiment predictions.
- **Benchmark Dashboard Tab:** Inspect model metrics, ROC curves, and confusion matrices.

### ⚡ FastAPI REST API Server
Start the high-performance REST API:
```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```
Or double-click `run_api.bat` on Windows.
- Interactive Swagger UI documentation: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Alternative ReDoc documentation: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

#### Sample API Request (cURL)
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"review": "An extraordinary cinematic masterpiece! Superb performances and unforgettable screenplay."}'
```

#### Sample API Response
```json
{
  "sentiment": "Positive",
  "label": 1,
  "confidence": 0.7964,
  "probability_positive": 0.7964,
  "probability_negative": 0.2036,
  "cleaned_text": "extraordinary cinematic masterpiece superb performances unforgettable screenplay",
  "explanation": {
    "positive_indicators": [
      {"word": "superb", "weight": 0.4993},
      {"word": "performance", "weight": 0.3798},
      {"word": "masterpiece", "weight": 0.2844}
    ],
    "negative_indicators": []
  }
}
```

---

## 🧪 Running Automated Tests

Run the complete test suite (15 unit & integration tests):
```bash
python -m unittest discover tests -v
```

---

## 📜 License
Distributed under the MIT License. See `LICENSE` for more information.

## 👤 Author
**Shubham Yadav**
- GitHub: [@Shubham-svg-ux](https://github.com/Shubham-svg-ux)
