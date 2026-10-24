"""
Model Training and Benchmarking Module for Movie Review Sentiment Classification.
Trains and compares 5 ML classifiers:
  1. Logistic Regression
  2. Calibrated Linear Support Vector Classifier (LinearSVC)
  3. Multinomial Naive Bayes
  4. Complement Naive Bayes
  5. Random Forest Classifier

Identifies the champion model and serializes all models, vectorizers, and metrics.
"""

import os
import time
import json
from typing import Dict, Any, Tuple
import joblib
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_dataset
from src.preprocessing import preprocess_text
from src.features import fit_and_save_vectorizer

MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
)
REPORTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports"
)


def get_candidate_models() -> Dict[str, Any]:
    """Returns dictionary of candidate ML classifiers for comparison."""
    return {
        "Logistic Regression": LogisticRegression(
            C=1.0, max_iter=1000, random_state=42
        ),
        "Linear SVC (Calibrated)": CalibratedClassifierCV(
            LinearSVC(C=1.0, max_iter=2000, random_state=42),
            cv=3,
        ),
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.5),
        "Complement Naive Bayes": ComplementNB(alpha=0.5),
        "Random Forest": RandomForestClassifier(
            n_estimators=120, max_depth=30, random_state=42, n_jobs=-1
        ),
    }


def evaluate_model(
    model,
    X_test_tfidf,
    y_test,
) -> Dict[str, Any]:
    """Computes comprehensive evaluation metrics on held-out test set."""
    start_time = time.time()
    y_pred = model.predict(X_test_tfidf)
    infer_time_ms = ((time.time() - start_time) / len(y_test)) * 1000

    # Probability estimation for ROC-AUC
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test_tfidf)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_test_tfidf)
        y_prob = 1 / (1 + np.exp(-scores))
    else:
        y_prob = y_pred

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="binary")
    rec = recall_score(y_test, y_pred, average="binary")
    f1 = f1_score(y_test, y_pred, average="binary")
    try:
        auc = roc_auc_score(y_test, y_prob)
    except Exception:
        auc = 0.5

    cm = confusion_matrix(y_test, y_pred).tolist()

    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(auc), 4),
        "inference_latency_ms": round(float(infer_time_ms), 4),
        "confusion_matrix": cm,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def train_and_compare_models() -> pd.DataFrame:
    """
    Complete training workflow:
      1. Loads movie reviews dataset
      2. Preprocesses text (HTML stripping, tokenization, stopword removal)
      3. Fits TF-IDF Vectorizer
      4. Trains 5 distinct ML models
      5. Evaluates and benchmarks each model
      6. Serializes best model, all models, and comparative report
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("=" * 70)
    print("STEP 1: Loading Dataset & Splitting...")
    print("=" * 70)
    X_train_raw, X_test_raw, y_train, y_test = load_dataset()
    print(f"Train samples: {len(X_train_raw)}, Test samples: {len(X_test_raw)}")

    print("\n" + "=" * 70)
    print("STEP 2: Text Preprocessing & Cleaning...")
    print("=" * 70)
    t0 = time.time()
    X_train_clean = [preprocess_text(doc) for doc in X_train_raw]
    X_test_clean = [preprocess_text(doc) for doc in X_test_raw]
    print(f"Preprocessing completed in {time.time() - t0:.2f} seconds.")

    print("\n" + "=" * 70)
    print("STEP 3: TF-IDF Feature Extraction (Unigrams + Bigrams)...")
    print("=" * 70)
    vec_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
    vectorizer, X_train_tfidf = fit_and_save_vectorizer(
        X_train_clean,
        save_path=vec_path,
        ngram_range=(1, 2),
        max_features=10000,
        min_df=3,
        sublinear_tf=True,
    )
    X_test_tfidf = vectorizer.transform(X_test_clean)
    print(f"Feature matrix: Train {X_train_tfidf.shape}, Test {X_test_tfidf.shape}")

    print("\n" + "=" * 70)
    print("STEP 4: Training & Comparing Multiple ML Classifiers...")
    print("=" * 70)

    models = get_candidate_models()
    results = []
    trained_models = {}

    for name, clf in models.items():
        print(f"\n--> Training: {name}...")
        start_train = time.time()
        clf.fit(X_train_tfidf, y_train)
        train_time = round(time.time() - start_train, 3)

        eval_res = evaluate_model(clf, X_test_tfidf, y_test)
        eval_res["model_name"] = name
        eval_res["train_time_sec"] = train_time
        results.append(eval_res)
        trained_models[name] = clf

        # Save individual model artifact
        slug = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        model_save_path = os.path.join(MODELS_DIR, f"{slug}.joblib")
        joblib.dump(clf, model_save_path)

        print(
            f"    [Result] Accuracy: {eval_res['accuracy']:.4f} | "
            f"F1: {eval_res['f1_score']:.4f} | "
            f"ROC-AUC: {eval_res['roc_auc']:.4f} | "
            f"Train Time: {train_time}s"
        )

    # Convert results to DataFrame
    df_results = pd.DataFrame(
        [
            {
                "Model": r["model_name"],
                "Accuracy": r["accuracy"],
                "Precision": r["precision"],
                "Recall": r["recall"],
                "F1-Score": r["f1_score"],
                "ROC-AUC": r["roc_auc"],
                "Train Time (s)": r["train_time_sec"],
                "Inference Latency (ms)": r["inference_latency_ms"],
            }
            for r in results
        ]
    ).sort_values(by="F1-Score", ascending=False).reset_index(drop=True)

    # Save benchmark table
    csv_path = os.path.join(REPORTS_DIR, "model_comparison.csv")
    df_results.to_csv(csv_path, index=False)
    print("\n" + "=" * 70)
    print("MODEL COMPARISON BENCHMARK:")
    print("=" * 70)
    print(df_results.to_string(index=False))

    # Identify and save Champion Model
    best_row = df_results.iloc[0]
    best_name = best_row["Model"]
    best_clf = trained_models[best_name]

    best_model_path = os.path.join(MODELS_DIR, "best_model.joblib")
    joblib.dump(best_clf, best_model_path)
    print(f"\n[CHAMPION MODEL]: {best_name} (F1-Score: {best_row['F1-Score']:.4f}, Accuracy: {best_row['Accuracy']:.4f})")
    print(f"Saved champion model to {best_model_path}")

    # Save metadata JSON
    metadata = {
        "champion_model": best_name,
        "metrics": {
            "accuracy": float(best_row["Accuracy"]),
            "f1_score": float(best_row["F1-Score"]),
            "precision": float(best_row["Precision"]),
            "recall": float(best_row["Recall"]),
            "roc_auc": float(best_row["ROC-AUC"]),
        },
        "all_models": df_results.to_dict(orient="records"),
        "vocab_size": len(vectorizer.vocabulary_),
        "classes": ["Negative", "Positive"],
        "dataset": "IMDb Movie Reviews (5,000 balanced benchmark)",
    }
    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Save test predictions for evaluation visualization
    eval_cache = {
        "y_test": y_test.tolist(),
        "models": {
            r["model_name"]: {
                "y_pred": [int(x) for x in r["y_pred"]],
                "y_prob": [float(x) for x in r["y_prob"]],
            }
            for r in results
        },
    }
    with open(os.path.join(REPORTS_DIR, "eval_cache.json"), "w", encoding="utf-8") as f:
        json.dump(eval_cache, f)

    return df_results


if __name__ == "__main__":
    train_and_compare_models()
