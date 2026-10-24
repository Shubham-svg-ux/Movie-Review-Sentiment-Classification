"""
Streamlit Web Application for Movie Review Sentiment Classification.
Features:
  - Real-time interactive sentiment classification with confidence score
  - Transparent NLP explainability (feature attribution & word weights)
  - Multi-model selector (Logistic Regression, LinearSVC, Naive Bayes, Random Forest)
  - Batch review analysis via CSV upload or multi-line text with export
  - Model comparison & evaluation benchmark dashboard
"""

import os
import sys
import json
import pandas as pd
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.predict import SentimentPredictor
from src.preprocessing import preprocess_text

# Page configuration
st.set_page_config(
    page_title="Movie Sentiment AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #6c757d;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #1f77b4;
    }
    .badge-pos {
        background-color: #d4edda;
        color: #155724;
        font-weight: 700;
        padding: 6px 14px;
        border-radius: 20px;
        display: inline-block;
        font-size: 1.1rem;
    }
    .badge-neg {
        background-color: #f8d7da;
        color: #721c24;
        font-weight: 700;
        padding: 6px 14px;
        border-radius: 20px;
        display: inline-block;
        font-size: 1.1rem;
    }
    .word-pos {
        background-color: #e2f0d9;
        color: #276a3c;
        padding: 3px 8px;
        margin: 2px;
        border-radius: 4px;
        font-size: 0.9rem;
        display: inline-block;
    }
    .word-neg {
        background-color: #fce4d6;
        color: #c65911;
        padding: 3px 8px;
        margin: 2px;
        border-radius: 4px;
        font-size: 0.9rem;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_predictor(model_slug: str = "best_model"):
    """Loads and caches sentiment predictor for selected model."""
    model_file = f"{model_slug}.joblib"
    model_path = os.path.join(MODELS_DIR, model_file)
    vectorizer_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
    return SentimentPredictor(model_path=model_path, vectorizer_path=vectorizer_path)


@st.cache_data
def load_benchmark_data():
    """Loads benchmark comparison CSV and metadata."""
    csv_path = os.path.join(REPORTS_DIR, "model_comparison.csv")
    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")

    df = pd.read_csv(csv_path) if os.path.exists(csv_path) else None
    meta = None
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    return df, meta


# Sidebar
with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/clapperboard.png",
        width=70,
    )
    st.markdown("### 🎬 **Model Settings**")

    model_options = {
        "Champion (Logistic Regression)": "best_model",
        "Linear SVC (Calibrated)": "linear_svc_calibrated",
        "Multinomial Naive Bayes": "multinomial_naive_bayes",
        "Complement Naive Bayes": "complement_naive_bayes",
        "Random Forest": "random_forest",
    }

    selected_label = st.selectbox(
        "Active Classifier Architecture:",
        options=list(model_options.keys()),
        index=0,
    )
    selected_slug = model_options[selected_label]

    st.markdown("---")
    st.markdown("### 📌 **Project Details**")
    st.markdown(
        """
        - **Pipeline**: N-Gram TF-IDF + Machine Learning
        - **Vocabulary**: 10,000 Unigrams & Bigrams
        - **Preprocessing**: HTML Clean, Contraction Expand, Negation-Preserving Tokenizer
        - **Dataset**: IMDb Benchmark
        """
    )
    st.markdown("---")
    st.markdown("Developed with ❤️ using Python, Scikit-Learn & Streamlit")

# Main Header
st.markdown("<div class='main-header'>🎬 Movie Review Sentiment Classification</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-header'>High-accuracy binary sentiment classifier leveraging TF-IDF feature engineering and comparative machine learning architectures.</div>",
    unsafe_allow_html=True,
)

# Load selected model predictor
try:
    predictor = load_predictor(selected_slug)
except Exception as e:
    st.error(f"Error loading model: {e}. Please ensure training has been executed (`python src/train.py`).")
    st.stop()

# Tabs
tab1, tab2, tab3 = st.tabs(
    [
        "🔍 Single Review Analyzer",
        "📁 Batch Classification",
        "📊 Model Comparison & Benchmarks",
    ]
)

# ----------------- TAB 1: Single Review Analyzer -----------------
with tab1:
    st.markdown("#### Test Movie Review Sentiment")

    # Sample presets
    col_pre1, col_pre2, col_pre3 = st.columns(3)
    preset_text = ""
    if col_pre1.button("✨ Sample: Acclaimed Masterpiece"):
        preset_text = "An absolute cinematic masterpiece! The emotional depth, direction, and brilliant performances make this an unforgettable experience."
    if col_pre2.button("👎 Sample: Boring & Dreadful"):
        preset_text = "A complete disaster. Terrible pacing, predictable plot twists, dreadful acting, and utterly boring from start to finish."
    if col_pre3.button("⚖️ Sample: Mixed Review with Negation"):
        preset_text = "The cinematography was stunning and the soundtrack was not bad at all, though the screenplay could have been tighter."

    review_input = st.text_area(
        "Enter or paste movie review text below:",
        value=preset_text,
        height=140,
        placeholder="e.g. This film exceeded all my expectations! Wonderful acting and breathtaking visuals.",
    )

    col_btn, col_info = st.columns([1, 4])
    analyze_clicked = col_btn.button("🚀 Analyze Sentiment", type="primary")

    if analyze_clicked or review_input:
        if not review_input.strip():
            st.warning("Please enter a review to analyze.")
        else:
            with st.spinner("Analyzing text and computing sentiment features..."):
                res = predictor.predict(review_input)

            st.markdown("---")
            col_res1, col_res2 = st.columns([1, 1])

            with col_res1:
                st.markdown("##### Prediction Result")
                if res["sentiment"] == "Positive":
                    st.markdown(
                        f"<div class='badge-pos'>👍 POSITIVE SENTIMENT</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"<div class='badge-neg'>👎 NEGATIVE SENTIMENT</div>",
                        unsafe_allow_html=True,
                    )

                st.markdown(f"**Confidence Score:** `{res['confidence'] * 100:.2f}%`")
                st.progress(res["confidence"])

                col_prob1, col_prob2 = st.columns(2)
                col_prob1.metric("Probability Positive", f"{res['probability_positive'] * 100:.1f}%")
                col_prob2.metric("Probability Negative", f"{res['probability_negative'] * 100:.1f}%")

            with col_res2:
                st.markdown("##### 🧠 NLP Explainability (Key Drivers)")
                st.caption("Words and n-grams having the strongest impact on this classification:")

                expl = res["explanation"]
                if expl["positive_indicators"]:
                    st.markdown("**Positive Contributing Cues:**")
                    html_pos = "".join(
                        [
                            f"<span class='word-pos'>{item['word']} (+{item['weight']})</span>"
                            for item in expl["positive_indicators"]
                        ]
                    )
                    st.markdown(html_pos, unsafe_allow_html=True)

                if expl["negative_indicators"]:
                    st.markdown("**Negative Contributing Cues:**")
                    html_neg = "".join(
                        [
                            f"<span class='word-neg'>{item['word']} ({item['weight']})</span>"
                            for item in expl["negative_indicators"]
                        ]
                    )
                    st.markdown(html_neg, unsafe_allow_html=True)

                with st.expander("Show Cleaned & Tokenized Text"):
                    st.code(res["cleaned_text"])


# ----------------- TAB 2: Batch Classification -----------------
with tab2:
    st.markdown("#### Batch Review Analysis")
    st.write("Upload a CSV file containing movie reviews or enter multiple reviews (one per line).")

    batch_mode = st.radio("Input Method:", ["Enter Multiple Reviews", "Upload CSV File"], horizontal=True)

    batch_reviews = []
    if batch_mode == "Enter Multiple Reviews":
        sample_batch = (
            "One of the best movies of the decade! Loved the emotional depth.\n"
            "Utterly horrible. A waste of time and money.\n"
            "Charming characters and brilliant direction.\n"
            "Boring screenplay and awful performances.\n"
            "Not disappointing at all, highly recommended!"
        )
        batch_input = st.text_area("Paste reviews (one review per line):", value=sample_batch, height=180)
        if batch_input.strip():
            batch_reviews = [line.strip() for line in batch_input.split("\n") if line.strip()]

    else:
        uploaded_file = st.file_uploader("Upload CSV file (must have a 'review' column):", type=["csv"])
        if uploaded_file is not None:
            df_upload = pd.read_csv(uploaded_file)
            if "review" in df_upload.columns:
                batch_reviews = df_upload["review"].dropna().tolist()
                st.success(f"Loaded {len(batch_reviews)} reviews from CSV.")
            else:
                st.error("Uploaded CSV must have a column named 'review'.")

    if st.button("⚡ Run Batch Classification", type="primary"):
        if not batch_reviews:
            st.warning("No reviews found to classify.")
        else:
            with st.spinner(f"Classifying {len(batch_reviews)} reviews in batch..."):
                batch_res = predictor.predict_batch(batch_reviews)

            df_out = pd.DataFrame(batch_res)
            st.markdown("##### Batch Classification Results")

            # Summary Metrics
            pos_count = (df_out["sentiment"] == "Positive").sum()
            neg_count = (df_out["sentiment"] == "Negative").sum()
            total = len(df_out)

            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("Total Reviews", total)
            col_m2.metric("Positive Reviews", f"{pos_count} ({pos_count/total*100:.1f}%)")
            col_m3.metric("Negative Reviews", f"{neg_count} ({neg_count/total*100:.1f}%)")

            # Table
            st.dataframe(df_out[["raw_text", "sentiment", "confidence", "probability_positive", "probability_negative"]], use_container_width=True)

            # Download CSV
            csv_data = df_out.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Annotated Results as CSV",
                data=csv_data,
                file_name="classified_movie_reviews.csv",
                mime="text/csv",
            )


# ----------------- TAB 3: Model Comparison & Benchmarks -----------------
with tab3:
    st.markdown("#### Comprehensive ML Model Comparison")
    st.write(
        "Evaluation results across 5 distinct machine learning models trained on identical TF-IDF features "
        "and tested on a held-out test set of 1,000 balanced IMDb reviews."
    )

    df_models, meta = load_benchmark_data()
    if df_models is not None:
        st.dataframe(
            df_models.style.highlight_max(
                subset=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
                color="#d4edda",
            ),
            use_container_width=True,
        )

        st.markdown("---")
        st.markdown("##### 📈 Visual Performance Analysis")

        col_fig1, col_fig2 = st.columns(2)
        bar_path = os.path.join(FIGURES_DIR, "model_comparison_metrics.png")
        roc_path = os.path.join(FIGURES_DIR, "roc_curves.png")

        if os.path.exists(bar_path):
            col_fig1.image(bar_path, caption="Metric Comparison (Accuracy, F1, Precision, Recall)")
        if os.path.exists(roc_path):
            col_fig2.image(roc_path, caption="Receiver Operating Characteristic (ROC) Curves")

        cm_path = os.path.join(FIGURES_DIR, "confusion_matrices.png")
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Confusion Matrices Across All Models on Held-Out Test Set")
    else:
        st.info("Run `python src/train.py` and `python src/evaluate.py` to populate benchmark results.")
