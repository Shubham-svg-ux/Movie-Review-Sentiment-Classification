"""
Evaluation and Visualization Module.
Generates publication-quality charts:
  1. Model comparison metric bars (Accuracy, Precision, Recall, F1-Score)
  2. Multi-model Confusion Matrix heatmaps
  3. Comparative ROC Curves
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix

REPORTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports"
)
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")


def generate_visualizations():
    """Generates and saves all evaluation plots."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

    # Load comparison CSV
    csv_path = os.path.join(REPORTS_DIR, "model_comparison.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing {csv_path}. Please run src/train.py first.")

    df_models = pd.read_csv(csv_path)

    # 1. Model Comparison Bar Chart
    plt.figure(figsize=(11, 6), dpi=300)
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    df_melt = pd.melt(
        df_models, id_vars=["Model"], value_vars=metrics, var_name="Metric", value_name="Score"
    )

    ax = sns.barplot(
        data=df_melt,
        x="Model",
        y="Score",
        hue="Metric",
        palette="viridis",
    )
    plt.title("Comparative Performance Across Machine Learning Models", fontsize=14, weight="bold", pad=15)
    plt.ylabel("Score (0.0 - 1.0)", fontsize=12)
    plt.xlabel("Model Architecture", fontsize=12)
    plt.ylim(0.70, 1.0)
    plt.legend(loc="lower right", frameon=True)
    plt.xticks(rotation=15, ha="right")

    # Add data labels
    for p in ax.patches:
        height = p.get_height()
        if height > 0.05:
            ax.annotate(
                f"{height:.2f}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=8,
                xytext=(0, 2),
                textcoords="offset points",
            )

    bar_path = os.path.join(FIGURES_DIR, "model_comparison_metrics.png")
    plt.savefig(bar_path, bbox_inches="tight")
    plt.close()
    print(f"Saved metric comparison chart to {bar_path}")

    # Load evaluation cache for ROC & Confusion Matrix
    cache_path = os.path.join(REPORTS_DIR, "eval_cache.json")
    if not os.path.exists(cache_path):
        return

    with open(cache_path, "r", encoding="utf-8") as f:
        cache = json.load(f)

    y_test = np.array(cache["y_test"])
    models_data = cache["models"]

    # 2. Confusion Matrices Plot (Grid of heatmaps)
    num_models = len(models_data)
    fig, axes = plt.subplots(1, num_models, figsize=(4 * num_models, 3.6), dpi=300)
    if num_models == 1:
        axes = [axes]

    classes = ["Negative", "Positive"]
    for i, (name, data) in enumerate(models_data.items()):
        cm = confusion_matrix(y_test, data["y_pred"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=axes[i],
            xticklabels=classes,
            yticklabels=classes if i == 0 else False,
        )
        axes[i].set_title(name, fontsize=11, weight="bold")
        axes[i].set_xlabel("Predicted Label")
        if i == 0:
            axes[i].set_ylabel("True Label")

    fig.suptitle("Confusion Matrices on Held-Out Test Set (1,000 Reviews)", fontsize=14, weight="bold", y=1.05)
    cm_path = os.path.join(FIGURES_DIR, "confusion_matrices.png")
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrices to {cm_path}")

    # 3. Multi-Model ROC Curves
    plt.figure(figsize=(8, 6), dpi=300)
    for name, data in models_data.items():
        y_prob = np.array(data["y_prob"])
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_score = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_score:.4f})")

    plt.plot([0, 1], [0, 1], color="grey", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.50)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=12)
    plt.title("Receiver Operating Characteristic (ROC) Curves", fontsize=14, weight="bold", pad=15)
    plt.legend(loc="lower right", frameon=True)

    roc_path = os.path.join(FIGURES_DIR, "roc_curves.png")
    plt.savefig(roc_path, bbox_inches="tight")
    plt.close()
    print(f"Saved ROC curves to {roc_path}")


if __name__ == "__main__":
    generate_visualizations()
