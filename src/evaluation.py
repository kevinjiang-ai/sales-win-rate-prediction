"""
Model evaluation: precision, recall, AUC-ROC, calibration curve,
confusion matrix, and feature importance.
Saves all plots to reports/figures/.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)

FIGURES_DIR = Path(__file__).parent.parent / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_model(
    model,
    X_test,
    y_test,
    model_name: str,
    feature_names: list = None,
) -> dict:
    """
    Full evaluation suite. Returns a dict of metrics.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    roc_auc = roc_auc_score(y_test, y_prob)
    avg_precision = average_precision_score(y_test, y_prob)

    report = classification_report(y_test, y_pred, output_dict=True)
    precision = report["1"]["precision"]
    recall = report["1"]["recall"]
    f1 = report["1"]["f1-score"]

    metrics = {
        "model": model_name,
        "roc_auc": round(roc_auc, 4),
        "avg_precision": round(avg_precision, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }

    print(f"\n{'='*50}")
    print(f"  {model_name}")
    print(f"{'='*50}")
    print(f"  AUC-ROC:         {roc_auc:.4f}")
    print(f"  Avg Precision:   {avg_precision:.4f}")
    print(f"  Precision@0.5:   {precision:.4f}")
    print(f"  Recall@0.5:      {recall:.4f}")
    print(f"  F1@0.5:          {f1:.4f}")
    print()

    # --- ROC Curve ---
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax,
                                      name=model_name, color="steelblue")
    ax.set_title(f"ROC Curve — {model_name}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"roc_{_slug(model_name)}.png", dpi=150)
    plt.close(fig)

    # --- Precision-Recall Curve ---
    prec_arr, rec_arr, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(rec_arr, prec_arr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec_arr, prec_arr, color="darkorange", lw=2,
            label=f"PR AUC = {pr_auc:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve — {model_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"pr_{_slug(model_name)}.png", dpi=150)
    plt.close(fig)

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Lost", "Won"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {model_name}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"cm_{_slug(model_name)}.png", dpi=150)
    plt.close(fig)

    # --- Calibration Curve ---
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(prob_pred, prob_true, "s-", color="steelblue", label=model_name)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"Calibration Curve — {model_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"cal_{_slug(model_name)}.png", dpi=150)
    plt.close(fig)

    # --- Feature Importance (GB only) ---
    if hasattr(model, "feature_importances_") and feature_names:
        importances = model.feature_importances_
        top_n = 20
        indices = np.argsort(importances)[::-1][:top_n]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(
            [feature_names[i] for i in indices[::-1]],
            importances[indices[::-1]],
            color="steelblue",
        )
        ax.set_title(f"Top {top_n} Feature Importances — {model_name}")
        ax.set_xlabel("Importance")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"fi_{_slug(model_name)}.png", dpi=150)
        plt.close(fig)

    return metrics


def save_metrics(all_metrics: list, path: Path = None) -> None:
    if path is None:
        path = Path(__file__).parent.parent / "reports" / "metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetrics saved → {path}")


def _slug(name: str) -> str:
    return name.lower().replace(" ", "_")
