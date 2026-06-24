
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os, joblib, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from src.config_loader import load_config
from src.data_loader import load_dataset
from src.preprocessor import preprocess_data

PLOTS_DIR = Path("artifacts/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data & models ──────────────────────────────────────────────────────
config = load_config()
df = load_dataset(config["data"]["file_path"])
X_train, X_test, y_train, y_test = preprocess_data(
    df=df,
    target_col=config["data"]["target_column"],
    test_size=config["data"]["test_size"],
    random_state=config["data"]["random_state"],
    scaler_path=f"{config['artifacts']['directory']}/{config['artifacts']['scaler_filename']}"
)

xgb  = joblib.load(f"{config['artifacts']['directory']}/{config['artifacts']['xgb_model_filename']}")
ann  = joblib.load(f"{config['artifacts']['directory']}/{config['artifacts']['ann_model_filename']}")

xgb_preds  = xgb.predict(X_test)
xgb_proba  = xgb.predict_proba(X_test)[:, 1]
ann_preds  = ann.predict(X_test)
ann_proba  = ann.predict_proba(X_test)[:, 1]

FEATURE_NAMES = df.drop(config["data"]["target_column"], axis=1).columns.tolist()


# ── 1. Confusion Matrices ────────────────────────────────────────────────────
def plot_confusion(preds, title, path):
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Legit", "Phish"], yticklabels=["Legit", "Phish"], ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")

plot_confusion(xgb_preds, "XGBoost – Confusion Matrix", PLOTS_DIR / "cm_xgboost.png")
plot_confusion(ann_preds, "ANN MLP – Confusion Matrix",  PLOTS_DIR / "cm_ann.png")


# ── 2. ROC Curves (overlay) ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
for preds, proba, label, color in [
    (xgb_preds, xgb_proba, "XGBoost", "#1f77b4"),
    (ann_preds,  ann_proba,  "ANN MLP",  "#ff7f0e"),
]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, label=f"{label} (AUC = {roc_auc:.4f})")

ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve Comparison")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(PLOTS_DIR / "roc_curves.png", dpi=150)
plt.close(fig)
print(f"Saved: {PLOTS_DIR / 'roc_curves.png'}")


# ── 3. Precision-Recall Curves ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
for preds, proba, label, color in [
    (xgb_preds, xgb_proba, "XGBoost", "#1f77b4"),
    (ann_preds,  ann_proba,  "ANN MLP",  "#ff7f0e"),
]:
    prec, rec, _ = precision_recall_curve(y_test, proba)
    ap = average_precision_score(y_test, proba)
    ax.plot(rec, prec, color=color, label=f"{label} (AP = {ap:.4f})")

ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve Comparison")
ax.legend(loc="lower left")
fig.tight_layout()
fig.savefig(PLOTS_DIR / "pr_curves.png", dpi=150)
plt.close(fig)
print(f"Saved: {PLOTS_DIR / 'pr_curves.png'}")


# ── 4. XGBoost Feature Importances ──────────────────────────────────────────
importances = xgb.feature_importances_
fi_df = pd.DataFrame({"Feature": FEATURE_NAMES, "Importance": importances})
fi_df = fi_df.sort_values("Importance", ascending=False).head(20)

fig, ax = plt.subplots(figsize=(8, 7))
sns.barplot(data=fi_df, x="Importance", y="Feature", palette="viridis", ax=ax)
ax.set_title("XGBoost – Top 20 Feature Importances")
fig.tight_layout()
fig.savefig(PLOTS_DIR / "xgb_feature_importance.png", dpi=150)
plt.close(fig)
print(f"Saved: {PLOTS_DIR / 'xgb_feature_importance.png'}")


# ── 5. Per-class Metrics Side-by-Side ────────────────────────────────────────
def get_metrics(preds, label):
    r = classification_report(y_test, preds, output_dict=True)
    return {
        "Model": label,
        "Precision (Legit)": r["0"]["precision"],
        "Recall (Legit)":    r["0"]["recall"],
        "F1 (Legit)":        r["0"]["f1-score"],
        "Precision (Phish)": r["1"]["precision"],
        "Recall (Phish)":    r["1"]["recall"],
        "F1 (Phish)":        r["1"]["f1-score"],
        "Accuracy":          r["accuracy"],
    }

metrics = pd.DataFrame([get_metrics(xgb_preds, "XGBoost"), get_metrics(ann_preds, "ANN MLP")])
metrics = metrics.set_index("Model").T

fig, ax = plt.subplots(figsize=(10, 5))
metrics.plot(kind="bar", ax=ax, colormap="tab10", edgecolor="white")
ax.set_ylim(0.88, 1.01)
ax.set_title("Model Performance Comparison – Per-class Metrics")
ax.set_ylabel("Score"); ax.set_xticklabels(metrics.index, rotation=30, ha="right")
ax.legend(loc="lower right")
ax.axhline(0.97, color="grey", linestyle="--", linewidth=0.7, alpha=0.6)
fig.tight_layout()
fig.savefig(PLOTS_DIR / "metrics_comparison.png", dpi=150)
plt.close(fig)
print(f"Saved: {PLOTS_DIR / 'metrics_comparison.png'}")


# ── 6. SHAP Summary (XGBoost) ────────────────────────────────────────────────
try:
    import shap
    explainer = shap.TreeExplainer(xgb)
    X_test_df = pd.DataFrame(X_test, columns=FEATURE_NAMES)
    shap_values = explainer.shap_values(X_test_df)
    fig, ax = plt.subplots(figsize=(9, 7))
    shap.summary_plot(shap_values, X_test_df, show=False, plot_type="dot", max_display=20)
    plt.title("SHAP Summary – XGBoost")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'shap_summary.png'}")
except Exception as e:
    print(f"SHAP plot skipped: {e}")


# ── 7. Save metrics JSON ─────────────────────────────────────────────────────
summary = {
    "XGBoost": {
        "accuracy": float(classification_report(y_test, xgb_preds, output_dict=True)["accuracy"]),
        "roc_auc":  float(auc(*roc_curve(y_test, xgb_proba)[:2])),
        "f1_phish": float(classification_report(y_test, xgb_preds, output_dict=True)["1"]["f1-score"]),
    },
    "ANN_MLP": {
        "accuracy": float(classification_report(y_test, ann_preds, output_dict=True)["accuracy"]),
        "roc_auc":  float(auc(*roc_curve(y_test, ann_proba)[:2])),
        "f1_phish": float(classification_report(y_test, ann_preds, output_dict=True)["1"]["f1-score"]),
    }
}
with open(PLOTS_DIR / "metrics_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"Saved: {PLOTS_DIR / 'metrics_summary.json'}")

print("\n All evaluation artifacts saved to artifacts/plots/")
