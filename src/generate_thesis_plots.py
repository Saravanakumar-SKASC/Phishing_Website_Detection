
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json, joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report,
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix,
)
from src.config_loader import load_config
from src.data_loader import load_dataset
from src.preprocessor import preprocess_data

PLOTS_DIR = Path("artifacts/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="tab10")

config = load_config()
art = config["artifacts"]
df = load_dataset(config["data"]["file_path"])
feature_names = [c for c in df.columns if c != config["data"]["target_column"]]

X_train, X_test, y_train, y_test = preprocess_data(
    df=df, target_col=config["data"]["target_column"],
    test_size=config["data"]["test_size"], random_state=config["data"]["random_state"],
    scaler_path=f"{art['directory']}/{art['scaler_filename']}"
)
y_arr = np.array(y_test)

# ── Load all models ──────────────────────────────────────────────────────────
xgb    = joblib.load(f"{art['directory']}/{art['xgb_model_filename']}")
ann    = joblib.load(f"{art['directory']}/{art['ann_model_filename']}")
tabnet = joblib.load(f"{art['directory']}/{art['tabnet_model_prefix']}.pkl")
rf     = joblib.load(f"{art['directory']}/{art['rf_model_filename']}")
lgbm   = joblib.load(f"{art['directory']}/{art['lgbm_model_filename']}")
stack_bundle = joblib.load(f"{art['directory']}/{art['stack_model_filename']}")
stack_meta   = stack_bundle["meta"]
stack_test   = stack_bundle["test_preds"]

# Compute predictions
models_info = {
    "XGBoost":  (xgb.predict(X_test),    xgb.predict_proba(X_test)[:, 1]),
    "ANN MLP":  (ann.predict(X_test),    ann.predict_proba(X_test)[:, 1]),
    "TabNet":   (tabnet.predict(X_test), tabnet.predict_proba(X_test)[:, 1]),
    "Rand. Forest": (rf.predict(X_test), rf.predict_proba(X_test)[:, 1]),
    "LightGBM": (lgbm.predict(X_test),  lgbm.predict_proba(X_test)[:, 1]),
    "Stacking": (stack_meta.predict(stack_test), stack_meta.predict_proba(stack_test)[:, 1]),
}

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


# ── 1. Master metrics table ──────────────────────────────────────────────────
rows = []
for name, (preds, proba) in models_info.items():
    r = classification_report(y_arr, preds, output_dict=True)
    rows.append({
        "Model":      name,
        "Accuracy":   round(r["accuracy"], 4),
        "ROC-AUC":    round(auc(*roc_curve(y_arr, proba)[:2]), 4),
        "F1 (Phish)": round(r["1"]["f1-score"], 4),
        "Precision":  round(r["1"]["precision"], 4),
        "Recall":     round(r["1"]["recall"], 4),
    })
metrics_df = pd.DataFrame(rows)
metrics_df.to_csv(PLOTS_DIR / "all_models_metrics.csv", index=False)
print("Metrics table:")
print(metrics_df.to_string(index=False))
print()


# ── 2. ROC curves — all 6 models ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
for (name, (preds, proba)), color in zip(models_info.items(), COLORS):
    fpr, tpr, _ = roc_curve(y_arr, proba)
    roc_auc = auc(fpr, tpr)
    lw = 2.5 if name == "Stacking" else 1.5
    ls = "-" if name == "Stacking" else "--"
    ax.plot(fpr, tpr, color=color, lw=lw, ls=ls, label=f"{name} (AUC={roc_auc:.4f})")
ax.plot([0,1],[0,1],"k:",lw=0.8)
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curves — All Models", fontsize=13, fontweight="bold")
ax.legend(fontsize=9, loc="lower right")
fig.tight_layout()
fig.savefig(PLOTS_DIR / "thesis_roc_all_models.png", dpi=180)
plt.close(fig)
print(f"Saved: {PLOTS_DIR / 'thesis_roc_all_models.png'}")


# ── 3. Grouped bar chart — Accuracy + AUC + F1 ───────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
metric_cols = ["Accuracy", "ROC-AUC", "F1 (Phish)"]
titles      = ["Accuracy", "ROC-AUC", "F1-Score (Phishing class)"]
for ax, col, title in zip(axes, metric_cols, titles):
    bars = ax.bar(metrics_df["Model"], metrics_df[col], color=COLORS, edgecolor="white")
    ax.set_ylim(0.92, 1.002)
    ax.set_title(title, fontweight="bold")
    ax.set_xticklabels(metrics_df["Model"], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    for bar, val in zip(bars, metrics_df[col]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
                f"{val:.4f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
fig.suptitle("Model Comparison — PhishGuard AI", fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(PLOTS_DIR / "thesis_model_comparison.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {PLOTS_DIR / 'thesis_model_comparison.png'}")


# ── 4. Precision-Recall curves — all models ──────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
for (name, (preds, proba)), color in zip(models_info.items(), COLORS):
    prec, rec, _ = precision_recall_curve(y_arr, proba)
    ap = average_precision_score(y_arr, proba)
    lw = 2.5 if name == "Stacking" else 1.5
    ls = "-" if name == "Stacking" else "--"
    ax.plot(rec, prec, color=color, lw=lw, ls=ls, label=f"{name} (AP={ap:.4f})")
ax.set_xlabel("Recall", fontsize=12)
ax.set_ylabel("Precision", fontsize=12)
ax.set_title("Precision-Recall Curves — All Models", fontsize=13, fontweight="bold")
ax.legend(fontsize=9, loc="lower left")
fig.tight_layout()
fig.savefig(PLOTS_DIR / "thesis_pr_all_models.png", dpi=180)
plt.close(fig)
print(f"Saved: {PLOTS_DIR / 'thesis_pr_all_models.png'}")


# ── 5. Confusion matrix grid (2×3) ───────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
axes = axes.flatten()
for ax, (name, (preds, _)), color in zip(axes, models_info.items(), COLORS):
    cm = confusion_matrix(y_arr, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Legit","Phish"], yticklabels=["Legit","Phish"],
                linewidths=0.5, linecolor="white")
    ax.set_title(name, fontweight="bold", color=color)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
fig.suptitle("Confusion Matrices — All Models", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(PLOTS_DIR / "thesis_confusion_grid.png", dpi=180)
plt.close(fig)
print(f"Saved: {PLOTS_DIR / 'thesis_confusion_grid.png'}")


# ── 6. TabNet attention mask visualization ───────────────────────────────────
try:
    # Get attention masks for a sample of test instances
    explain_matrix, masks = tabnet.explain(X_test[:200])
    # explain_matrix shape: (n_samples, n_features) — global attention weight per feature
    mean_attention = np.mean(explain_matrix, axis=0)
    att_df = pd.DataFrame({"Feature": feature_names, "Attention": mean_attention})
    att_df = att_df.sort_values("Attention", ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(9, 7))
    bars = ax.barh(att_df["Feature"][::-1], att_df["Attention"][::-1],
                   color="#2ca02c", edgecolor="white")
    ax.set_xlabel("Mean Attention Weight", fontsize=12)
    ax.set_title("TabNet — Feature Attention Weights\n(Which features TabNet focused on most)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "tabnet_attention_weights.png", dpi=180)
    plt.close(fig)
    print(f"Saved: {PLOTS_DIR / 'tabnet_attention_weights.png'}")

    # Per-step attention masks (shows the sequential reasoning)
    n_steps = len(masks)
    fig, axes = plt.subplots(1, n_steps, figsize=(4*n_steps, 5))
    for step_idx, (ax, mask) in enumerate(zip(axes, masks)):
        mean_mask = np.mean(mask, axis=0)
        top_n = 10
        top_idx = np.argsort(mean_mask)[-top_n:][::-1]
        ax.barh([feature_names[i] for i in top_idx[::-1]],
                [mean_mask[i] for i in top_idx[::-1]], color=COLORS[step_idx % len(COLORS)])
        ax.set_title(f"Step {step_idx+1}", fontweight="bold")
        ax.set_xlabel("Attention")
    fig.suptitle("TabNet Sequential Attention — Per Step\n(Interpretable reasoning chain)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "tabnet_step_attention.png", dpi=180)
    plt.close(fig)
    print(f"Saved: {PLOTS_DIR / 'tabnet_step_attention.png'}")
except Exception as e:
    print(f"TabNet attention plot skipped: {e}")


# ── 7. Stacking meta-learner coefficient plot ────────────────────────────────
base_names = list(stack_bundle["base_learners"].keys())
coefs      = stack_meta.coef_[0]
fig, ax = plt.subplots(figsize=(7, 4))
bar_colors = ["#d62728" if c > 0 else "#1f77b4" for c in coefs]
ax.bar(base_names, coefs, color=bar_colors, edgecolor="white")
ax.axhline(0, color="black", lw=0.8)
ax.set_ylabel("Meta-learner Coefficient (trust weight)", fontsize=11)
ax.set_title("Stacking Ensemble — How Much the Meta-Learner Trusts Each Base Model",
             fontsize=11, fontweight="bold")
ax.set_xticklabels(base_names, rotation=15)
for i, (x, v) in enumerate(zip(base_names, coefs)):
    ax.text(i, v + 0.05, f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
fig.tight_layout()
fig.savefig(PLOTS_DIR / "stacking_meta_coefficients.png", dpi=180)
plt.close(fig)
print(f"Saved: {PLOTS_DIR / 'stacking_meta_coefficients.png'}")


# ── 8. Save updated metrics JSON ─────────────────────────────────────────────
summary = {}
for name, (preds, proba) in models_info.items():
    r = classification_report(y_arr, preds, output_dict=True)
    summary[name] = {
        "accuracy":  round(r["accuracy"], 4),
        "roc_auc":   round(auc(*roc_curve(y_arr, proba)[:2]), 4),
        "f1_phish":  round(r["1"]["f1-score"], 4),
        "precision": round(r["1"]["precision"], 4),
        "recall":    round(r["1"]["recall"], 4),
    }
with open(PLOTS_DIR / "all_models_metrics.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"Saved: {PLOTS_DIR / 'all_models_metrics.json'}")

print("\n All thesis plots generated in artifacts/plots/")
