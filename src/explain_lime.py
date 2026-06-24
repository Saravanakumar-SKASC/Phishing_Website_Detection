
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lime
import lime.lime_tabular
from src.utils import print_header

PLOTS_DIR = Path("artifacts/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def explain_with_lime(model, X_train, X_test, y_test, feature_names,
                      n_samples=5, n_features_display=15):
    """
    Generate LIME explanations for:
      - n_samples correctly predicted phishing URLs
      - n_samples correctly predicted legitimate URLs
    Saves individual explanation plots + a side-by-side summary.
    """
    print_header("LIME: Local Explanations (XAI)")

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=["Legitimate", "Phishing"],
        mode="classification",
        discretize_continuous=False,
        random_state=42,
    )

    preds = model.predict(X_test)
    y_arr = np.array(y_test)

    # Find correctly predicted phishing and legitimate samples
    correct_phish = np.where((preds == 1) & (y_arr == 1))[0]
    correct_legit = np.where((preds == 0) & (y_arr == 0))[0]

    def explain_sample(idx, label):
        exp = explainer.explain_instance(
            data_row=X_test[idx],
            predict_fn=model.predict_proba,
            num_features=n_features_display,
            num_samples=500,
        )
        fig = exp.as_pyplot_figure()
        fig.suptitle(f"LIME — {label} (sample #{idx})", fontsize=11, y=1.01)
        fig.tight_layout()
        safe_label = label.replace(" ", "_").replace("#", "")
        out_path = PLOTS_DIR / f"lime_{safe_label}_{idx}.png"
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")
        return exp

    print(f"\nExplaining {n_samples} phishing samples...")
    phish_exps = [explain_sample(idx, f"Phishing") for idx in correct_phish[:n_samples]]

    print(f"\nExplaining {n_samples} legitimate samples...")
    legit_exps  = [explain_sample(idx, f"Legitimate") for idx in correct_legit[:n_samples]]

    # Aggregate feature weights across all LIME explanations
    _aggregate_lime_weights(phish_exps + legit_exps, feature_names)

    return phish_exps, legit_exps


def _aggregate_lime_weights(exps, feature_names):
    """Average LIME weights across all explained samples → global view."""
    weight_sums = {f: 0.0 for f in feature_names}
    count = {f: 0 for f in feature_names}

    for exp in exps:
        for feat_desc, weight in exp.as_list():
            # feat_desc is something like "SSLfinal_State <= 0.50"
            # extract the feature name prefix
            matched = next((f for f in feature_names if feat_desc.startswith(f)), None)
            if matched:
                weight_sums[matched] += abs(weight)
                count[matched] += 1

    avg_weights = {f: (weight_sums[f] / count[f] if count[f] > 0 else 0) for f in feature_names}
    sorted_feats = sorted(avg_weights.items(), key=lambda x: x[1], reverse=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    feats, weights = zip(*sorted_feats[:20])
    ax.barh(feats[::-1], weights[::-1], color="#2ca02c")
    ax.set_xlabel("Mean |LIME Weight| across all explained samples")
    ax.set_title("LIME Aggregated Feature Importance (Global View)")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "lime_aggregate_importance.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {PLOTS_DIR / 'lime_aggregate_importance.png'}")
