

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from src.utils import print_header

PLOTS_DIR = Path("artifacts/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _flip_value(v):
    """Flip a ternary feature: -1→1, 0→1, 1→0 (attacker makes it look legitimate)."""
    if v == 1:
        return 0     # was suspicious → flip to legitimate
    else:
        return 1     # was unknown/legitimate → flip to suspicious (less likely attack direction)


def single_feature_attack(model, X_test, y_test, feature_names, scaler=None):
    """
    For each feature, flip its value on ALL phishing samples and measure accuracy drop.
    Returns a DataFrame sorted by accuracy drop (most exploitable first).
    """
    print_header("ADVERSARIAL: Single-Feature Flip Attack")

    # Work in original (unscaled) space — we need integer feature values
    # If scaler provided, inverse-transform first
    if scaler is not None:
        X_orig = scaler.inverse_transform(X_test)
        X_orig = np.round(X_orig).astype(int)
        X_orig = np.clip(X_orig, -1, 1)
    else:
        X_orig = X_test.copy().astype(int)

    y_test_arr = np.array(y_test)
    phish_mask = y_test_arr == 1   # only attack phishing samples

    baseline_preds = model.predict(X_test)
    baseline_acc   = accuracy_score(y_test_arr, baseline_preds)
    print(f"Baseline accuracy: {baseline_acc:.4f}")

    results = []
    for i, feat in enumerate(feature_names):
        X_adv_orig = X_orig.copy()
        # Only flip phishing samples (attacker wants to evade detection)
        X_adv_orig[phish_mask, i] = np.where(
            X_adv_orig[phish_mask, i] == 1, 0, X_adv_orig[phish_mask, i]
        )
        # Re-scale
        if scaler is not None:
            X_adv = scaler.transform(X_adv_orig.astype(float))
        else:
            X_adv = X_adv_orig.astype(float)

        adv_preds = model.predict(X_adv)
        adv_acc   = accuracy_score(y_test_arr, adv_preds)
        acc_drop  = baseline_acc - adv_acc

        results.append({
            "Feature":       feat,
            "Baseline_Acc":  round(baseline_acc, 4),
            "Adv_Acc":       round(adv_acc, 4),
            "Acc_Drop":      round(acc_drop, 4),
        })

    df = pd.DataFrame(results).sort_values("Acc_Drop", ascending=False)
    df.to_csv(PLOTS_DIR / "adversarial_single_flip.csv", index=False)
    print(f"\nTop 10 most exploitable features:")
    print(df.head(10).to_string(index=False))
    print(f"\nSaved: {PLOTS_DIR / 'adversarial_single_flip.csv'}")
    return df


def budget_attack(model, X_test, y_test, feature_names, scaler=None, max_budget=10):
    """
    Greedy budget attack: attacker flips up to K features per phishing sample,
    choosing whichever flip most reduces the model's phishing probability.
    Plots F1 and accuracy vs. attacker budget K.
    """
    print_header(f"ADVERSARIAL: Budget Attack (K=1..{max_budget})")

    if scaler is not None:
        X_orig = scaler.inverse_transform(X_test)
        X_orig = np.round(X_orig).astype(int)
        X_orig = np.clip(X_orig, -1, 1)
    else:
        X_orig = X_test.copy().astype(int)

    y_test_arr = np.array(y_test)
    phish_mask = y_test_arr == 1
    n_features = X_orig.shape[1]

    accs, f1s = [], []

    for budget in range(0, max_budget + 1):
        X_adv_orig = X_orig.copy()

        if budget > 0:
            for sample_idx in np.where(phish_mask)[0]:
                row = X_adv_orig[sample_idx].copy()
                flipped = set()

                for _ in range(budget):
                    best_feat, best_gain = None, -np.inf

                    for feat_idx in range(n_features):
                        if feat_idx in flipped:
                            continue
                        if row[feat_idx] != 1:
                            continue  # only flip suspicious → 0

                        trial = row.copy()
                        trial[feat_idx] = 0
                        trial_scaled = scaler.transform(trial.reshape(1, -1).astype(float)) if scaler else trial.reshape(1, -1).astype(float)
                        gain = model.predict_proba(trial_scaled)[0, 1]  # lower prob = better for attacker... we want to minimize
                        # Actually we want the flip that MOST reduces phish probability
                        gain = -model.predict_proba(trial_scaled)[0, 1]

                        if gain > best_gain:
                            best_gain = gain
                            best_feat = feat_idx

                    if best_feat is not None:
                        row[best_feat] = 0
                        flipped.add(best_feat)

                X_adv_orig[sample_idx] = row

        if scaler is not None:
            X_adv = scaler.transform(X_adv_orig.astype(float))
        else:
            X_adv = X_adv_orig.astype(float)

        preds = model.predict(X_adv)
        accs.append(accuracy_score(y_test_arr, preds))
        f1s.append(f1_score(y_test_arr, preds))
        print(f"  Budget K={budget:2d} | Acc={accs[-1]:.4f} | F1={f1s[-1]:.4f}")

    # ── Plot ────────────────────────────────────────────────────────────────
    budgets = list(range(0, max_budget + 1))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(budgets, accs, "o-", color="#1f77b4", label="Accuracy")
    ax.plot(budgets, f1s,  "s--", color="#d62728", label="F1 (Phish)")
    ax.set_xlabel("Attacker Budget K (# features flipped per phishing URL)")
    ax.set_ylabel("Score")
    ax.set_title("Adversarial Robustness — Budget Attack on XGBoost")
    ax.legend()
    ax.set_ylim(0.5, 1.02)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "adversarial_budget_curve.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved: {PLOTS_DIR / 'adversarial_budget_curve.png'}")

    return budgets, accs, f1s


def plot_single_flip_bar(df, top_n=15):
    """Bar chart of accuracy drop per feature for single-flip attack."""
    df_top = df.head(top_n)
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#d62728" if d > 0.01 else "#ff7f0e" if d > 0.003 else "#aec7e8"
              for d in df_top["Acc_Drop"]]
    ax.barh(df_top["Feature"][::-1], df_top["Acc_Drop"][::-1], color=colors[::-1])
    ax.set_xlabel("Accuracy Drop after Feature Flip")
    ax.set_title(f"Adversarial Vulnerability — Top {top_n} Features (Single-Flip Attack)")
    ax.axvline(0, color="black", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "adversarial_feature_vulnerability.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {PLOTS_DIR / 'adversarial_feature_vulnerability.png'}")
