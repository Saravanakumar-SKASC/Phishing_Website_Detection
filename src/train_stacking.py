
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import joblib
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from src.utils import print_header


def _make_base_learners():
    """Return fresh instances of all 4 base learners (no TabNet — sklearn API only)."""
    return {
        "XGBoost": XGBClassifier(
            n_estimators=400, max_depth=7, learning_rate=0.04,
            subsample=0.9, colsample_bytree=0.9, gamma=0.2,
            min_child_weight=3, reg_alpha=0.3, reg_lambda=1.0,
            n_jobs=-1, eval_metric="logloss", random_state=42,
        ),
        "ANN": MLPClassifier(
            hidden_layer_sizes=(128, 64, 32), activation="relu",
            solver="adam", learning_rate_init=0.001, alpha=0.0005,
            batch_size=64, max_iter=400, early_stopping=True,
            validation_fraction=0.1, random_state=42,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=500, max_depth=15, min_samples_split=5,
            min_samples_leaf=2, max_features="sqrt",
            n_jobs=-1, class_weight="balanced", random_state=42,
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=400, max_depth=7, learning_rate=0.04,
            subsample=0.9, colsample_bytree=0.9, num_leaves=63,
            reg_alpha=0.3, reg_lambda=1.0, min_child_samples=20,
            n_jobs=-1, verbose=-1, random_state=42,
        ),
    }


def train_stacking(X_train, X_test, y_train, y_test, save_path, n_folds=5):
    print_header("TRAINING Stacking Ensemble — OOF Meta-Learning (MLflow Enabled)")

    base_learners = _make_base_learners()
    n_base = len(base_learners)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    # ── Step 1: Generate OOF meta-features for training set ──────────────────
    print(f"\nGenerating {n_folds}-fold OOF predictions for {n_base} base learners...")
    oof_train = np.zeros((X_train.shape[0], n_base))
    test_preds = np.zeros((X_test.shape[0], n_base))

    for col_idx, (name, clf) in enumerate(base_learners.items()):
        fold_test_preds = np.zeros((n_folds, X_test.shape[0]))

        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_tr, X_val = X_train[tr_idx], X_train[val_idx]
            y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

            clf.fit(X_tr, y_tr)
            oof_train[val_idx, col_idx] = clf.predict_proba(X_val)[:, 1]
            fold_test_preds[fold] = clf.predict_proba(X_test)[:, 1]

        # Average test predictions across folds
        test_preds[:, col_idx] = fold_test_preds.mean(axis=0)
        print(f"  [{col_idx+1}/{n_base}] {name} OOF done")

    # ── Step 2: Train meta-learner on OOF predictions ────────────────────────
    print("\nTraining meta-learner (Logistic Regression) on OOF features...")
    meta = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=42)
    meta.fit(oof_train, y_train)

    # ── Step 3: Evaluate on test set ─────────────────────────────────────────
    preds = meta.predict(test_preds)
    proba = meta.predict_proba(test_preds)[:, 1]

    acc   = accuracy_score(y_test, preds)
    auc   = roc_auc_score(y_test, proba)
    report = classification_report(y_test, preds, output_dict=True)

    # ── Step 4: Log to MLflow ────────────────────────────────────────────────
    with mlflow.start_run(run_name="StackingEnsemble_Phishing"):
        mlflow.log_param("base_learners", list(base_learners.keys()))
        mlflow.log_param("meta_learner",  "LogisticRegression")
        mlflow.log_param("n_folds",       n_folds)
        mlflow.log_metric("accuracy",  acc)
        mlflow.log_metric("roc_auc",   auc)
        mlflow.log_metric("precision", report["1"]["precision"])
        mlflow.log_metric("recall",    report["1"]["recall"])
        mlflow.log_metric("f1_score",  report["1"]["f1-score"])

        # Log meta-learner coefficients (how much it trusts each base model)
        for name, coef in zip(base_learners.keys(), meta.coef_[0]):
            mlflow.log_metric(f"meta_coef_{name}", float(coef))

        bundle = {"meta": meta, "base_learners": base_learners,
                  "oof_train": oof_train, "test_preds": test_preds}
        joblib.dump(bundle, save_path)

    print(f"\nMeta-learner trust coefficients:")
    for name, coef in zip(base_learners.keys(), meta.coef_[0]):
        print(f"  {name:15s}: {coef:+.4f}")

    print("\nStacking Ensemble Accuracy:", acc)
    print("Stacking Ensemble ROC-AUC :", auc)
    print("Classification Report:\n", classification_report(y_test, preds))
    print(f"Stacking model bundle saved → {save_path}")

    return meta, base_learners, preds, proba, report, meta.coef_[0]
