
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import joblib
import mlflow
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from src.utils import print_header


def train_tabnet(X_train, X_test, y_train, y_test, params, save_prefix, feature_names=None):
    print_header("TRAINING TabNet — Attention-Based Tabular DL (MLflow Enabled)")

    tab_params = {
        "n_d":              params.get("n_d", 32),
        "n_a":              params.get("n_a", 32),
        "n_steps":          params.get("n_steps", 5),
        "gamma":            params.get("gamma", 1.3),
        "lambda_sparse":    params.get("lambda_sparse", 1e-4),
        "momentum":         params.get("momentum", 0.02),
        "epsilon":          params.get("epsilon", 1e-15),
        "seed":             params.get("seed", 42),
        "verbose":          0,
    }

    fit_params = {
        "max_epochs":        params.get("max_epochs", 200),
        "patience":          params.get("patience", 20),
        "batch_size":        params.get("batch_size", 256),
        "virtual_batch_size": params.get("virtual_batch_size", 128),
        "drop_last":         False,
    }

    with mlflow.start_run(run_name="TabNet_Phishing"):
        mlflow.log_params({**tab_params, **fit_params})

        model = TabNetClassifier(**tab_params)
        model.fit(
            X_train, y_train.values,
            eval_set=[(X_test, y_test.values)],
            eval_metric=["accuracy"],
            **fit_params
        )

        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, proba)
        report = classification_report(y_test, preds, output_dict=True)

        mlflow.log_metric("accuracy",  acc)
        mlflow.log_metric("roc_auc",   auc)
        mlflow.log_metric("precision", report["1"]["precision"])
        mlflow.log_metric("recall",    report["1"]["recall"])
        mlflow.log_metric("f1_score",  report["1"]["f1-score"])

        # TabNet native save (saves JSON + zip)
        model.save_model(save_prefix)
        # Also pickle the full object for joblib.load compatibility
        joblib.dump(model, save_prefix + ".pkl")

        # Log global feature importance (average attention across all steps & samples)
        if feature_names is not None:
            fi = model.feature_importances_
            for name, imp in zip(feature_names, fi):
                mlflow.log_metric(f"feat_imp_{name}", float(imp))

        print("TabNet Accuracy:", acc)
        print("TabNet ROC-AUC :", auc)
        print("TabNet Best Epoch:", model.best_epoch)
        print("Classification Report:\n", classification_report(y_test, preds))
        print(f"TabNet model saved → {save_prefix}(.zip / .pkl)")

    return model, preds, proba, report
