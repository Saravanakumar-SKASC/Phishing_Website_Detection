
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import lightgbm as lgb
from src.utils import print_header


def train_random_forest(X_train, X_test, y_train, y_test, params, save_path):
    print_header("TRAINING Random Forest (MLflow Enabled)")

    with mlflow.start_run(run_name="RandomForest_Phishing"):
        mlflow.log_params(params)

        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]

        acc   = accuracy_score(y_test, preds)
        auc   = roc_auc_score(y_test, proba)
        report = classification_report(y_test, preds, output_dict=True)

        mlflow.log_metric("accuracy",  acc)
        mlflow.log_metric("roc_auc",   auc)
        mlflow.log_metric("precision", report["1"]["precision"])
        mlflow.log_metric("recall",    report["1"]["recall"])
        mlflow.log_metric("f1_score",  report["1"]["f1-score"])

        joblib.dump(model, save_path)
        mlflow.sklearn.log_model(model, name="rf_model")

        print("Random Forest Accuracy:", acc)
        print("Random Forest ROC-AUC :", auc)
        print("Classification Report:\n", classification_report(y_test, preds))
        print(f"Random Forest model saved → {save_path}")

    return model, preds, proba, report


def train_lightgbm(X_train, X_test, y_train, y_test, params, save_path):
    print_header("TRAINING LightGBM (MLflow Enabled)")

    with mlflow.start_run(run_name="LightGBM_Phishing"):
        mlflow.log_params(params)

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)],
        )

        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]

        acc   = accuracy_score(y_test, preds)
        auc   = roc_auc_score(y_test, proba)
        report = classification_report(y_test, preds, output_dict=True)

        mlflow.log_metric("accuracy",  acc)
        mlflow.log_metric("roc_auc",   auc)
        mlflow.log_metric("precision", report["1"]["precision"])
        mlflow.log_metric("recall",    report["1"]["recall"])
        mlflow.log_metric("f1_score",  report["1"]["f1-score"])

        joblib.dump(model, save_path)

        print("LightGBM Accuracy:", acc)
        print("LightGBM ROC-AUC :", auc)
        print("Classification Report:\n", classification_report(y_test, preds))
        print(f"LightGBM model saved → {save_path}")

    return model, preds, proba, report
