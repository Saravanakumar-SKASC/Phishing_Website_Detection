import joblib
import mlflow
import mlflow.xgboost

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from src.utils import print_header


def train_xgboost(X_train, X_test, y_train, y_test, params, save_path):
    print_header("TRAINING XGBOOST (MLflow Enabled)")

    params = params.copy()

    with mlflow.start_run(run_name="XGBoost_Phishing"):

        mlflow.log_params(params)

        model = XGBClassifier(**params)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, proba)
        report = classification_report(y_test, preds, output_dict=True)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric("precision", report["1"]["precision"])
        mlflow.log_metric("recall", report["1"]["recall"])
        mlflow.log_metric("f1_score", report["1"]["f1-score"])

        joblib.dump(model, save_path)
        mlflow.xgboost.log_model(model, name="xgboost_model")

        print("XGBoost Accuracy:", acc)
        print("XGBoost ROC-AUC :", auc)
        print("Classification Report:\n", classification_report(y_test, preds))
        print(f"XGBoost model saved → {save_path}")

        return model, preds, proba, report
