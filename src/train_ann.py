import joblib
import mlflow
import mlflow.sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from src.utils import print_header


def train_ann(X_train, X_test, y_train, y_test, params, save_path):
    print_header("TRAINING ANN MLPClassifier (MLflow Enabled)")

    params = params.copy()
    if "hidden_layers" in params:
        params["hidden_layer_sizes"] = tuple(params["hidden_layers"])
        del params["hidden_layers"]

    with mlflow.start_run(run_name="ANN_MLP_Phishing"):

        mlflow.log_params(params)

        ann = MLPClassifier(**params)
        ann.fit(X_train, y_train)

        preds = ann.predict(X_test)
        proba = ann.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, proba)
        report = classification_report(y_test, preds, output_dict=True)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric("precision", report["1"]["precision"])
        mlflow.log_metric("recall", report["1"]["recall"])
        mlflow.log_metric("f1_score", report["1"]["f1-score"])

        joblib.dump(ann, save_path)
        mlflow.sklearn.log_model(
            ann,
            name="ann_model",
            skops_trusted_types=[
                "sklearn.neural_network._multilayer_perceptron.MLPClassifier",
                "sklearn.neural_network._stochastic_optimizers.AdamOptimizer",
                "sklearn.neural_network._stochastic_optimizers.SGDOptimizer",
            ],
        )

        print("ANN Accuracy:", acc)
        print("ANN ROC-AUC :", auc)
        print("Report:\n", classification_report(y_test, preds))
        print(f"ANN model saved → {save_path}")

        return ann, preds, proba, report
