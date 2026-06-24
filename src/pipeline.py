import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config
from src.data_loader import load_dataset
from src.preprocessor import preprocess_data
from src.train_xgboost import train_xgboost
from src.train_ann import train_ann
from src.train_tabnet import train_tabnet
from src.train_advanced_models import train_random_forest, train_lightgbm
from src.train_stacking import train_stacking
import mlflow
mlflow.set_experiment("PhishGuard_AI")


def run_pipeline():
    config = load_config()
    art = config["artifacts"]
    art_dir = art["directory"]

    df = load_dataset(config["data"]["file_path"])
    feature_names = [c for c in df.columns if c != config["data"]["target_column"]]

    X_train, X_test, y_train, y_test = preprocess_data(
        df=df,
        target_col=config["data"]["target_column"],
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
        scaler_path=f"{art_dir}/{art['scaler_filename']}"
    )

    # ── Baseline models ──────────────────────────────────────────────────────
    train_xgboost(
        X_train, X_test, y_train, y_test,
        params=config["xgboost"],
        save_path=f"{art_dir}/{art['xgb_model_filename']}"
    )

    train_ann(
        X_train, X_test, y_train, y_test,
        params=config["mlp"],
        save_path=f"{art_dir}/{art['ann_model_filename']}"
    )

    # ── Novel: TabNet ────────────────────────────────────────────────────────
    train_tabnet(
        X_train, X_test, y_train, y_test,
        params=config["tabnet"],
        save_prefix=f"{art_dir}/{art['tabnet_model_prefix']}",
        feature_names=feature_names,
    )

    # ── Novel: Additional base learners ──────────────────────────────────────
    train_random_forest(
        X_train, X_test, y_train, y_test,
        params=config["random_forest"],
        save_path=f"{art_dir}/{art['rf_model_filename']}"
    )

    train_lightgbm(
        X_train, X_test, y_train, y_test,
        params=config["lightgbm"],
        save_path=f"{art_dir}/{art['lgbm_model_filename']}"
    )

    # ── Novel: Stacking Ensemble ─────────────────────────────────────────────
    train_stacking(
        X_train, X_test, y_train, y_test,
        save_path=f"{art_dir}/{art['stack_model_filename']}",
        n_folds=5
    )

    print("\n Pipeline Completed Successfully!")
