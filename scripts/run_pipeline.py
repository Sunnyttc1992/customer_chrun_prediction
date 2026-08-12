"""
Runs sequentially: load → validate → preprocess → feature engineering → train → select best model.
"""

import argparse
import json
import os
import sys
import time

import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Allow local package imports from the repository root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features
from src.utils.validate_data import validate_telco_data


def _save_artifacts(artifacts_dir: str, feature_columns, scaler, numeric_features):
    os.makedirs(artifacts_dir, exist_ok=True)

    with open(os.path.join(artifacts_dir, "feature_columns.json"), "w", encoding="utf-8") as f:
        json.dump(feature_columns, f, indent=2)

    artifact_payload = {
        "feature_columns": feature_columns,
        "numeric_features": numeric_features,
    }
    joblib.dump({"payload": artifact_payload, "scaler": scaler}, os.path.join(artifacts_dir, "preprocessing.pkl"))

    print(f"✅ Saved artifacts: {len(feature_columns)} feature columns, {len(numeric_features)} numeric scalers")


def _build_classifier_candidates(scale_pos_weight: float):
    return {
        "logistic_regression": LogisticRegression(
            solver="liblinear",
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ),
        "xgboost": XGBClassifier(
            n_estimators=250,
            learning_rate=0.06,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
            n_jobs=-1,
        ),
    }


def _score_model(model, X_test: pd.DataFrame, y_test: pd.Series, threshold: float):
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    return {
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "threshold": threshold,
    }


def _log_model_run(model_name: str, model, params: dict, metrics: dict, X_train, X_test, y_train, y_test):
    mlflow.log_param("model_name", model_name)
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)

    if model_name == "xgboost":
        mlflow.xgboost.log_model(model, artifact_path="model")
    else:
        mlflow.sklearn.log_model(model, artifact_path="model")

    mlflow.log_text(
        json.dumps({"feature_count": X_train.shape[1], "train_rows": X_train.shape[0]}, indent=2),
        artifact_file="dataset_summary.json",
    )


def main(args):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    mlruns_path = args.mlflow_uri or f"file://{project_root}/mlruns"
    mlflow.set_tracking_uri(mlruns_path)
    mlflow.set_experiment(args.experiment)

    raw_path = args.input
    artifacts_dir = os.path.join(project_root, "artifacts")
    model_path = os.path.join(artifacts_dir, "best_model.joblib")

    print("🔄 Loading raw dataset...")
    df = load_data(raw_path)
    print(f"✅ Loaded dataset with shape {df.shape}")

    print("🔍 Validating raw data...")
    valid, issues = validate_telco_data(df)
    if not valid:
        raise ValueError(f"Data validation failed: {issues}")

    print("🔧 Cleaning and preprocessing...")
    df = preprocess_data(df)

    processed_path = os.path.join(project_root, "data", "processed", "processed_customer_churn.csv")
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df.to_csv(processed_path, index=False)
    print(f"✅ Saved processed data to {processed_path}")

    target_col = args.target.lower()
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found after preprocessing")

    print("🛠️  Engineering features...")
    df_enc = build_features(df, target_col=target_col)
    for c in df_enc.select_dtypes(include=["bool"]).columns:
        df_enc[c] = df_enc[c].astype(int)

    feature_cols = [c for c in df_enc.columns if c != target_col]
    numeric_features = [c for c in df_enc[feature_cols].select_dtypes(include=["number"]).columns]

    print(f"✅ Feature engineering created {len(feature_cols)} features")

    scaler = StandardScaler()
    scaler.fit(df_enc[numeric_features])
    df_enc[numeric_features] = scaler.transform(df_enc[numeric_features])
    print("✅ Scaled numeric features for model training")

    _save_artifacts(artifacts_dir, feature_cols, scaler, numeric_features)

    with mlflow.start_run(run_name="data_preparation"):
        mlflow.log_metric("data_quality_pass", int(valid))
        mlflow.log_text(json.dumps(issues, indent=2), artifact_file="data_validation.json")
        mlflow.log_artifact(processed_path, artifact_path="processed_data")
        mlflow.log_artifact(os.path.join(artifacts_dir, "preprocessing.pkl"))
        mlflow.log_artifact(os.path.join(artifacts_dir, "feature_columns.json"))
        mlflow.log_param("scale_pos_weight", float((df_enc[target_col] == 0).sum() / max((df_enc[target_col] == 1).sum(), 1)))

    X = df_enc.drop(columns=[target_col])
    y = df_enc[target_col].astype(int)

    print("📊 Splitting train/test data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        stratify=y,
        random_state=42,
    )
    print(f"✅ Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

    candidates = _build_classifier_candidates(scale_pos_weight)
    best_model = None
    best_score = -1
    best_metrics = None
    best_name = None
    best_params = None

    for model_name, estimator in candidates.items():
        print(f"🚀 Training candidate: {model_name}")
        t0 = time.time()
        estimator.fit(X_train, y_train)
        elapsed = time.time() - t0

        metrics = _score_model(estimator, X_test, y_test, args.threshold)
        metrics["train_time"] = elapsed
        print(f"   {model_name}: F1={metrics['f1']:.3f}, ROC AUC={metrics['roc_auc']:.3f}")

        with mlflow.start_run(run_name=f"candidate_{model_name}"):
            _log_model_run(model_name, estimator, estimator.get_params(), metrics, X_train, X_test, y_train, y_test)
            mlflow.log_param("scale_pos_weight", scale_pos_weight)

        if metrics["f1"] > best_score:
            best_score = metrics["f1"]
            best_model = estimator
            best_metrics = metrics
            best_name = model_name
            best_params = estimator.get_params()

    if best_model is None:
        raise RuntimeError("No candidate model was successfully trained")

    joblib.dump(best_model, model_path)
    print(f"💾 Best model saved to {model_path}")

    with mlflow.start_run(run_name="best_model"):
        mlflow.log_param("best_model", best_name)
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items() if k in ["n_estimators", "max_depth", "learning_rate", "solver"]})
        mlflow.log_metrics({f"best_{k}": v for k, v in best_metrics.items()})
        mlflow.log_artifact(model_path, artifact_path="best_model")

    print("🎯 Model selection complete")
    print(f"   Best model: {best_name} (F1={best_score:.3f})")
    print(f"   Precision: {best_metrics['precision']:.3f}, Recall: {best_metrics['recall']:.3f}, ROC AUC: {best_metrics['roc_auc']:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the customer churn MLOps pipeline")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to raw CSV file, e.g. data/raw/customer_churn_dataset.csv",
    )
    parser.add_argument("--target", type=str, default="churn", help="Name of the target column")
    parser.add_argument("--threshold", type=float, default=0.35, help="Classification threshold for inference")
    parser.add_argument("--test_size", type=float, default=0.2, help="Fraction of data reserved for testing")
    parser.add_argument("--experiment", type=str, default="Telco Churn", help="MLflow experiment name")
    parser.add_argument(
        "--mlflow_uri",
        type=str,
        default=None,
        help="Optional MLflow tracking URI override",
    )
    args = parser.parse_args()
    main(args)
