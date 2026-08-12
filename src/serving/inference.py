import json
import os
import sys
from typing import Any, Dict

import joblib
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(ROOT_DIR, "src"))

from src.data.preprocess import preprocess_data
from src.features.build_features import build_features

ARTIFACT_DIR = os.path.join(ROOT_DIR, "artifacts")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "best_model.joblib")
PREPROCESSING_PATH = os.path.join(ARTIFACT_DIR, "preprocessing.pkl")
FEATURE_COLUMNS_PATH = os.path.join(ARTIFACT_DIR, "feature_columns.json")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Missing model archive at {MODEL_PATH}. Run scripts/run_pipeline.py first."
    )

if not os.path.exists(PREPROCESSING_PATH):
    raise FileNotFoundError(
        f"Missing preprocessing artifact at {PREPROCESSING_PATH}. Run scripts/run_pipeline.py first."
    )

if not os.path.exists(FEATURE_COLUMNS_PATH):
    raise FileNotFoundError(
        f"Missing feature column metadata at {FEATURE_COLUMNS_PATH}. Run scripts/run_pipeline.py first."
    )

MODEL = joblib.load(MODEL_PATH)
PREPROCESSING = joblib.load(PREPROCESSING_PATH)
FEATURE_COLUMNS = json.loads(open(FEATURE_COLUMNS_PATH, encoding="utf-8").read())
SCALER = PREPROCESSING.get("scaler")
NUMERIC_FEATURES = PREPROCESSING.get("payload", {}).get("numeric_features", [])


def _prepare_input(raw_data: Dict[str, Any]) -> pd.DataFrame:
    df = pd.DataFrame([raw_data])
    if "customer_id" in df.columns:
        df = df.drop(columns=["customer_id"])

    df = preprocess_data(df)
    df = build_features(df, target_col="churn")
    df = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)

    numeric_columns = [c for c in NUMERIC_FEATURES if c in df.columns]
    if SCALER is not None and numeric_columns:
        df[numeric_columns] = SCALER.transform(df[numeric_columns])

    return df


def predict(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    df = _prepare_input(raw_data)
    probabilities = MODEL.predict_proba(df)[:, 1]
    probability = float(probabilities[0])
    prediction_label = int(MODEL.predict(df)[0])

    return {
        "prediction": "Likely to churn" if prediction_label == 1 else "Not likely to churn",
        "probability": probability,
        "confidence": round(probability, 3),
        "raw_prediction": prediction_label,
    }
