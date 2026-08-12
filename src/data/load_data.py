import os
import pandas as pd

REQUIRED_COLUMNS = {
    "customer_id",
    "tenure",
    "monthly_charges",
    "total_charges",
    "contract",
    "payment_method",
    "internet_service",
    "tech_support",
    "online_security",
    "support_calls",
    "churn",
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )
    return df


def load_data(file_path: str) -> pd.DataFrame:
    """Load a raw CSV dataset and enforce expected churn schema."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    df = pd.read_csv(file_path)
    df = _normalize_columns(df)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    return df
