import numpy as np
import pandas as pd


def preprocess_data(df: pd.DataFrame, target_column: str = 'churn') -> pd.DataFrame:
    """Clean the raw churn dataset before feature engineering."""
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(' ', '_', regex=False)
        .str.replace('-', '_', regex=False)
    )

    if 'customer_id' in df.columns:
        df = df.drop(columns=['customer_id'])

    if 'total_charges' in df.columns:
        df['total_charges'] = pd.to_numeric(df['total_charges'], errors='coerce')

    if target_column in df.columns:
        df[target_column] = (
            df[target_column]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({'yes': 1, 'no': 0})
        )

    duplicate_count = int(df.duplicated().sum())
    if duplicate_count > 0:
        print(f"   🔁 Dropping {duplicate_count} duplicate rows")
        df = df.drop_duplicates()

    dropped_target = 0
    if target_column in df.columns:
        dropped_target = int(df[target_column].isna().sum())
        if dropped_target > 0:
            print(f"   ⚠️  Dropping {dropped_target} rows with invalid target values")
            df = df[df[target_column].notna()]

    numeric_cols = [
        col for col in df.select_dtypes(include=["number"]).columns
        if col != target_column
    ]
    for col in numeric_cols:
        if df[col].isna().any():
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)
            print(f"   ✨ Filled missing values in '{col}' with median={median_value:.2f}")

    categorical_cols = [
        col for col in df.select_dtypes(include=["object"]).columns
        if col != target_column
    ]
    if categorical_cols:
        df[categorical_cols] = df[categorical_cols].fillna('Unknown')

    for col in numeric_cols:
        lower = float(df[col].quantile(0.01))
        upper = float(df[col].quantile(0.99))
        if lower < upper:
            original_bounds = (df[col].min(), df[col].max())
            df[col] = df[col].clip(lower=lower, upper=upper)
            print(
                f"   🎯 Clipped '{col}' to [{lower:.2f}, {upper:.2f}] "
                f"from original range {original_bounds[0]:.2f}-{original_bounds[1]:.2f}"
            )

    if df.isna().sum().sum() > 0:
        fill_counts = df.isna().sum().loc[lambda x: x > 0].to_dict()
        print(f"   ⚠️  Filling remaining missing values: {fill_counts}")
        df = df.fillna(0)

    print(f"   ✅ Preprocessed dataset shape: {df.shape}")
    return df

