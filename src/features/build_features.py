import numpy as np
import pandas as pd


def _map_binary_series(s: pd.Series) -> pd.Series:
    """Convert deterministic 2-category values into binary 0/1 integers."""
    values = list(pd.Series(s.dropna().unique()).astype(str))
    categories = {v.strip().lower() for v in values}

    if categories <= {"yes", "no"} and categories:
        return s.map({"No": 0, "Yes": 1, "no": 0, "yes": 1}).astype("Int64")

    if len(values) == 2:
        sorted_values = sorted(values)
        mapping = {sorted_values[0]: 0, sorted_values[1]: 1}
        return s.astype(str).map(mapping).astype("Int64")

    return s


def build_features(df: pd.DataFrame, target_col: str = "churn") -> pd.DataFrame:
    """Create deterministic features for customer churn prediction."""
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )

    if target_col in df.columns:
        obj_cols = [c for c in df.select_dtypes(include=["object"]).columns if c != target_col]
    else:
        obj_cols = df.select_dtypes(include=["object"]).columns.tolist()

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    print(f"   📊 Found {len(obj_cols)} categorical and {len(numeric_cols)} numeric columns")

    df["is_new_customer"] = np.where(df["tenure"] <= 6, 1, 0)
    df["is_long_term_customer"] = np.where(df["tenure"] >= 36, 1, 0)
    df["monthly_charges_log"] = np.log(df["monthly_charges"].fillna(0) + 1)
    df["high_price_flag"] = np.where(df["monthly_charges"] > 81, 1, 0)
    df["price_to_tenure_ratio"] = df["monthly_charges"] / df["tenure"].replace(0, np.nan)
    df["price_to_tenure_ratio"] = df["price_to_tenure_ratio"].fillna(0)
    df["is_auto_pay"] = np.where(
        df["payment_method"].astype(str).str.contains("Credit|Debit|UPI", case=False, na=False),
        1,
        0,
    )
    df["high_support_calls"] = np.where(df["support_calls"] >= 3, 1, 0)
    df["support_calls_per_month"] = df["support_calls"] / df["tenure"].replace(0, np.nan)
    df["support_calls_per_month"] = df["support_calls_per_month"].fillna(0)
    df["recent_issue_proxy"] = np.where(
        (df["support_calls"] > 2) & (df["tenure"] <= 3),
        1,
        0,
    )
    df["high_price_new_customer"] = np.where(
        (df["high_price_flag"] == 1) & (df["is_new_customer"] == 1),
        1,
        0,
    )
    df["month_to_month_high_support"] = np.where(
        (df["contract"] == "Month-to-month") & (df["high_support_calls"] == 1),
        1,
        0,
    )
    df["long_term_low_support"] = np.where(
        (df["is_long_term_customer"] == 1) & (df["high_support_calls"] == 0),
        1,
        0,
    )

    obj_cols = [c for c in df.select_dtypes(include=["object"]).columns if c != target_col]
    binary_cols = []
    multi_cols = []
    for c in obj_cols:
        unique_values = set(df[c].dropna().astype(str).str.strip().str.lower())
        if unique_values <= {"yes", "no"}:
            binary_cols.append(c)
        elif df[c].dropna().nunique() == 2:
            binary_cols.append(c)
        elif df[c].dropna().nunique() > 2:
            multi_cols.append(c)

    print(f"   🔢 Binary features: {len(binary_cols)} | Multi-category features: {len(multi_cols)}")
    if binary_cols:
        print(f"      Binary: {binary_cols}")
    if multi_cols:
        print(f"      Multi-category: {multi_cols}")

    for c in binary_cols:
        original_dtype = df[c].dtype
        df[c] = _map_binary_series(df[c].astype(str))
        df[c] = df[c].fillna(0).astype(int)
        print(f"      ✅ {c}: {original_dtype} → binary (0/1)")

    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype(int)
        print(f"   🔄 Converted {len(bool_cols)} boolean columns to int: {bool_cols}")

    if multi_cols:
        print(f"   🌟 Applying one-hot encoding to {len(multi_cols)} multi-category columns...")
        original_shape = df.shape
        df = pd.get_dummies(df, columns=multi_cols, drop_first=True)
        new_features = df.shape[1] - original_shape[1] + len(multi_cols)
        print(f"      ✅ Created {new_features} new features from {len(multi_cols)} categorical columns")

    for c in binary_cols:
        if pd.api.types.is_integer_dtype(df[c]):
            df[c] = df[c].fillna(0).astype(int)

    print(f"✅ Feature engineering complete: {df.shape[1]} final features")
    return df

