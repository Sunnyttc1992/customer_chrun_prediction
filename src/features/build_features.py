import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def _map_binary_series(s: pd.Series) -> pd.Series:
    """
    Apply deterministic binary encoding to 2-category features.
    
    This function implements the core binary encoding logic that converts
    categorical features with exactly 2 values into 0/1 integers. The mappings
    are deterministic and must be consistent between training and serving.

    """
    # Get unique values and remove NaN
    vals = list(pd.Series(s.dropna().unique()).astype(str))
    valset = set(vals)

    # === DETERMINISTIC BINARY MAPPINGS ===
    # CRITICAL: These exact mappings are hardcoded in serving pipeline
    
    # Yes/No mapping (most common pattern in telecom data)
    if valset == {"Yes", "No"}:
        return s.map({"No": 0, "Yes": 1}).astype("Int64")


    # === GENERIC BINARY MAPPING ===
    # For any other 2-category feature, use stable alphabetical ordering
    if len(vals) == 2:
        # Sort values to ensure consistent mapping across runs
        sorted_vals = sorted(vals)
        mapping = {sorted_vals[0]: 0, sorted_vals[1]: 1}
        return s.astype(str).map(mapping).astype("Int64")

    # === NON-BINARY FEATURES ===
    # Return unchanged - will be handled by one-hot encoding
    return s


def build_features(df: pd.DataFrame, target_col: str = "churn") -> pd.DataFrame:
    """
    Build engineered features for customer churn prediction.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing raw customer features.

    Returns
    -------
    pd.DataFrame
        DataFrame with engineered features added.
    """

    df = df.copy()
    # Find categorical columns (object dtype) excluding the target variable
    obj_cols = [c for c in df.select_dtypes(include=["object"]).columns if c != target_col]
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
   
    print(f"   📊 Found {len(obj_cols)} categorical and {len(numeric_cols)} numeric columns")

    # ---- Tenure-based features ----
    df["is_new_customer"] = np.where(df["tenure"] <= 6, 1, 0)
    df["is_long_term_customer"] = np.where(df["tenure"] >= 36, 1, 0)

    # ---- Pricing & spend ----
    df["monthly_charges_log"] = np.log(df["monthly_charges"] + 1)
    df["high_price_flag"] = np.where(df["monthly_charges"] > 81, 1, 0)

    # Avoid division by zero
    df["price_to_tenure_ratio"] = df["monthly_charges"] / df["tenure"].replace(0, np.nan)

    # ---- Payment method ----
    df["is_auto_pay"] = np.where(
        df["payment_method"].str.contains("Credit|Debit|UPI", case=False, na=False),
        1,
        0,
    )

    # ---- Support calls ----
    df["high_support_calls"] = np.where(df["support_calls"] >= 3, 1, 0)
    df["support_calls_per_month"] = df["support_calls"] / df["tenure"].replace(0, np.nan)
    df["recent_issue_proxy"] = np.where(
        (df["support_calls"] > 2) & (df["tenure"] <= 3), 1, 0
    )

    # ---- Interaction features ----
    df["high_price_new_customer"] = np.where(
        (df["high_price_flag"] == 1) & (df["is_new_customer"] == 1), 1, 0
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

    # Find categorical columns (object dtype) excluding the target variable
    obj_cols = [c for c in df.select_dtypes(include=["object"]).columns if c != target_col]
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    

    # Binary features (exactly 2 unique values) get binary encoding
    # Multi-category features (>2 unique values) get one-hot encoding
    binary_cols = [c for c in obj_cols if df[c].dropna().nunique() == 2]
    multi_cols = [c for c in obj_cols if df[c].dropna().nunique() > 2]
    
    print(f"   🔢 Binary features: {len(binary_cols)} | Multi-category features: {len(multi_cols)}")
    if binary_cols:
        print(f"      Binary: {binary_cols}")
    if multi_cols:
        print(f"      Multi-category: {multi_cols}")

    # === STEP 3: Apply Binary Encoding ===
    # Convert 2-category features to 0/1 using deterministic mappings
    for c in binary_cols:
        original_dtype = df[c].dtype
        df[c] = _map_binary_series(df[c].astype(str))
        print(f"      ✅ {c}: {original_dtype} → binary (0/1)")

    # === STEP 4: Convert Boolean Columns ===
    # XGBoost requires integer inputs, not boolean
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype(int)
        print(f"   🔄 Converted {len(bool_cols)} boolean columns to int: {bool_cols}")

    # === STEP 5: One-Hot Encoding for Multi-Category Features ===
    # CRITICAL: drop_first=True prevents multicollinearity
    if multi_cols:
        print(f"   🌟 Applying one-hot encoding to {len(multi_cols)} multi-category columns...")
        original_shape = df.shape
        
        # Apply one-hot encoding with drop_first=True (same as serving)
        df = pd.get_dummies(df, columns=multi_cols, drop_first=True)
        
        new_features = df.shape[1] - original_shape[1] + len(multi_cols)
        print(f"      ✅ Created {new_features} new features from {len(multi_cols)} categorical columns")

    # === STEP 6: Data Type Cleanup ===
    # Convert nullable integers (Int64) to standard integers for XGBoost
    for c in binary_cols:
        if pd.api.types.is_integer_dtype(df[c]):
            # Fill any NaN values with 0 and convert to int
            df[c] = df[c].fillna(0).astype(int)

    print(f"✅ Feature engineering complete: {df.shape[1]} final features")
    
    return df

