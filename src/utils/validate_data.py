from typing import Dict, List, Tuple

import pandas as pd


def validate_telco_data(df: pd.DataFrame) -> Tuple[bool, Dict[str, List[str]]]:
    """Validate raw Telco churn data for schema correctness and basic business rules."""
    issues = {
        "missing_columns": [],
        "null_values": [],
        "invalid_categories": [],
        "range_violations": [],
    }

    required_columns = {
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

    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        issues["missing_columns"] = missing_columns

    nullable_categoricals = {
        "contract",
        "payment_method",
        "internet_service",
        "tech_support",
        "online_security",
    }
    for col in required_columns & set(df.columns):
        if df[col].isnull().any():
            if col not in nullable_categoricals:
                issues["null_values"].append(col)

    if "contract" in df.columns:
        allowed = {"Month-to-month", "One year", "Two year", "Unknown"}
        invalid = sorted(set(df["contract"].dropna().unique()) - allowed)
        if invalid:
            issues["invalid_categories"].append(f"contract: {invalid}")

    if "internet_service" in df.columns:
        allowed = {"DSL", "Fiber", "No", "Unknown"}
        invalid = sorted(set(df["internet_service"].dropna().unique()) - allowed)
        if invalid:
            issues["invalid_categories"].append(f"internet_service: {invalid}")

    if "payment_method" in df.columns:
        allowed = {"Cash", "Credit", "Debit", "UPI", "Unknown"}
        invalid = sorted(set(df["payment_method"].dropna().unique()) - allowed)
        if invalid:
            issues["invalid_categories"].append(f"payment_method: {invalid}")

    if "tech_support" in df.columns:
        allowed = {"Yes", "No", "yes", "no", "Unknown"}
        invalid = sorted(set(df["tech_support"].dropna().unique()) - allowed)
        if invalid:
            issues["invalid_categories"].append(f"tech_support: {invalid}")

    if "online_security" in df.columns:
        allowed = {"Yes", "No", "yes", "no", "Unknown"}
        invalid = sorted(set(df["online_security"].dropna().unique()) - allowed)
        if invalid:
            issues["invalid_categories"].append(f"online_security: {invalid}")

    if "churn" in df.columns:
        allowed = {"Yes", "No", "yes", "no", 0, 1}
        invalid = sorted(set(df["churn"].dropna().unique()) - allowed)
        if invalid:
            issues["invalid_categories"].append(f"churn: {invalid}")

    if "tenure" in df.columns:
        invalid_count = int((~df["tenure"].between(0, 120)).sum())
        if invalid_count:
            issues["range_violations"].append(f"tenure: {invalid_count} rows outside [0, 120]")

    if "monthly_charges" in df.columns:
        invalid_count = int((~df["monthly_charges"].between(0, 200)).sum())
        if invalid_count:
            issues["range_violations"].append(f"monthly_charges: {invalid_count} rows outside [0, 200]")

    if "total_charges" in df.columns:
        invalid_count = int((~df["total_charges"].between(0, 10000)).sum())
        if invalid_count:
            issues["range_violations"].append(f"total_charges: {invalid_count} rows outside [0, 10000]")

    if "support_calls" in df.columns:
        invalid_count = int((df["support_calls"] < 0).sum())
        if invalid_count:
            issues["range_violations"].append(f"support_calls: {invalid_count} negative values")

    success = all(len(v) == 0 for v in issues.values())

    print("🔍 Data validation summary:")
    print(f"   Required columns present: {len(required_columns) - len(issues['missing_columns'])}/{len(required_columns)}")
    print(f"   Columns with null values: {issues['null_values']}")
    print(f"   Invalid categorical values: {issues['invalid_categories']}")
    print(f"   Numeric range issues: {issues['range_violations']}")

    return success, issues
