import argparse
import os
from pathlib import Path

import pandas as pd

from src.data.load_data import load_data


REPORT_DIR = Path("reports")


def generate_eda_report(df: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "eda_summary.md"
    correlation_path = output_dir / "correlation_matrix.csv"

    target = "churn"
    churn_series = df[target].astype(str).str.lower().map({"yes": 1, "no": 0})

    lines = [
        "# EDA Summary",
        "",
        f"- Rows: {df.shape[0]}",
        f"- Columns: {df.shape[1]}",
        "",
        "## Missing values",
        "",
    ]

    missing = df.isna().sum()
    for col, count in missing.items():
        lines.append(f"- {col}: {count}")

    lines.extend([
        "",
        "## Target distribution",
        "",
        f"- Churn rate: {churn_series.mean():.3f} ({int(churn_series.sum())}/{len(churn_series)})",
        "",
        "## Numeric summaries",
        "",
    ])

    numeric = df.select_dtypes(include=["number"]).describe().transpose()
    lines.extend([f"- {idx}: mean={row['mean']:.2f}, std={row['std']:.2f}, min={row['min']:.2f}, max={row['max']:.2f}" for idx, row in numeric.iterrows()])

    lines.extend([
        "",
        "## Categorical top values",
        "",
    ])

    for col in df.select_dtypes(include=["object"]).columns:
        top = df[col].value_counts().head(5)
        lines.append(f"- **{col}**")
        for value, count in top.items():
            lines.append(f"  - {value}: {count}")
        lines.append("")

    corr = df.select_dtypes(include=["number"]).corr()
    corr.to_csv(correlation_path, index=True)

    lines.extend([
        "## Correlation matrix",
        "",
        f"Correlation matrix saved to `{correlation_path}`.",
    ])

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main():
    parser = argparse.ArgumentParser(description="Run EDA and save a summary report.")
    parser.add_argument("--input", required=True, help="Path to raw churn CSV")
    parser.add_argument("--output", default="reports", help="Directory to write reports")
    args = parser.parse_args()

    df = load_data(args.input)
    report_path = generate_eda_report(df, Path(args.output))
    print(f"✅ EDA summary written to {report_path}")


if __name__ == "__main__":
    main()
