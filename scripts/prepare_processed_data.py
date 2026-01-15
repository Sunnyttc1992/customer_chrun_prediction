import os , sys
import pandas as pd

# make src importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from src.data.preprocess import preprocess_data
from src.features.build_features import build_features

RAW = "data/raw/customer_churn_dataset.csv"
PROCESSED = "data/processed/processed_customer_churn.csv"

# 1. Load raw data
df = pd.read_csv(RAW)
# 2. Process data
df = preprocess_data(df)
# 3 ensure target is 0/1 only if still object

if df['churn'].dtype == 'object':
    df['churn'] = df['churn'].map({'No': 0, 'Yes': 1}).astype('Int64')

# Sanity check target values
assert set(df['churn'].unique()).issubset({0, 1}), "Target column 'churn' must contain only 0 and 1 values."
assert df['churn'].isnull().sum() == 0, "Target column 'churn' contains null values."

# 4. Build features
df = build_features(df, target_col='churn')

# 5. Save processed data
os.makedirs(os.path.dirname(PROCESSED), exist_ok=True)
df.to_csv(PROCESSED, index=False)
print(f"✅ Processed data saved to {PROCESSED} with shape {df.shape}")
