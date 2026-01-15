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
# minimal cleaning + type fixes only
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('-', '_')

# run build_features while payment_method still exists
df = build_features(df, target_col='churn')

# now preprocess (but remove target encoding from preprocess to avoid double work)
df = preprocess_data(df)
# save processed data
os.makedirs(os.path.dirname(PROCESSED), exist_ok=True)
df.to_csv(PROCESSED, index=False)
print(f"✅ Processed data saved to {PROCESSED}")
