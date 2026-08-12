# Customer Churn Prediction

This repository contains an end-to-end MLOps pipeline for customer churn prediction.
It includes raw data ingestion, validation, cleaning, feature engineering, model training with MLflow tracking, model selection, and a Streamlit production-ready UI.

## Project structure

- `data/raw/` - raw input dataset
- `data/processed/` - cleaned and processed dataset outputs
- `mlruns/` - MLflow experiment tracking
- `scripts/run_pipeline.py` - full training and model selection pipeline
- `scripts/run_eda.py` - exploratory data analysis report generator
- `app.py` - Streamlit customer churn prediction app
- `src/` - reusable data, feature, model, and serving modules
- `artifacts/` - saved model and preprocessing artifacts

## Getting started

1. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

2. Run the data pipeline:

```bash
python scripts/run_pipeline.py --input data/raw/customer_churn_dataset.csv
```

This will:
- validate the raw dataset
- preprocess and save cleaned data to `data/processed/`
- engineer features
- train logistic regression, random forest, and XGBoost candidates
- select the best model by F1 score
- save the best model to `artifacts/best_model.joblib`
- log parameters, metrics, and artifacts to `mlruns/`

3. Generate an EDA report:

```bash
python scripts/run_eda.py --input data/raw/customer_churn_dataset.csv
```

The report will be written to `reports/eda_summary.md` and the correlation matrix to `reports/correlation_matrix.csv`.

4. Run the Streamlit app:

```bash
streamlit run app.py
```

## Production serving

The Streamlit UI loads the serialized best model and preprocessing artifacts from `artifacts/`.
It uses the same preprocessing and feature engineering code as training to prevent train/serve skew.

## Notes

- The `scripts/run_pipeline.py` pipeline uses MLflow local tracking to keep model comparisons reproducible.
- The Streamlit app provides clear input validation, confidence indicators, and a calm, minimal interface.
- If you update the raw dataset, rerun the pipeline before launching the app.
