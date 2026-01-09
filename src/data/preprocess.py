import pandas as pd


def preprocess_data(df: pd.DataFrame, target_column: str= 'churn') -> pd.DataFrame:
    """
    Preprocess the input DataFrame by handling missing values and encoding categorical variables.

    Parameters:
    df (pd.DataFrame): The input DataFrame to preprocess.
    target_column (str): The name of the target column to exclude from preprocessing.

    Returns:
    pd.DataFrame: The preprocessed DataFrame.
    """
    # tidy header
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
    
    #drop unnecessary columns
    if 'customer_id' in df.columns:
        df = df.drop(columns=['customer_id'])

    # Handle missing values
    df = df.fillna(df.median(numeric_only=True))
    df = df.fillna('Unknown')

    # Encode categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    categorical_cols = [col for col in categorical_cols if col != target_column]

    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # target to 0/1 it's yes/no
    if target_column in df.columns and df[target_column].dtype == 'object':
        df[target_column] = df[target_column].map({'yes': 1, 'no': 0})

    # Total charges ofthen has blanks in this dataset ->coerce to flot
    if 'total_charges' in df.columns:
        df['total_charges'] = pd.to_numeric(df['total_charges'], errors='coerce')

    # NA strategy
    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = df[num_cols].fillna(0)


    return df

