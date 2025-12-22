import pandas as pd

# -------- Per-file operations --------
def remove_duplicates(df):
    return df.drop_duplicates()

def handle_missing_values(df, method="delete", fill_value=None):
    if method == "delete":
        return df.dropna()
    elif method == "zero":
        return df.fillna(0)
    elif method == "fill" and fill_value is not None:
        return df.fillna(fill_value)
    return df

def standardize_data(df):
    df_copy = df.copy()
    for col in df_copy.columns:
        if df_copy[col].dtype == object:
            df_copy[col] = df_copy[col].str.strip().str.lower()
        else:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='ignore')
    return df_copy

# -------- Cross-file operation (example merge) --------
def merge_datasets(dfs):
    return pd.concat(dfs, ignore_index=True)
