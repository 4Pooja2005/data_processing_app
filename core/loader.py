import pandas as pd

def load_file(path):
    """Load CSV or Excel file as a DataFrame"""
    if path.endswith(".csv"):
        return pd.read_csv(path)
    elif path.endswith(".xlsx") or path.endswith(".xls"):
        return pd.read_excel(path)
    else:
        raise ValueError("Unsupported file format")
