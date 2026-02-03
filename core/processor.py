import pandas as pd
from num2words import num2words
from word2number import w2n

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
    """Standardize entire dataframe based on column types"""
    df_copy = df.copy()
    for col in df_copy.columns:
        if df_copy[col].dtype == object:
            # Apply strip and lowercase (equivalent to previous logic)
            df_copy = standardize_column(df_copy, col, 'strip')
            df_copy = standardize_column(df_copy, col, 'lowercase')
        else:
            df_copy = standardize_column(df_copy, col, 'to_numeric')
    return df_copy

def standardize_column(df, column, method, **kwargs):
    """Apply specific standardization to a single column"""
    df_copy = df.copy()
    col_data = df_copy[column]

    if method == 'lowercase':
        df_copy[column] = col_data.str.lower()
    elif method == 'uppercase':
        df_copy[column] = col_data.str.upper()
    elif method == 'title':
        df_copy[column] = col_data.str.title()
    elif method == 'strip':
        df_copy[column] = col_data.str.strip()
    elif method == 'round':
        decimals = kwargs.get('decimals', 0)
        df_copy[column] = col_data.round(decimals)
    elif method == 'to_numeric':
        try:
            df_copy[column] = pd.to_numeric(col_data)
        except (ValueError, TypeError):
            pass
    elif method == 'num_to_words':
        df_copy[column] = df_copy[column].apply(lambda x: num2words(x) if pd.notnull(x) else x)
    elif method == 'words_to_num':
        def safe_w2n(x):
            try:
                if pd.isnull(x): return x
                return w2n.word_to_num(str(x))
            except:
                return x
        df_copy[column] = df_copy[column].apply(safe_w2n)
    # Add more methods as needed
    return df_copy

# -------- Cross-file operation (example merge) --------
def merge_datasets(dfs):
    if not dfs:
        return pd.DataFrame()

    # 1. Check if all datasets have identical columns (Case: Append/Union)
    first_cols = set(dfs[0].columns)
    all_same_cols = all(set(df.columns) == first_cols for df in dfs[1:])

    if all_same_cols:
        # Vertical stack + remove duplicates
        return pd.concat(dfs, ignore_index=True).drop_duplicates()
    
    # 2. Otherwise try to merge on common columns (Case: Join)
    merged_df = dfs[0]
    for df_next in dfs[1:]:
        # Find intersection of columns
        common_cols = list(set(merged_df.columns) & set(df_next.columns))
        
        if common_cols:
            # Relational merge (outer join to keep all data)
            merged_df = pd.merge(merged_df, df_next, on=common_cols, how='outer')
        else:
            # No shared keys, just append (will create NaNs)
            merged_df = pd.concat([merged_df, df_next], ignore_index=True)
            
    return merged_df
