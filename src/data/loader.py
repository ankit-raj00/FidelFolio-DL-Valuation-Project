import pandas as pd
import numpy as np
import os

def load_data(file_path):
    """
    Load data from CSV file.
    
    Args:
        file_path (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}. Please ensure the file exists.")
    
    df = pd.read_csv(file_path)
    print(f"Loaded data with shape: {df.shape}")
    return df

def clean_column_names(df):
    """
    Strip whitespace from column names.
    """
    df.columns = df.columns.str.strip()
    return df

def convert_to_numeric(df, ignore_cols=None):
    """
    Convert potential numeric columns to numeric type, handling commas.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        ignore_cols (list, optional): List of columns to skip. Defaults to None.
        
    Returns:
        pd.DataFrame: DataFrame with converted columns.
    """
    if ignore_cols is None:
        ignore_cols = []
    
    # Identify object columns that are not in the ignore list
    potential_cols = df.select_dtypes(include='object').columns
    cols_to_convert = [c for c in potential_cols if c not in ignore_cols]
    
    converted_count = 0
    for col in cols_to_convert:
        try:
            # Convert to string, remove commas, then to numeric
            col_str = df[col].astype(str).str.replace(',', '', regex=False)
            converted_col = pd.to_numeric(col_str, errors='coerce')
            
            # Check if majority are non-null to verify it's a valid numeric col
            if pd.api.types.is_numeric_dtype(converted_col) and converted_col.notnull().sum() > 0:
                df[col] = converted_col
                converted_count += 1
        except Exception as e:
            print(f"Warning: Could not convert column {col}: {e}")
            
    print(f"Converted {converted_count} columns to numeric.")
    return df
