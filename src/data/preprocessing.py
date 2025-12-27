import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler

def impute_data(df, cols_to_impute, n_neighbors=5):
    """
    Perform KNN Imputation on specified columns.
    """
    if not cols_to_impute:
        print("No columns to impute.")
        return df

    data_to_impute = df[cols_to_impute].copy()
    
    # Check if impute is needed
    if not data_to_impute.isnull().any().any():
        print("No missing values found for imputation.")
        return df

    print(f"Imputing {len(cols_to_impute)} columns using KNN (k={n_neighbors})...")
    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_array = imputer.fit_transform(data_to_impute)
    
    df_imputed = df.copy()
    df_imputed[cols_to_impute] = imputed_array
    return df_imputed

def scale_features(df, cols_to_scale):
    """
    Apply RobustScaler to specified columns.
    Returns the dataframe with scaled columns and the scaler object.
    """
    if not cols_to_scale:
        return df, None
        
    print(f"Scaling {len(cols_to_scale)} columns using RobustScaler...")
    scaler = RobustScaler()
    df_scaled = df.copy()
    df_scaled[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    
    return df_scaled, scaler

def cap_outliers(df, cols_to_check, z_threshold=3.0):
    """
    Cap outliers based on Z-score threshold.
    """
    if not cols_to_check:
        return df
    
    print(f"Capping outliers (Z-score > {z_threshold})...")
    df_capped = df.copy()
    capped_count = 0
    
    for col in cols_to_check:
        col_data = df_capped[col]
        if col_data.dropna().empty:
            continue
            
        mean = col_data.mean()
        std = col_data.std()
        
        if std > 1e-6: # Avoid division by zero or constant columns
            lower = mean - z_threshold * std
            upper = mean + z_threshold * std
            
            # Check if any values are outside bounds
            if (col_data < lower).any() or (col_data > upper).any():
                df_capped[col] = col_data.clip(lower, upper)
                capped_count += 1
                
    print(f"Capped outliers in {capped_count} columns.")
    return df_capped
