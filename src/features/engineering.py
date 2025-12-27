import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

def create_yoy_diffs(df, id_col, year_col, target_cols=None, ignore_cols=None):
    """
    Create Year-over-Year absolute differences for numeric features.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        id_col (str): Column identifying the entity (e.g., Company).
        year_col (str): Column identifying the time (e.g., Year).
        target_cols (list, optional): List of target columns to exclude.
        ignore_cols (list, optional): Additional columns to exclude.
        
    Returns:
        pd.DataFrame: DataFrame with new diff features.
        list: List of names of created features.
    """
    if target_cols is None: target_cols = []
    if ignore_cols is None: ignore_cols = []
    
    # Identify numeric columns to process
    numeric_cols = df.select_dtypes(include=np.number).columns
    cols_to_process = [
        c for c in numeric_cols 
        if c not in target_cols + ignore_cols + [id_col, year_col]
        and not c.endswith('_Diff') # Avoid diffing diffs if run multiple times
    ]
    
    print(f"Calculating YoY differences for {len(cols_to_process)} features...")
    
    # Sort for correct shifting
    df_sorted = df.sort_values(by=[id_col, year_col]).copy()
    
    new_features = []
    for col in cols_to_process:
        diff_col_name = f'FE_{col}_Diff'
        # Group by company and take diff
        df_sorted[diff_col_name] = df_sorted.groupby(id_col)[col].diff().astype(np.float32)
        new_features.append(diff_col_name)
        
    print(f"Created {len(new_features)} YoY difference features.")
    return df_sorted, new_features

def apply_pca(df, feature_cols, variance_ratio=0.95):
    """
    Apply PCA to reduce dimensionality.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        feature_cols (list): List of columns to use for PCA.
        variance_ratio (float): Desired explained variance ratio.
        
    Returns:
        pd.DataFrame: DataFrame with PCA components added.
        list: List of PCA component column names.
        PCA: Fitted PCA object.
    """
    if not feature_cols:
        print("No features provided for PCA.")
        return df, [], None
        
    print(f"Applying PCA (retaining {variance_ratio*100:.0f}% variance)...")
    
    X = df[feature_cols].values
    pca = PCA(n_components=variance_ratio, svd_solver='full')
    X_pca = pca.fit_transform(X)
    
    n_components = pca.n_components_
    print(f"PCA reduced features from {len(feature_cols)} to {n_components}.")
    
    pc_cols = [f'PC_{i+1}' for i in range(n_components)]
    df_pca = pd.DataFrame(X_pca, columns=pc_cols, index=df.index)
    
    # Concatenate PCA columns back (ensuring index alignment)
    df_result = pd.concat([df, df_pca], axis=1)
    
    return df_result, pc_cols, pca
