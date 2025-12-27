import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

def create_sequences(df, feature_cols, target_cols, id_col_encoded, year_col, min_history_years=2):
    """
    Create sequences for time-series models (LSTM/RNN).
    
    Args:
        df (pd.DataFrame): Preprocessed DataFrame.
        feature_cols (list): List of feature column names (e.g., PCA comps).
        target_cols (list): List of target column names.
        id_col_encoded (str): Name of the encoded ID column.
        year_col (str): Name of the year column.
        min_history_years (int): Minimum sequence length needed.
        
    Returns:
        tuple: (X_padded, X_company_ids, y_targets, sequence_years, max_sequence_len)
    """
    all_sequences = []
    all_company_ids = []
    all_targets = []
    all_years = [] # New list to store years
    
    max_len = 0
    skipped_nan_input = 0
    skipped_nan_target = 0
    
    # Ensure simplified data view
    # Group by Encoded Company ID
    company_groups = df.groupby(id_col_encoded)
    
    for company_id, group in company_groups:
        group = group.sort_values(year_col)
        features = group[feature_cols].values
        targets = group[target_cols].values
        years = group[year_col].values # Capture years
        
        # We need at least 'min_history_years' of history to make a prediction
        # If min_history_years=2, we can start predicting from index 2 (using 0,1 as history)
        # Actually in expanding window, we usually take history [0...i-1] to predict target [i]
        
        for i in range(min_history_years, len(group)):
            # Sequence: History up to i-1
            current_sequence = features[:i, :]
            # Target: Value at i
            current_target = targets[i, :]
            # Year: Value at i
            current_year = years[i]
            
            # Check for NaNs
            if np.isnan(current_sequence).any():
                skipped_nan_input += 1
                continue
                
            if np.isnan(current_target).any():
                skipped_nan_target += 1
                continue
                
            all_sequences.append(current_sequence)
            all_company_ids.append(company_id)
            all_targets.append(current_target)
            all_years.append(current_year)
            
            if len(current_sequence) > max_len:
                max_len = len(current_sequence)
                
    if skipped_nan_input > 0:
        print(f"Warning: Skipped {skipped_nan_input} sequences due to NaNs in inputs.")
    if skipped_nan_target > 0:
        print(f"Warning: Skipped {skipped_nan_target} sequences due to NaNs in targets.")
        
    if not all_sequences:
        return np.array([]), np.array([]), np.array([]), np.array([]), 0
        
    # Pad sequences
    # Padding 'pre' matches the logic that recent data is at the end
    X_padded = pad_sequences(all_sequences, maxlen=max_len, dtype='float32', 
                             padding='pre', truncating='pre', value=0.0)
    
    X_company_ids = np.array(all_company_ids, dtype='int32')
    y_targets = np.array(all_targets, dtype='float32')
    sequence_years = np.array(all_years, dtype='int32')
    
    return X_padded, X_company_ids, y_targets, sequence_years, max_len
