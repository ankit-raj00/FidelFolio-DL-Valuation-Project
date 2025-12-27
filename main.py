import argparse
import sys
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import json # Added json import

# Import local modules
# Ensure src is in path if running directly
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils import load_config
from src.data import load_data, clean_column_names, convert_to_numeric, impute_data, scale_features, cap_outliers
from src.features import create_yoy_diffs, apply_pca, create_sequences
from src.models import build_mlp_model, build_lstm_model, build_encoder_decoder_model
from src.utils.logger import setup_logger # Added setup_logger import

# Initialize Logger (Default)
logger = setup_logger()

def main():
    parser = argparse.ArgumentParser(description="FidelFolio ML Pipeline")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    args = parser.parse_args()

    # Initialize a default logger first to handle potential config loading errors
    global logger
    logger = setup_logger()

    # 1. Load Configuration
    logger.info(f"Loading configuration from {args.config}")
    try:
        config = load_config(args.config)
        
        # Re-initialize logger with model name from config
        model_type = config['training'].get('model_type', 'mlp')
        logger = setup_logger(model_name=model_type)
        logger.info(f"Loaded configuration from {args.config}. Logger updated for model: {model_type}")
        
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return

    # 2. Load Data
    data_path = config['data']['raw_path']
    try:
        df = load_data(data_path)
    except FileNotFoundError:
        logger.error(f"Data file not found at {data_path}. Please place 'FidelFolio_Dataset.csv' in the data directory.") # Changed print to logger.error
        return

    # 3. Initial Preprocessing
    logger.info("--- 1. Cleaning & Numeric Conversion ---") # Changed print to logger.info
    df = clean_column_names(df)
    df = convert_to_numeric(df, ignore_cols=[config['data']['id_col']])

    # 4. Feature Engineering (YoY Diffs)
    logger.info("--- 2. Feature Engineering (YoY Diffs) ---") # Changed print to logger.info
    id_col = config['data']['id_col']
    year_col = config['data']['year_col']
    target_cols = config['data']['target_cols']
    
    # Exclude targets from diff calculation
    df, new_features = create_yoy_diffs(df, id_col, year_col, target_cols=target_cols)

    # 5. Advanced Preprocessing (Imputation, Scaling, Capping)
    logger.info("--- 3. Imputation, Scaling, Capping ---") # Changed print to logger.info
    # Identify columns to process: Original Numeric + Engineered
    numeric_cols = df.select_dtypes(include=np.number).columns
    cols_to_process = [c for c in numeric_cols if c not in target_cols + [year_col]]
    
    # Impute
    df = impute_data(df, cols_to_process, n_neighbors=config['preprocessing']['knn_imputer']['n_neighbors'])
    
    # Scale
    df, scaler = scale_features(df, cols_to_process)
    
    # Cap
    df = cap_outliers(df, cols_to_process, z_threshold=config['preprocessing']['outlier_capping']['z_score_threshold'])

    # 6. PCA (Optional)
    feature_cols_for_model = cols_to_process # Default
    if config['preprocessing']['pca']['enabled']:
        logger.info("--- 4. PCA ---") # Changed print to logger.info
        df, pc_cols, pca_model = apply_pca(df, cols_to_process, variance_ratio=config['preprocessing']['pca']['variance_ratio'])
        if pc_cols:
            feature_cols_for_model = pc_cols

    # 7. Encode ID
    logger.info("--- 5. Encoding Company IDs ---") # Changed print to logger.info
    le = LabelEncoder()
    df['Company_ID_Encoded'] = le.fit_transform(df[id_col])
    num_companies = df['Company_ID_Encoded'].nunique()
    
    # 8. Scale Targets (Independently)
    logger.info("--- 6. Scaling Targets ---") # Changed print to logger.info
    target_scalers = {}
    for t in target_cols:
        _, t_scaler = scale_features(df, [t]) # Scale in place for df, but get scaler
        target_scalers[t] = t_scaler

    # 9. Training Loop (Expanding Window)
    model_type = config['training'].get('model_type', 'mlp')
    logger.info(f"--- 7. Starting Training ({model_type.upper()}) ---")
    unique_years = sorted(df[year_col].unique())
    min_history = config['preprocessing']['sequence_generation']['min_history_years']
    
    # Create metrics/results directory
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    metrics_file = os.path.join(results_dir, f"metrics_{model_type}.json")

    # Load existing metrics to append to
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            all_metrics = json.load(f)
    else:
        all_metrics = []
    
    current_run_metrics = [] # Track metrics for just this execution run
    
    if len(unique_years) <= min_history:
        logger.warning("Not enough history for training.") # Changed print to logger.warning
        return

    metrics = []

    for i in range(min_history, len(unique_years)):
        test_year = unique_years[i]
        train_end_year = unique_years[i-1]
        
        logger.info(f"\n=== Fold: Test Year {test_year} (Train <= {train_end_year}) ===")
        
        # Split Data
        train_df = df[df[year_col] <= train_end_year]
        # For test, we look for companies that have data in test_year AND sufficient history
        # (This logic is simplified for the main script, real logic is in sequences.py mostly)
        
        # Prepare Sequences
        # Note: sequences.py handles the complex logic of looking back 'min_history'
        # Prepare Sequences
        # Note: sequences.py handles the complex logic of looking back 'min_history'
        X_train_seq, X_train_cid, y_train_all, _, max_len_train = create_sequences(
            train_df, feature_cols_for_model, target_cols, 'Company_ID_Encoded', year_col, min_history_years=min_history
        )
        
        if len(X_train_seq) == 0:
            logger.warning("No training data generated.")
            continue
            
        logger.info(f"Training Samples: {len(X_train_seq)}")
        
        # Prepare Test Data (Candidate companies in test_year)
        test_candidate_df = df[df[year_col] <= test_year]
        X_test_seq_all, X_test_cid_all, y_test_all_all, test_years_all, _ = create_sequences(
            test_candidate_df, feature_cols_for_model, target_cols, 'Company_ID_Encoded', year_col, min_history_years=min_history
        )
        
        # Filter for actual test year
        test_mask = (test_years_all == test_year)
        X_test_seq = X_test_seq_all[test_mask]
        X_test_cid = X_test_cid_all[test_mask]
        y_test_all = y_test_all_all[test_mask]

        if len(X_test_seq) == 0:
            logger.warning(f"No test samples found for year {test_year}.") # Changed print to logger.warning
            continue
            
        # Ensure test sequences are padded to same max_len as train
        if X_test_seq.shape[1] < max_len_train:
             X_test_seq = tf.keras.preprocessing.sequence.pad_sequences(X_test_seq, maxlen=max_len_train, dtype='float32', padding='pre', truncating='pre', value=0.0)
        elif X_test_seq.shape[1] > max_len_train:
             # This strictly shouldn't happen if train set includes all history, but safe to truncate
             X_test_seq = X_test_seq[:, -max_len_train:, :]

        logger.info(f"Test Samples: {len(X_test_seq)}") # Changed print to logger.info

        # Train per target
        for t_idx, target_name in enumerate(target_cols):
            logger.info(f"  Training for {target_name}...") # Changed print to logger.info
            y_train = y_train_all[:, t_idx]
            y_test = y_test_all[:, t_idx]
            
            # Build Model
            if model_type == 'mlp':
                model = build_mlp_model(max_len_train, len(feature_cols_for_model), num_companies, 1, config)
            elif model_type == 'lstm':
                model = build_lstm_model(max_len_train, len(feature_cols_for_model), num_companies, 1, config)
            elif model_type == 'encoder_decoder':
                model = build_encoder_decoder_model(max_len_train, len(feature_cols_for_model), num_companies, 1, config)
            
            # Callbacks
            es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=config['training']['early_stopping_patience'], restore_best_weights=True)
            
            model.fit(
                [X_train_seq, X_train_cid], y_train,
                validation_split=config['training']['validation_split'],
                epochs=config['training']['epochs'],
                batch_size=config['training']['batch_size'],
                verbose=0,
                callbacks=[es]
            )
            
            # Basic Evaluation
            # Predict
            y_pred_scaled = model.predict([X_test_seq, X_test_cid], verbose=0)
            
            # Inverse Transform
            # We need the specific scaler for this target
            if target_name in target_scalers:
                scaler = target_scalers[target_name]
                # RobustScaler expects 2D array, we have (N, 1) or (N,)
                y_pred_original = scaler.inverse_transform(y_pred_scaled).flatten()
                y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
                
                # Calculate RMSE
                rmse_original = np.sqrt(np.mean((y_pred_original - y_test_original) ** 2))
                logger.info(f"  Result -> Test RMSE (Original Scale): {rmse_original:.4f}")

                # Save Metrics to List
                metric_entry = {
                    "model": model_type,
                    "test_year": int(test_year),
                    "target": target_name,
                    "rmse": float(rmse_original),
                    "timestamp": pd.Timestamp.now().isoformat()
                }
                all_metrics.append(metric_entry)
                current_run_metrics.append(metric_entry)

            else:
                 logger.warning("  Warning: Scaler not found for inverse transform.")

    # Save Detailed Metrics to JSON
    with open(metrics_file, 'w') as f:
        json.dump(all_metrics, f, indent=4)
        logger.info(f"Detailed metrics saved to {metrics_file}")

    # Calculate and Save Overall Summary for Context
    if current_run_metrics:
        logger.info("--- Overall Performance Summary (Current Run) ---")
        summary_df = pd.DataFrame(current_run_metrics)
        avg_rmse = summary_df.groupby('target')['rmse'].mean().to_dict()
        
        summary_file = os.path.join(results_dir, f"summary_{model_type}.json")
        
        # Load existing summary if exists or create new
        if os.path.exists(summary_file):
             with open(summary_file, 'r') as f:
                all_summaries = json.load(f)
        else:
             all_summaries = []

        summary_entry = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "model": model_type,
            "average_rmse_per_target": avg_rmse,
            "overall_mean_rmse": summary_df['rmse'].mean()
        }
        all_summaries.append(summary_entry)
        
        with open(summary_file, 'w') as f:
            json.dump(all_summaries, f, indent=4)
            
        for target, score in avg_rmse.items():
            logger.info(f"  Target: {target} | Mean RMSE: {score:.4f}")
            
        logger.info(f"Overall Mean RMSE: {summary_entry['overall_mean_rmse']:.4f}")
        logger.info(f"Summary metrics saved to {summary_file}")

    logger.info("Pipeline finished successfully.")

if __name__ == "__main__":
    main()
