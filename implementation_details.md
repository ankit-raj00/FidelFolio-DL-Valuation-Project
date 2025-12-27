# Implementation Details

This document outlines the step-by-step logic and engineering decisions implemented in the FidelFolio Deep Learning pipeline.

## 1. Data Cleaning & Loading
**Source**: `src/data/loader.py` & `src/data/preprocessing.py`

1.  **Column Sanitization**: All column names are stripped of whitespace and special characters to ensure safe Python attribute access.
2.  **Numeric Conversion**: The raw dataset contains numbers formatted as strings (e.g., "1,234.56"). We systematically:
    *   Remove commas.
    *   Coerce errors to NaN to identify bad data.
    *   Exclude the ID column (`Company`) from this conversion.

## 2. Feature Engineering
**Source**: `src/features/engineering.py`

To capture growth trends rather than just absolute values, we calculate **Year-over-Year (YoY) Differences**:
*   For every numerical feature (e.g., `Revenue`), we compute: `Value(Year) - Value(Year-1)`.
*   These "delta" features are vital for stationary time-series modeling.
*   **Result**: The dataset size doubles in feature count (Originals + Diffs).

## 3. Advanced Preprocessing
**Source**: `src/data/preprocessing.py`

This step handles the "messiness" of real-world financial data:

### A. Missing Value Imputation (KNN)
We use `KNNImputer` (k=5) to fill missing values.
*   *Why?* Financial ratios are often correlated. If a company misses one profit metric but has others, its "neighbors" in feature space provide a better estimate than a simple mean.

### B. Outlier Capping (Z-Score)
Financial data often has extreme outliers (e.g., a startup growing 1000%).
*   We calculate Z-scores for each column.
*   Values beyond **3 standard deviations** are capped (clipped) to the ±3σ boundary.
*   This prevents gradients from exploding during neural network training.

### C. Feature Scaling (RobustScaler)
We use `RobustScaler` instead of `StandardScaler`.
*   *Why?* It scales data based on the median and IQR (Inter-Quartile Range). It is more robust to outliers that might still exist after capping.

## 4. Dimensionality Reduction (PCA)
**Source**: `src/features/engineering.py`

To prevent the "Curse of Dimensionality" and speed up training:
*   We apply **PCA (Principal Component Analysis)**.
*   We keep enough components to explain **95% of the variance**.
*   This compresses correlated financial metrics into a smaller set of orthogonal features.

## 5. Sequence Generation (Expanding Window)
**Source**: `src/features/sequences.py`

For LSTM/RNN models, we cannot just use single rows. We need history.
*   **Grouping**: We group data by `Company ID` and sort by `Year`.
*   **Sliding Window**: For a target at year `T`, we generate a sequence of features from `[T-History ... T-1]`.
*   **Expanding Window Validation**:
    *   **Fold 1**: Train on data <= 2018, Test on 2019.
    *   **Fold 2**: Train on data <= 2019 (includes old test set), Test on 2020.
    *   **Fold 3**: Train on data <= 2020, Test on 2021.
## 6. Company Embeddings
**Source**: `src/models/*.py`

The `Company` column is categorical, but One-Hot Encoding would be inefficient with hundreds of unique companies (high cardinality). Instead, we use **Entity Embeddings**:
*   **Encoding**: Each company is mapped to a unique integer ID.
*   **Embedding Layer**: The neural network learns a dense vector representation (size 16) for each company during training.
*   **Fusion**: This static "company vector" captures intrinsic properties of the firm (e.g., sector, management quality) and is concatenated with the time-series features before being fed into the dense/LSTM layers.
