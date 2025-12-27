# FidelFolio DL Valuation Project

Modular Deep Learning pipeline for predicting market capitalization growth of listed companies using fundamental financial indicators.

## Project Structure

```text
FidelFolio_Project/
├── config/
│   └── config.yaml           # Hyperparameters & settings
├── data/
│   └── FidelFolio_Dataset.csv # [REQUIRED] Place your dataset here
├── experiments/              # Original Jupyter Notebooks
├── src/                      # Source code
│   ├── data/                 # Loading & Preprocessing
│   ├── features/             # Feature Engineering & Sequences
│   ├── models/               # MLP, LSTM, Encoder-Decoder Architectures
│   └── utils/                # Utilities
├── main.py                   # CLI Entry Point
├── pyproject.toml            # Build configuration
└── setup.py                  # Setup script
```

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -e .
   ```

2. **Data Setup**:
   Place your `FidelFolio_Dataset.csv` file into the `data/` directory.

## Usage

Run the training pipeline using `main.py`:

```bash
# Run with default MLP model
python main.py

# Run with LSTM model
python main.py --model lstm

# Run with Encoder-Decoder model
python main.py --model encoder_decoder

# Use a custom config file
python main.py --config config/my_custom_config.yaml
```

## Configuration

Modify `config/config.yaml` to adjust:
- `preprocessing`: Imputation neighbors, outlier capping thresholds, PCA parameters.
- `models`: Layer sizes, dropout rates, embedding dimensions.
- `training`: Epochs, batch size, learning rate.

## Models implemented

- **MLP**: Feed-forward network with company embeddings.
- **LSTM**: Recurrent neural network handling sequential financial history.
- **Encoder-Decoder**: LSTM encoder to capture context, Dense decoder for prediction.

## Pipeline Flow

```mermaid
graph TD
    A[Start: main.py] --> B{Load Config}
    B --> C[Load Config.yaml]
    C --> D[Load Data]
    D --> E[Data Cleaning]
    E --> F[Feature Engineering: YoY Diffs]
    F --> G[Preprocessing (Impute, Scale, Cap)]
    G --> H{PCA Enabled?}
    H -- Yes --> I[Apply PCA]
    H -- No --> J[Skip PCA]
    I --> K[Encode Company IDs]
    J --> K
    K --> L[Start Loop (Expanding Window)]
    L --> M[Generate Sequences]
    M --> N[Train Model (MLP/LSTM/Enc-Dec)]
    N --> O[Predict & Evaluate]
    O --> P{Next Year?}
    P -- Yes --> L
    P -- No --> Q[Finish]
```
