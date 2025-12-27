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
# Run the pipeline (uses config/config.yaml by default)
python main.py

# Run with a specific configuration file (e.g. for testing)
python main.py --config config/test_config.yaml
```

To switch models (MLP / LSTM / Encoder-Decoder), edit `model_type` in `config/config.yaml`.

## Configuration

Modify `config/config.yaml` to adjust:
- `preprocessing`: Imputation neighbors, outlier capping thresholds, PCA parameters.
- `models`: Layer sizes, dropout rates, embedding dimensions.
- `training`: Epochs, batch size, learning rate.

## Models Implemented

### 1. MLP (Multi-Layer Perceptron)
A traditional feed-forward network that flattens time-series data into a single vector, combined with learned company embeddings.

```mermaid
graph TD
    subgraph Inputs
    A["Sequence Input (Time x Feats)"] --> B[Flatten]
    C["Company ID"] --> D[Embedding]
    D --> E[Flatten Embedding]
    end
    
    B --> F[Concatenate]
    E --> F
    
    subgraph MLP_Layers
    F --> G["Dense Layer 1 (ReLU)"]
    G --> H[Dropout]
    H --> I["Dense Layer 2 (ReLU)"]
    I --> J[Dropout]
    end
    
    J --> K["Output (Regression)"]
```

### 2. LSTM (Long Short-Term Memory)
A Recurrent Neural Network (RNN) designed to capture temporal dependencies in financial data.

```mermaid
graph TD
    subgraph Inputs
    A["Sequence Input"] --> B[Masking]
    C["Company ID"] --> D[Embedding]
    D --> E[Flatten Embedding]
    end
    
    subgraph LSTM_Stack
    B --> F["LSTM Layer 1 (return_seq=True)"]
    F --> G[Dropout]
    G --> H["LSTM Layer 2 (return_seq=False)"]
    H --> I[Dropout]
    end
    
    I --> J[Concatenate]
    E --> J
    
    subgraph Prediction
    J --> K["Dense Layer (ReLU)"]
    K --> L[Dropout]
    L --> M["Output"]
    end
```

### 3. Encoder-Decoder
Uses an LSTM as an encoder to compress the time-series context into a hidden state, which is then passed to a Dense decoder for prediction.

```mermaid
graph TD
    subgraph Encoder
    A["Sequence Input"] --> B[Masking]
    B --> C["LSTM Encoder"]
    C -- "Extract Context (State H)" --> D[Context Vector]
    end
    
    subgraph Context_Fusion
    E["Company ID"] --> F[Embedding]
    F --> G[Flatten]
    D --> H[Concatenate]
    G --> H
    end
    
    subgraph Decoder
    H --> I["Dense Decoder (ReLU)"]
    I --> J[Dropout]
    J --> K["Output"]
    end
```

## Pipeline Flow

```mermaid
graph TD
    A["Start: main.py"] --> B{"Load Config"}
    B --> C["Load Config.yaml"]
    C --> D["Load Data"]
    D --> E["Data Cleaning"]
    E --> F["Feature Engineering: YoY Diffs"]
    F --> G["Preprocessing (Impute, Scale, Cap)"]
    G --> H{"PCA Enabled?"}
    H -- Yes --> I["Apply PCA"]
    H -- No --> J["Skip PCA"]
    I --> K["Encode Company IDs"]
    J --> K
    K --> L["Start Loop (Expanding Window)"]
    L --> M["Generate Sequences"]
    M --> N["Train Model (MLP/LSTM/Enc-Dec)"]
    N --> O["Predict & Evaluate"]
    O --> P{"Next Year?"}
    P -- Yes --> L
    P -- No --> Q["Finish"]
```
