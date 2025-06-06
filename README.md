# Retail Sales Forecasting

This project implements an advanced sales forecasting solution for retail stores using a combination of traditional time series methods and modern machine learning approaches.

## Project Structure

```
.
├── data/                  # Data directory
│   ├── train.csv         # Training data
│   ├── test.csv          # Test data
│   ├── store.csv         # Store metadata
│   └── sample_submission.csv  # Submission format
├── src/                  # Source code
│   ├── data_processing.py    # Data preprocessing utilities
│   ├── feature_engineering.py # Feature engineering pipeline
│   ├── models.py         # Model implementations
│   └── utils.py          # Utility functions
├── notebooks/            # Jupyter notebooks for analysis
├── requirements.txt      # Project dependencies
└── main.py              # Main execution script
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Features

- Advanced feature engineering including:
  - Temporal features (day of week, month, holidays)
  - Rolling window statistics
  - Store and item-level aggregations
  - Lag features
- Multiple modeling approaches:
  - LightGBM/CatBoost for tabular data
  - Prophet for time series components
  - LSTM/Transformer for sequence modeling
- Anomaly detection and handling
- Multi-store correlation analysis
- Confidence interval estimation

## Usage

1. Run the complete pipeline:
```bash
python main.py
```

2. For individual components:
```bash
python main.py --mode train    # Training only
python main.py --mode predict  # Prediction only
```

## Model Performance

The solution combines multiple models to achieve robust predictions:
- LightGBM for capturing complex feature interactions
- Prophet for handling seasonality and trends
- LSTM for capturing long-term dependencies

## License

MIT License 