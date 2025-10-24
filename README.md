# S&P 500 Stock Movement Prediction Project

## Project Overview

This project implements a comprehensive machine learning pipeline for predicting S&P 500 stock movement direction using sequential validation methodology. The project follows time-series best practices to avoid data leakage and provides realistic performance estimates for stock prediction models.

### Key Features

- **Sequential Validation**: Time-ordered splits respecting temporal dependencies
- **Multiple Models**: Logistic Regression, Random Forest, and Gradient Boosted Trees (XGBoost)
- **Comprehensive Feature Engineering**: Technical indicators, lag features, and international market indices
- **Detailed Evaluation**: Confusion matrices, ROC curves, and Precision-Recall curves
- **Model Persistence**: All trained models saved for future use

## Project Structure

```
milestone_2/
├── dataset_creation/           # Data preprocessing and feature engineering
│   ├── create_technical_features.py    # Technical indicators calculation
│   ├── add_lag_features.py             # Lag features (returns from past N days)
│   ├── add_international_indices.py    # International market indices
│   └── sp500_data.zip                  # Raw S&P 500 stock data
├── plot_creation/             # Visualization scripts
│   ├── confusion_matrix_analysis.py    # Confusion matrix generation
│   ├── precision_recall_analysis.py   # Precision-Recall curves
│   └── roc_curve_analysis.py          # ROC curve analysis
├── report/                    # Project documentation
│   └── S&P500 Tickets Prediction Report - Google Docs.pdf
├── saved_models/              # Trained model artifacts
│   ├── fold_1/ to fold_5/     # Models from each validation fold
│   ├── fold_final/            # Final models trained on all data
│   └── model_summary.txt      # Performance summary
├── sequential_splits_v1/      # Time-ordered data splits
│   ├── fold_X_train.csv       # Training data for each fold
│   ├── fold_X_validation.csv  # Validation data for each fold
│   ├── final_test.csv         # Final test set
│   └── splits_summary.csv     # Split information summary
├── sequential_validation.py   # Data splitting script
├── train_models.py           # Model training script
├── test_models.py            # Model testing script
└── README.md                 # This file
```

## Quick Start

### Prerequisites

- Python 3.8+
- Required libraries (see Installation section)

### Installation

1. **Clone or download the project**
2. **Install required libraries**:
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib yfinance tqdm
```

3. **Extract raw data**:
```bash
cd dataset_creation
unzip sp500_data.zip
cd ..
```

### Running the Complete Pipeline

1. **Create sequential validation splits**:
```bash
python sequential_validation.py
```

2. **Train models**:
```bash
python train_models.py
```

3. **Test models**:
```bash
python test_models.py
```

4. **Generate visualizations**:
```bash
python plot_creation/confusion_matrix_analysis.py
python plot_creation/roc_curve_analysis.py
python plot_creation/precision_recall_analysis.py
```

## Detailed Component Description

### 1. Dataset Creation (`dataset_creation/`)

#### `create_technical_features.py`
- **Purpose**: Calculates technical indicators for each stock
- **Features**: Price ratios, moving averages, volatility, momentum, RSI, Bollinger Bands, MACD, etc.
- **Input**: Raw stock CSV files from `sp500_data/`
- **Output**: `sp500_daily_technical_features.csv`

#### `add_lag_features.py`
- **Purpose**: Adds lag features (returns from past N days)
- **Features**: Lag returns (1, 2, 3, 4, 5, 10, 15, 20, 22 days), lag statistics
- **Input**: Technical features dataset
- **Output**: Enhanced dataset with lag features

#### `add_international_indices.py`
- **Purpose**: Incorporates international market indices as features
- **Indices**: Nikkei 225, Hang Seng, All Ordinaries, DAX, FTSE 100, NYSE Composite, DJIA
- **Features**: Returns, moving averages, volatility for each index
- **Input**: Dataset with lag features
- **Output**: Final dataset with international market features

### 2. Sequential Validation (`sequential_validation.py`)

- **Purpose**: Creates time-ordered data splits for realistic evaluation
- **Method**: Sequential validation with expanding training windows
- **Splits**: 5 folds + final test set
- **Output**: `sequential_splits_v1/` directory with train/validation/test sets

#### Split Details:
- **Fold 1**: Train (2018-2019), Validate (2019-2020)
- **Fold 2**: Train (2018-2020), Validate (2020-2021)
- **Fold 3**: Train (2018-2021), Validate (2021-2023)
- **Fold 4**: Train (2018-2023), Validate (2023-2024)
- **Fold 5**: Train (2018-2024), Validate (2024-2025)
- **Final Test**: 2025 data

### 3. Model Training (`train_models.py`)

- **Purpose**: Trains and evaluates multiple models using sequential validation
- **Models**: 
  - Logistic Regression (with feature scaling)
  - Random Forest (with class balancing)
  - Gradient Boosted Trees/XGBoost (with class balancing)
- **Features**: 112 engineered features per stock per day
- **Output**: Trained models saved in `saved_models/`

#### Model Configuration:
```python
# Logistic Regression
LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')

# Random Forest
RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)

# XGBoost
XGBClassifier(n_estimators=100, random_state=42, scale_pos_weight=1, n_jobs=-1, eval_metric='logloss')
```

### 4. Model Testing (`test_models.py`)

- **Purpose**: Loads trained models and evaluates them on test set
- **Evaluation**: All models across all folds + final models
- **Metrics**: Accuracy, Precision, Recall, F1-Score, AUC
- **Output**: `test_results.txt` with comprehensive results

### 5. Visualization (`plot_creation/`)

#### `confusion_matrix_analysis.py`
- **Purpose**: Generates confusion matrices for all models
- **Output**: Individual and comparison confusion matrix plots
- **Files**: `confusion_matrices/` directory

#### `roc_curve_analysis.py`
- **Purpose**: Creates ROC curves and AUC analysis
- **Output**: ROC curve plots and threshold analysis
- **Files**: `roc_curves/` directory

#### `precision_recall_analysis.py`
- **Purpose**: Generates Precision-Recall curves
- **Output**: PR curves and optimal threshold analysis
- **Files**: `precision_recall/` directory

## Dataset Information

### Data Sources
- **S&P 500 Stocks**: 501 companies with daily OHLCV data
- **Time Period**: 2018-2025 (training), 2025+ (testing)
- **International Indices**: 7 major global market indices

### Feature Categories
1. **Technical Indicators** (40+ features):
   - Price ratios (High/Low, Close/Open, etc.)
   - Moving averages (5-day, 20-day, 50-day)
   - Volume indicators
   - Volatility measures
   - Momentum indicators (RSI, MACD)
   - Bollinger Bands

2. **Lag Features** (20+ features):
   - Returns from past 1, 2, 3, 4, 5, 10, 15, 20, 22 days
   - Lag statistics (mean, std of returns)
   - Lag of technical indicators

3. **International Market Features** (35+ features):
   - Returns, moving averages, volatility for each index
   - Price-to-MA ratios for global markets

### Target Variable
- **Binary Classification**: 0 (Down movement), 1 (Up movement)
- **Definition**: Next day's return > 0
- **Class Distribution**: ~47.7% Down, ~52.3% Up (slightly imbalanced)

## Methodology

### Sequential Validation
- **Why**: Avoids data leakage in time series prediction
- **Method**: Expanding training windows with fixed validation periods
- **Advantage**: More realistic performance estimates than random CV

### Feature Engineering
- **Technical Analysis**: Standard financial indicators
- **Lag Features**: Captures momentum and mean reversion
- **Global Context**: International market influence

### Model Selection
- **Linear Model**: Logistic Regression (baseline)
- **Tree-based**: Random Forest (ensemble)
- **Gradient Boosting**: XGBoost (advanced ensemble)

## Results Summary

### Model Performance (Final Test Set)
| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|-------|----------|-----------|--------|----------|-----|
| Logistic Regression | 0.5070 | 0.5572 | 0.2786 | 0.3715 | 0.5263 |
| Random Forest | 0.4784 | 0.5135 | 0.0490 | 0.0895 | 0.5003 |
| Gradient Boosted Trees | 0.4884 | 0.5238 | 0.2377 | 0.3270 | 0.4904 |

### Key Findings
- **Logistic Regression** performs best overall
- **All models** show performance close to random chance (~50% accuracy)
- **Stock prediction** is inherently challenging due to market efficiency
- **Sequential validation** provides realistic performance estimates

## Usage Examples

### Training New Models
```bash
# Create data splits
python sequential_validation.py

# Train models
python train_models.py
```

### Testing Existing Models
```bash
# Test all saved models
python test_models.py

# Generate specific visualizations
python plot_creation/confusion_matrix_analysis.py
```

### Loading Saved Models
```python
import joblib

# Load a specific model
model = joblib.load('./saved_models/fold_final/Logistic_Regression/model.joblib')
scaler = joblib.load('./saved_models/fold_final/Logistic_Regression/scaler.joblib')

# Load metrics
metrics = joblib.load('./saved_models/fold_final/Logistic_Regression/metrics.joblib')
print(f"AUC: {metrics['auc']:.4f}")
```

## Dependencies

### Core Libraries
```python
pandas>=1.3.0          # Data manipulation
numpy>=1.21.0          # Numerical computing
scikit-learn>=1.0.0    # Machine learning
xgboost>=1.5.0         # Gradient boosting
```

### Visualization
```python
matplotlib>=3.5.0      # Plotting
seaborn>=0.11.0        # Statistical visualization
```

### Data Processing
```python
joblib>=1.1.0          # Model persistence
yfinance>=0.1.70       # Financial data
tqdm>=4.62.0           # Progress bars
```

### Installation Command
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib yfinance tqdm
```

## Configuration

### Model Parameters
Models can be customized by modifying parameters in `train_models.py`:

```python
# Logistic Regression
LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')

# Random Forest
RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)

# XGBoost
XGBClassifier(n_estimators=100, random_state=42, scale_pos_weight=1, n_jobs=-1, eval_metric='logloss')
```

### Data Splits
Sequential validation parameters can be modified in `sequential_validation.py`:
- `n_splits`: Number of validation folds (default: 5)
- `test_size`: Proportion for final test set (default: 0.2)

## File Descriptions

### Core Scripts
- `sequential_validation.py`: Creates time-ordered data splits
- `train_models.py`: Trains and saves models using sequential validation
- `test_models.py`: Loads and evaluates saved models

### Data Processing
- `dataset_creation/create_technical_features.py`: Technical indicators
- `dataset_creation/add_lag_features.py`: Lag features
- `dataset_creation/add_international_indices.py`: International indices

### Visualization
- `plot_creation/confusion_matrix_analysis.py`: Confusion matrices
- `plot_creation/roc_curve_analysis.py`: ROC curves
- `plot_creation/precision_recall_analysis.py`: Precision-Recall curves

### Output Files
- `saved_models/`: All trained models and metrics
- `sequential_splits_v1/`: Time-ordered data splits
- `test_results.txt`: Comprehensive test results
- `confusion_matrices/`: Confusion matrix plots
- `roc_curves/`: ROC curve plots
- `precision_recall/`: Precision-Recall plots

## Troubleshooting

### Common Issues

1. **Missing Dependencies**:
   ```bash
   pip install -r requirements.txt  # If available
   # Or install individually as shown above
   ```

2. **Data Not Found**:
   ```bash
   cd dataset_creation
   unzip sp500_data.zip
   cd ..
   ```

3. **Memory Issues**:
   - Reduce `n_estimators` in Random Forest/XGBoost
   - Use smaller batch sizes
   - Process data in chunks

4. **Model Loading Errors**:
   - Ensure models are trained first: `python train_models.py`
   - Check file paths in scripts

### Performance Tips
- Use `n_jobs=-1` for parallel processing
- Monitor memory usage with large datasets
- Consider feature selection for faster training

## References

- Sequential Validation methodology based on time series best practices
- Technical indicators from standard financial analysis
- Model evaluation following machine learning standards

## Contributing

To extend this project:
1. Add new features in `dataset_creation/`
2. Implement new models in `train_models.py`
3. Create additional visualizations in `plot_creation/`
4. Update this README with new components

## License

This project is for educational and research purposes. Please ensure compliance with data usage terms for financial data.

---

**Note**: Stock prediction is inherently difficult due to market efficiency. Results close to random chance (50% accuracy) are expected and normal for this type of problem.
