# Demand Forecasting - Refactored Workflow

## Overview

This project has been refactored into **two separate, modular Jupyter Notebooks** for better organization, scalability, and maintainability:

1. **`cleaning.ipynb`** - Data Preprocessing & Export
2. **`modeling.ipynb`** - Modeling & Evaluation

## Architecture

```
┌─────────────────────────────┐
│  Raw Data (9 CSV files)     │
│  from Google Drive          │
└──────────────┬──────────────┘
               │
               ▼
        ┌─────────────────┐
        │ cleaning.ipynb  │
        │                 │
        │ • Load data     │
        │ • Clean         │
        │ • Merge         │
        │ • Engineer      │
        │ • Split (80-20) │
        │ • Export data   │
        └────────┬────────┘
                 │
                 ▼
    ┌──────────────────────────┐
    │ Preprocessed Data Files  │
    │ (parquet format)         │
    │                          │
    │ • X_train.parquet        │
    │ • X_test.parquet         │
    │ • y_train.parquet        │
    │ • y_test.parquet         │
    │ • feature_names.pkl      │
    │ • label_encoders.pkl     │
    └───────────┬──────────────┘
                │
                ▼
        ┌──────────────────┐
        │ modeling.ipynb   │
        │                  │
        │ • Load data      │
        │ • Train model    │
        │ • Evaluate       │
        │ • Analyze        │
        │ • Export model   │
        └────────┬─────────┘
                 │
                 ▼
    ┌──────────────────────────┐
    │   Trained Model Files    │
    │                          │
    │ • model.pkl (.pkl)       │
    │ • model_metadata.pkl     │
    │ • Analysis plots (.png)  │
    └──────────────────────────┘
```

## Workflow

### Step 1: Data Preprocessing (cleaning.ipynb)

This notebook handles the entire data pipeline:

**Inputs:**
- 9 raw CSV files from Google Drive (sales, online, markdowns, etc.)

**Processing:**
1. Load and optimize memory usage
2. Clean each dataset (remove non-positive values, outliers)
3. Merge datasets intelligently:
   - Combine sales + online transactions
   - Add store information (area, division, format, city)
   - Add product catalog (department, class, subclass, type)
4. Feature Engineering:
   - **Temporal features**: day, month, year, dayofweek, week, quarter, is_weekend
   - **Cyclical features**: month_sin, month_cos
   - **Holiday features**: Russian holidays indicator
   - **Lag features**: 7, 14, 30-day rolling averages
5. Encode categorical variables using LabelEncoder
6. **Temporal train-test split**: 80% historical data for training, 20% recent data for testing (prevents data leakage)

**Outputs:**
- `preprocessed_data/X_train.parquet` - Training features (80% by date)
- `preprocessed_data/X_test.parquet` - Testing features (20% recent dates)
- `preprocessed_data/y_train.parquet` - Training target (log-transformed quantity)
- `preprocessed_data/y_test.parquet` - Testing target (log-transformed quantity)
- `preprocessed_data/feature_names.pkl` - Feature column names
- `preprocessed_data/label_encoders.pkl` - Category encoders for future predictions

### Step 2: Model Training & Evaluation (modeling.ipynb)

This notebook builds and evaluates the demand forecasting model:

**Inputs:**
- All preprocessed data files from Step 1

**Processing:**
1. Load preprocessed train/test data
2. Train LightGBM model with optimized hyperparameters:
   - 300 estimators
   - Learning rate: 0.05
   - Num leaves: 63
   - Early stopping enabled
3. Make predictions on both train and test sets
4. Evaluate performance:
   - **RMSLE** (Root Mean Squared Logarithmic Error) - main metric for skewed data
   - **MAE** (Mean Absolute Error) in original quantity units
   - **R² Score** - variance explained
5. Analyze feature importances (top 15 features identified)
6. Create visualizations:
   - Feature importance plot
   - Residuals analysis (4 subplots)
7. Serialize model to pickle file for future predictions

**Outputs:**
- `trained_models/demand_forecasting_model.pkl` - Ready-to-use model
- `trained_models/model_metadata.pkl` - Model info, features, metrics, encoders
- `feature_importance.png` - Top features visualization
- `residuals_analysis.png` - Error analysis plots

## Key Improvements

### 1. **Logical Separation**
- ✓ Preprocessing and modeling are completely decoupled
- ✓ Easy to run only what you need
- ✓ Better for collaboration (different team members can own each notebook)

### 2. **Data Consistency**
- ✓ Train/test split happens only once in `cleaning.ipynb`
- ✓ Model in `modeling.ipynb` always evaluates on the exact same test set
- ✓ Eliminates data leakage from random re-splits

### 3. **State Preservation**
- ✓ Preprocessed data saved in **Parquet format** (preserves dtypes, faster I/O)
- ✓ Feature names explicitly saved for reference
- ✓ Label encoders saved for consistent categorical encoding on new data
- ✓ Model metadata saved alongside the model

### 4. **Efficiency**
- ✓ No need to re-run expensive preprocessing when tweaking model
- ✓ Preprocessed data loads quickly (<1 second)
- ✓ Focus on model tuning without data pipeline overhead

## How to Use

### Running the Full Pipeline

```bash
# Step 1: Run data preprocessing
jupyter notebook cleaning.ipynb
# Execute all cells - takes 5-15 minutes depending on data size

# Step 2: Run model training
jupyter notebook modeling.ipynb
# Execute all cells - takes 2-5 minutes
```

### Making Predictions on New Data

```python
import pickle
import pandas as pd
import numpy as np

# Load the trained model
with open('trained_models/demand_forecasting_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load metadata
with open('trained_models/model_metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

# Prepare your new data (same features, same order)
# new_data must be a DataFrame with columns: metadata['feature_names']

# Make predictions (returns log-scale predictions)
y_pred_log = model.predict(new_data[metadata['feature_names']])

# Transform back to original quantity scale
y_pred_original = np.expm1(np.maximum(y_pred_log, 0))

print(f"Predicted demand: {y_pred_original}")
```

### Re-training with New Data

1. Add new data files to the data loading section in `cleaning.ipynb`
2. Re-run `cleaning.ipynb` entirely (generates new preprocessed files)
3. Run `modeling.ipynb` to train on the new data
4. New model will overwrite the previous `trained_models/demand_forecasting_model.pkl`

## File Structure

```
project_root/
│
├── cleaning.ipynb              # Data preprocessing notebook
├── modeling.ipynb              # Model training notebook
│
├── preprocessed_data/          # Generated by cleaning.ipynb
│   ├── X_train.parquet
│   ├── X_test.parquet
│   ├── y_train.parquet
│   ├── y_test.parquet
│   ├── feature_names.pkl
│   └── label_encoders.pkl
│
├── trained_models/             # Generated by modeling.ipynb
│   ├── demand_forecasting_model.pkl
│   └── model_metadata.pkl
│
├── feature_importance.png      # Generated by modeling.ipynb
├── residuals_analysis.png      # Generated by modeling.ipynb
│
└── README.md                   # This file
```

## Features Included in the Model

### Temporal Features
- `day` - Day of month (1-31)
- `month` - Month (1-12)
- `year` - Year
- `dayofweek` - Day of week (0=Monday, 6=Sunday)
- `week` - Week number (ISO)
- `quarter` - Quarter (1-4)
- `is_weekend` - Binary flag for weekends

### Cyclical Encoded Features
- `month_sin`, `month_cos` - Sinusoidal encoding of month for circular distance

### Lag Features
- `rolling_avg_quantity_w7` - 7-day rolling average
- `rolling_avg_quantity_w14` - 14-day rolling average
- `rolling_avg_quantity_w30` - 30-day rolling average

### Holiday Features
- `is_holiday` - Russian public holidays indicator

### Store & Product Features
- `store_id` - Store identifier
- `area` - Store sales area
- `price_base` - Base selling price
- `dept_name` - Product department (encoded)
- `class_name` - Product class (encoded)
- `subclass_name` - Product subclass (encoded)
- `item_type` - Product type (encoded)
- `format` - Store format (encoded)
- `division` - Store division (encoded)
- `city` - Store location (encoded)

## Model Details

### Algorithm: LightGBM (Light Gradient Boosting Machine)

**Why LightGBM?**
- Fast training (handles large datasets efficiently)
- Excellent for tabular data with mixed feature types
- Built-in handling of categorical variables
- Feature importance is interpretable
- Handles missing values automatically

**Hyperparameters:**
- `n_estimators=300` - Number of boosting rounds
- `learning_rate=0.05` - Step size shrinkage
- `num_leaves=63` - Max leaves per tree
- `colsample_bytree=0.8` - Column subsampling
- `subsample=0.8` - Row subsampling
- `early_stopping_rounds=10` - Stop if no improvement

### Target Transformation

The target variable (quantity) is **log-transformed** using `log1p`:
- Handles right-skewed demand distribution
- Reduces impact of outliers
- Improves model convergence
- RMSLE metric automatically accounts for this

Predictions are back-transformed using `expm1` to get original quantities.

## Performance Metrics

### RMSLE (Root Mean Squared Logarithmic Error)
- Emphasizes percentage error rather than absolute error
- Good for demand forecasting where small quantities are common
- Formula: $\sqrt{\frac{1}{n}\sum_{i=1}^{n}(\log(1+y_i) - \log(1+\hat{y}_i))^2}$

### MAE (Mean Absolute Error)
- Average absolute difference in original units
- Easy to interpret business-wise (e.g., "off by 5 units on average")
- Formula: $\frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$

### R² Score
- Proportion of variance explained by the model
- Range: 0 to 1 (higher is better)
- Formula: $1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$

## Tips for Improvement

1. **Add more lag features** - Create lags for 1, 2, 3 days to capture recent trend
2. **Seasonal decomposition** - Add day-of-week/month-of-year averages
3. **External variables** - Include competitor pricing, promotions, weather data
4. **Hyperparameter tuning** - Use Optuna or GridSearch for LightGBM parameters
5. **Ensemble methods** - Combine LightGBM with XGBoost/Neural Networks
6. **Forecast combinations** - Average predictions from multiple models

## Troubleshooting

### Issue: "FileNotFoundError: preprocessed_data/..."
**Solution:** Make sure `cleaning.ipynb` completed successfully and created the `preprocessed_data/` folder.

### Issue: Memory errors on large datasets
**Solution:** In `cleaning.ipynb`, reduce the sample size or use chunking to process data in batches.

### Issue: Model predictions are always the same value
**Solution:** Check that features are correctly loaded and have no missing values. Verify train/test data wasn't corrupted.

### Issue: Poor model performance
**Suggestions:**
- Add more relevant features
- Tune hyperparameters
- Check data quality (outliers, missing values)
- Visualize predictions vs actuals to identify patterns

## License

This project is for educational/demonstration purposes.

---

**Created:** April 2026  
**Status:** Production-Ready Architecture
