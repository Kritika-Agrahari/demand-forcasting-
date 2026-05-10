import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 🤖 Final Production Modeling — Demand Forecasting\n",
    "\n",
    "This notebook implements the winning configurations found during research:\n",
    "1. **LightGBM**: Champion for granular item-level forecasting.\n",
    "2. **SARIMA**: Champion for aggregate daily trends."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "%pip install lightgbm pmdarima statsmodels pandas numpy matplotlib seaborn scikit-learn -q\n",
    "\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import warnings, time, pickle, os\n",
    "import gc\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error\n",
    "from sklearn.model_selection import TimeSeriesSplit\n",
    "import lightgbm as lgb\n",
    "from statsmodels.tsa.statespace.sarimax import SARIMAX\n",
    "\n",
    "warnings.filterwarnings('ignore')\n",
    "print(\"✅ Libraries imported.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Load & Prepare Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "data_dir = 'preprocessed_data'\n",
    "try:\n",
    "    X_train = pd.read_parquet(f'{data_dir}/X_train.parquet')\n",
    "    X_test = pd.read_parquet(f'{data_dir}/X_test.parquet')\n",
    "    y_train = pd.read_parquet(f'{data_dir}/y_train.parquet').iloc[:, 0]\n",
    "    y_test = pd.read_parquet(f'{data_dir}/y_test.parquet').iloc[:, 0]\n",
    "    with open(f'{data_dir}/feature_names.pkl', 'rb') as f:\n",
    "        FEATS = pickle.load(f)\n",
    "    print(\"✅ Loaded data from parquet files.\")\n",
    "    print(\"\\nFeatures loaded:\")\n",
    "    print(FEATS)\n",
    "except Exception as e:\n",
    "    print(\"⚠️ Parquet files not found. Please run preprocessing.ipynb first or ensure data is loaded.\")\n",
    "    raise e"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Cross-Validation (Time Series Split)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"🚀 Running TimeSeriesSplit (3 Folds) on LightGBM...\")\n",
    "tscv = TimeSeriesSplit(n_splits=3)\n",
    "fold = 1\n",
    "for train_idx, val_idx in tscv.split(X_train):\n",
    "    print(f\"--- Fold {fold} ---\")\n",
    "    # To save memory, we're not deep-copying here since it's just for fast CV metric calculation\n",
    "    X_cv_train = X_train.iloc[train_idx]\n",
    "    y_cv_train = y_train.iloc[train_idx]\n",
    "    X_cv_val = X_train.iloc[val_idx]\n",
    "    y_cv_val = y_train.iloc[val_idx]\n",
    "    \n",
    "    cv_model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, num_leaves=63, n_jobs=-1)\n",
    "    cv_model.fit(X_cv_train, y_cv_train, eval_set=[(X_cv_val, y_cv_val)], callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])\n",
    "    \n",
    "    cv_pred = cv_model.predict(X_cv_val)\n",
    "    cv_rmse = np.sqrt(mean_squared_error(np.expm1(y_cv_val), np.expm1(np.maximum(cv_pred, 0))))\n",
    "    print(f\"Fold {fold} RMSE: {cv_rmse:.4f}\")\n",
    "    fold += 1\n",
    "    del X_cv_train, y_cv_train, X_cv_val, y_cv_val, cv_model\n",
    "    gc.collect()\n",
    "print(\"✅ Cross-Validation Complete.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Train Item-Level Champion (LightGBM)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "if 'X_train' in locals():\n",
    "    print(\"🚀 Training Final Tuned LightGBM...\")\n",
    "    # Create a proper validation set from the LAST 15% of training data chronologically\n",
    "    val_size = int(len(X_train) * 0.15)\n",
    "    X_val = X_train.iloc[-val_size:].copy()\n",
    "    y_val = y_train.iloc[-val_size:].copy()\n",
    "    X_tr = X_train.iloc[:-val_size].copy()\n",
    "    y_tr = y_train.iloc[:-val_size].copy()\n",
    "    \n",
    "    # Free memory\n",
    "    del X_train, y_train\n",
    "    gc.collect()\n",
    "\n",
    "    lgb_model = lgb.LGBMRegressor(\n",
    "        n_estimators=500, \n",
    "        learning_rate=0.05, \n",
    "        num_leaves=63,\n",
    "        colsample_bytree=0.8, \n",
    "        subsample=0.8, \n",
    "        random_state=42, \n",
    "        verbose=-1,\n",
    "        n_jobs=-1\n",
    "    )\n",
    "\n",
    "    # EARLY STOPPING FIX: Use X_val, y_val, not X_test\n",
    "    lgb_model.fit(\n",
    "        X_tr, y_tr, \n",
    "        eval_set=[(X_val, y_val)], \n",
    "        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]\n",
    "    )\n",
    "\n",
    "    lgb_pred = lgb_model.predict(X_test)\n",
    "    \n",
    "    # Aggregate back to daily quantities to compute MAPE properly (Apples to Apples)\n",
    "    test_df = X_test.copy()\n",
    "    test_df['pred_log'] = lgb_pred\n",
    "    test_df['actual_log'] = y_test.values\n",
    "    \n",
    "    daily_lgb = test_df.groupby('date').agg(\n",
    "        pred_qty=('pred_log', lambda x: np.expm1(np.maximum(x, 0)).sum()),\n",
    "        actual_qty=('actual_log', lambda x: np.expm1(x).sum())\n",
    "    ).reset_index()\n",
    "    \n",
    "    actuals = daily_lgb['actual_qty'].values\n",
    "    preds = daily_lgb['pred_qty'].values\n",
    "    mask = actuals > 0\n",
    "    \n",
    "    rmse_lgb = np.sqrt(mean_squared_error(actuals, preds))\n",
    "    mae_lgb = mean_absolute_error(actuals, preds)\n",
    "    mape_lgb = np.mean(np.abs((actuals[mask] - preds[mask]) / actuals[mask])) * 100\n",
    "    r2_lgb = r2_score(actuals, preds)\n",
    "    \n",
    "    print(f\"\\n✅ LightGBM Aggregate Performance:\")\n",
    "    print(f\"RMSE: {rmse_lgb:.4f}\")\n",
    "    print(f\"MAE:  {mae_lgb:.4f}\")\n",
    "    print(f\"MAPE: {mape_lgb:.4f}%\")\n",
    "    print(f\"R²:   {r2_lgb:.4f}\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Train Aggregate Champion (SARIMA)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"🚀 Training Aggregate SARIMA...\")\n",
    "try:\n",
    "    train_raw = pd.read_parquet(f'{data_dir}/train_raw.parquet')\n",
    "    test_raw = pd.read_parquet(f'{data_dir}/test_raw.parquet')\n",
    "    \n",
    "    sarima_train = train_raw.groupby('date')['quantity'].sum()\n",
    "    sarima_test = test_raw.groupby('date')['quantity'].sum()\n",
    "    \n",
    "    # Free memory\n",
    "    del train_raw, test_raw\n",
    "    gc.collect()\n",
    "    \n",
    "    sarima_model = SARIMAX(\n",
    "        sarima_train, \n",
    "        order=(1,1,2), \n",
    "        seasonal_order=(1,0,1,7), \n",
    "        enforce_stationarity=False, \n",
    "        enforce_invertibility=False\n",
    "    ).fit(disp=False)\n",
    "    \n",
    "    sarima_pred = sarima_model.forecast(steps=len(sarima_test))\n",
    "    sarima_pred = np.maximum(np.nan_to_num(sarima_pred, nan=0.0), 0)\n",
    "    \n",
    "    actuals_s = sarima_test.values\n",
    "    preds_s = sarima_pred.values\n",
    "    mask_s = actuals_s > 0\n",
    "    \n",
    "    rmse_s = np.sqrt(mean_squared_error(actuals_s, preds_s))\n",
    "    mae_s = mean_absolute_error(actuals_s, preds_s)\n",
    "    mape_s = np.mean(np.abs((actuals_s[mask_s] - preds_s[mask_s]) / actuals_s[mask_s])) * 100\n",
    "    r2_s = r2_score(actuals_s, preds_s)\n",
    "    \n",
    "    print(f\"\\n✅ SARIMA Aggregate Performance:\")\n",
    "    print(f\"RMSE: {rmse_s:.4f}\")\n",
    "    print(f\"MAE:  {mae_s:.4f}\")\n",
    "    print(f\"MAPE: {mape_s:.4f}%\")\n",
    "    print(f\"R²:   {r2_s:.4f}\")\n",
    "    \n",
    "    # Generate daily predictions to use in the plot\n",
    "    daily_pred_s = pd.DataFrame({'date': sarima_test.index, 'pred_qty': preds_s})\n",
    "except Exception as e:\n",
    "    print(\"⚠️ Failed to train SARIMA:\", e)\n",
    "    daily_pred_s = None\n",
    "    raise e"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Final Performance Dashboard & Model Save"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "if 'lgb_model' in locals():\n",
    "    fig, ax = plt.subplots(1, 3, figsize=(22, 6))\n",
    "\n",
    "    # Feature Importance\n",
    "    imp = pd.Series(lgb_model.feature_importances_, index=X_tr.columns).sort_values()\n",
    "    imp.tail(15).plot(kind='barh', ax=ax[0], color='steelblue')\n",
    "    ax[0].set_title(\"LightGBM Feature Importance (Top 15)\")\n",
    "    ax[0].grid(axis='x', alpha=0.3)\n",
    "\n",
    "    # Residual Distribution (Bias Check)\n",
    "    residuals = actuals - preds\n",
    "    print(f\"\\n🔍 BIAS CHECK: Mean Residual = {residuals.mean():.4f} (Positive = Under-forecasting, Negative = Over-forecasting)\")\n",
    "    \n",
    "    sns.histplot(residuals, bins=50, ax=ax[1], kde=True, color='darkorange')\n",
    "    ax[1].set_title(\"Error Distribution (Actuals - Preds)\")\n",
    "    # Fix x-axis to 1st and 99th percentiles\n",
    "    p1, p99 = np.percentile(residuals, 1), np.percentile(residuals, 99)\n",
    "    ax[1].set_xlim(p1, p99)\n",
    "    ax[1].grid(alpha=0.3)\n",
    "\n",
    "    # Forecast vs Actual Line Plot\n",
    "    ax[2].plot(daily_lgb['date'], actuals, label='Actual Demand', color='black', alpha=0.7)\n",
    "    ax[2].plot(daily_lgb['date'], preds, label='LightGBM Pred', color='steelblue', alpha=0.9)\n",
    "    if daily_pred_s is not None:\n",
    "        ax[2].plot(daily_pred_s['date'], daily_pred_s['pred_qty'], label='SARIMA Pred', color='darkorange', alpha=0.9, linestyle='--')\n",
    "    ax[2].set_title(\"Forecast vs Actual (Daily Aggregates)\")\n",
    "    ax[2].legend()\n",
    "    ax[2].grid(alpha=0.3)\n",
    "    plt.xticks(rotation=45)\n",
    "\n",
    "    plt.tight_layout()\n",
    "    plt.savefig('final_model_performance.png', dpi=300)\n",
    "    plt.show()\n",
    "\n",
    "    # SAVE MODEL AFTER VALIDATION PASSES\n",
    "    os.makedirs('trained_models', exist_ok=True)\n",
    "    with open('trained_models/lgb_model_final.pkl', 'wb') as f:\n",
    "        pickle.dump(lgb_model, f)\n",
    "    if 'sarima_model' in locals():\n",
    "        with open('trained_models/sarima_model_final.pkl', 'wb') as f:\n",
    "            pickle.dump(sarima_model, f)\n",
    "    print(\"\\n✓ Models saved to trained_models/ AFTER validation passes.\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open("Final_Production_Model.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)
