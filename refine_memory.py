import json

with open("2_comparison.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        src = "".join(cell['source'])
        
        # LightGBM cell memory fix
        if "lgb_model = lgb.LGBMRegressor(" in src and "X_val, y_val" in src:
            new_src = """import gc
gc.collect()

val_size = int(len(X_train) * 0.15)

# Free up memory by deleting X_train and y_train since we only need the splits
X_val = X_train.iloc[-val_size:].copy()
y_val = y_train.iloc[-val_size:].copy()
X_tr = X_train.iloc[:-val_size].copy()
y_tr = y_train.iloc[:-val_size].copy()

# IMPORTANT: X_train and y_train are deleted here to save memory. 
# If you need to re-run this cell, please re-run the Data Loading cell first.
del X_train, y_train
gc.collect()

lgb_model = lgb.LGBMRegressor(
    n_estimators=500, learning_rate=0.05, num_leaves=63,
    colsample_bytree=0.8, subsample=0.8, random_state=42, verbose=-1, n_jobs=-1
)
lgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric='rmse',
              callbacks=[lgb.early_stopping(stopping_rounds=50)])

lgb_pred = lgb_model.predict(X_test)

# Aggregate to daily to compare fairly
test_df = X_test.copy()
test_df['pred_log'] = lgb_pred
test_df['actual_log'] = y_test.values

daily_lgb = test_df.groupby('date').agg(
    pred_qty=('pred_log', lambda x: np.expm1(np.maximum(x, 0)).sum()),
    actual_qty=('actual_log', lambda x: np.expm1(x).sum())
).reset_index()

mask_lgb = daily_lgb.actual_qty > 0
lgb_mape = np.mean(np.abs((daily_lgb.actual_qty[mask_lgb] - daily_lgb.pred_qty[mask_lgb]) / daily_lgb.actual_qty[mask_lgb])) * 100

lgb_rmse = rmse_original(y_test, lgb_pred)
lgb_mae = mae_original(y_test, lgb_pred)
lgb_r2 = r2_original(y_test, lgb_pred)

print(f"{'='*50}")
print(f"LightGBM: RMSE={lgb_rmse:.4f}, MAE={lgb_mae:.4f}, R²={lgb_r2:.4f}, Aggregate MAPE={lgb_mape:.2f}%")
print(f"{'='*50}")

with open('trained_models/lgb_model.pkl', 'wb') as f:
    import pickle
    pickle.dump(lgb_model, f)
print("✓ LightGBM saved")
"""
            cell['source'] = [line + '\n' for line in new_src.split('\n')]
            
        # Clean up memory after XGBoost
        if "xgb_model = XGBRegressor(" in src:
            new_src = """import gc

xgb_model = XGBRegressor(
    n_estimators=500, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
    early_stopping_rounds=50, eval_metric='rmse', verbosity=0
)
xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

xgb_pred = xgb_model.predict(X_test)

# Aggregate to daily to compare fairly
test_df_xgb = X_test.copy()
test_df_xgb['pred_log'] = xgb_pred
test_df_xgb['actual_log'] = y_test.values

daily_xgb = test_df_xgb.groupby('date').agg(
    pred_qty=('pred_log', lambda x: np.expm1(np.maximum(x, 0)).sum()),
    actual_qty=('actual_log', lambda x: np.expm1(x).sum())
).reset_index()

mask_xgb = daily_xgb.actual_qty > 0
xgb_mape = np.mean(np.abs((daily_xgb.actual_qty[mask_xgb] - daily_xgb.pred_qty[mask_xgb]) / daily_xgb.actual_qty[mask_xgb])) * 100

xgb_rmse = rmse_original(y_test, xgb_pred)
xgb_mae = mae_original(y_test, xgb_pred)
xgb_r2 = r2_original(y_test, xgb_pred)

print(f"{'='*50}")
print(f"XGBoost: RMSE={xgb_rmse:.4f}, MAE={xgb_mae:.4f}, R²={xgb_r2:.4f}, Aggregate MAPE={xgb_mape:.2f}%")
print(f"{'='*50}")

with open('trained_models/xgb_model.pkl', 'wb') as f:
    import pickle
    pickle.dump(xgb_model, f)
print("✓ XGBoost saved")

# Free up tree model variables to save memory for Prophet/SARIMA
del xgb_model, X_tr, y_tr, X_val, y_val
gc.collect()
"""
            cell['source'] = [line + '\n' for line in new_src.split('\n')]

        # Clean up Prophet data loading
        if "train_raw = pd.read_parquet(" in src:
            new_src = src.replace("prophet_test.columns = ['ds', 'y']\n", "prophet_test.columns = ['ds', 'y']\n\n# Free up raw data memory\ndel train_raw, test_raw\nimport gc\ngc.collect()\n")
            cell['source'] = [line + '\n' for line in new_src.split('\n')]

with open("2_comparison.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
