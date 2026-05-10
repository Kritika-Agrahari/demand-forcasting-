import json

with open("2_comparison.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        src = "".join(cell['source'])
        if "lgb.LGBMRegressor(" in src and "lgb_model.fit(" in src:
            new_src = """val_size = int(len(X_train) * 0.15)
X_val, y_val = X_train.iloc[-val_size:], y_train.iloc[-val_size:]
X_tr, y_tr = X_train.iloc[:-val_size], y_train.iloc[:-val_size]

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
    pickle.dump(lgb_model, f)
print("✓ LightGBM saved")"""
            lines = new_src.split('\n')
            cell['source'] = [l + '\n' for l in lines[:-1]] + [lines[-1]]
            
        elif "XGBRegressor(" in src and "xgb_model.fit(" in src:
            new_src = """xgb_model = XGBRegressor(
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
    pickle.dump(xgb_model, f)
print("✓ XGBoost saved")"""
            lines = new_src.split('\n')
            cell['source'] = [l + '\n' for l in lines[:-1]] + [lines[-1]]

        elif "prophet_model = Prophet(" in src:
            new_src = src.replace("seasonality_mode='additive',  # Changed from multiplicative to additive (more stable)", "seasonality_mode='multiplicative',")
            new_src = new_src.replace("future = prophet_model.make_future_dataframe(periods=len(prophet_test))", "future = prophet_model.make_future_dataframe(periods=len(prophet_test), freq='D')")
            new_src = new_src.replace("prophet_pred = forecast.tail(len(prophet_test))['yhat'].values", "forecast_indexed = forecast.set_index('ds')\n    prophet_pred = forecast_indexed.loc[prophet_test['ds'], 'yhat'].values")
            lines = new_src.split('\n')
            new_lines = [l + '\n' for l in lines[:-1]]
            if len(lines) > 0:
                new_lines.append(lines[-1])
            cell['source'] = new_lines

compare_idx = -1
sarima_idx = -1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown':
        src = "".join(cell['source'])
        if "Compare All Models" in src:
            compare_idx = i
            cell['source'] = ["## 8. Compare All Models (LightGBM, XGBoost, Prophet, SARIMA)"]
        elif "Train SARIMA" in src:
            sarima_idx = i
            cell['source'] = ["## 7. Train SARIMA"]
        elif "Visualize Model Comparison" in src:
            cell['source'] = ["## 9. Visualize Model Comparison"]

if compare_idx != -1 and sarima_idx != -1 and compare_idx < sarima_idx:
    nb['cells'][compare_idx], nb['cells'][sarima_idx] = nb['cells'][sarima_idx], nb['cells'][compare_idx]

with open("2_comparison.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
