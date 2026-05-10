import pickle
import os
import numpy as np
import pandas as pd

class DemandForecastPipeline:
    def __init__(self, model_dir='trained_models', data_dir='preprocessed_data'):
        # Load best model (LightGBM)
        model_path = f'{model_dir}/lgb_model.pkl'
        if not os.path.exists(model_path):
            model_path = f'{model_dir}/demand_forecasting_model.pkl'
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        # Load encoders and feature list
        with open(f'{data_dir}/label_encoders.pkl', 'rb') as f:
            self.le_dict = pickle.load(f)
        with open(f'{data_dir}/feature_names.pkl', 'rb') as f:
            self.features = pickle.load(f)
        
        print("✓ Pipeline loaded successfully")
    
    def preprocess(self, df_raw):
        """Generate features from raw input"""
        df = df_raw.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        # Date features
        df['day']         = df['date'].dt.day
        df['month']       = df['date'].dt.month
        df['year']        = df['date'].dt.year
        df['dayofweek']   = df['date'].dt.dayofweek
        df['week']        = df['date'].dt.isocalendar().week.astype(int)
        df['quarter']     = df['date'].dt.quarter
        df['is_weekend']  = df['dayofweek'].isin([5, 6]).astype(int)
        df['month_sin']   = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos']   = np.cos(2 * np.pi * df['month'] / 12)
        
        # Advanced date features
        df['day_of_year'] = df['date'].dt.dayofyear
        df['days_since_start'] = (df['date'] - df['date'].min()).dt.days
        df['is_month_start'] = (df['date'].dt.day <= 3).astype(int)
        df['is_month_end'] = (df['date'].dt.day >= 28).astype(int)
        df['is_payday_near'] = ((df['date'].dt.day >= 23) | (df['date'].dt.day <= 5)).astype(int)
        
        # Cyclical encoding
        df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        df['week_sin'] = np.sin(2 * np.pi * df['week'] / 53)
        df['week_cos'] = np.cos(2 * np.pi * df['week'] / 53)
        
        # Use provided features or set defaults
        if 'price_base' not in df.columns:
            df['price_base'] = 500.0
        if 'is_holiday' not in df.columns:
            df['is_holiday'] = 0
            
        # Optional rolling/lag features - use defaults if not provided
        for col in ['rolling_avg_quantity_w7', 'rolling_avg_quantity_w14', 'rolling_avg_quantity_w30', 
                    'lag_7_quantity', 'lag_14_quantity', 'lag_28_quantity', 'lag_365_quantity']:
            if col not in df.columns:
                df[col] = 5.0
        
        # Categorical features - set defaults
        categorical_cols = ['area', 'dept_name', 'class_name', 'subclass_name', 'item_type', 'format', 'division', 'city']
        for col in categorical_cols:
            if col not in df.columns:
                df[col] = 'unknown'
        
        # Encode categoricals - IMPORTANT: handle NaN and missing
        for col in categorical_cols:
            if col in self.le_dict:
                le = self.le_dict[col]
                # Convert to string, handle NaN
                df[col] = df[col].fillna('unknown').astype(str)
                
                # Check if value is in encoder classes, else use 'unknown'
                df[col] = df[col].apply(lambda x: x if x in le.classes_ else 'unknown')
                
                # Transform to numeric
                df[col] = le.transform(df[col]).astype('float32')
        
        # Ensure all columns are numeric
        for col in df.columns:
            if col != 'date':
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Select only required features
        df = df[self.features]
        
        return df
    
    def predict(self, df_raw, return_interval=False):
        """Generate predictions"""
        X = self.preprocess(df_raw)
        pred_log = self.model.predict(X)
        pred     = np.expm1(np.maximum(pred_log, 0))
        
        results = df_raw[['date','item_id','store_id']].copy()
        results['predicted_quantity'] = np.round(pred, 2)
        
        if return_interval:
            results['lower_bound'] = np.round(pred * 0.80, 2)
            results['upper_bound'] = np.round(pred * 1.20, 2)
        
        return results

# Usage example
if __name__ == '__main__':
    pipeline = DemandForecastPipeline()
    
    # Example: predict for next 7 days
    sample_input = pd.DataFrame({
        'date':     pd.date_range('2024-01-01', periods=7),
        'item_id':  [1001] * 7,
        'store_id': [5]    * 7
    })
    
    predictions = pipeline.predict(sample_input, return_interval=True)
    print("\n✓ Predictions generated successfully!")
    print(predictions)
