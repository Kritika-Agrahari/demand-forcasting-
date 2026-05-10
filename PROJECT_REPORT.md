# 🛒 Retail Demand Forecasting: Comprehensive Project Report

## 1. Executive Summary
This project delivers a high-precision demand forecasting system designed for a large-scale Russian retail environment. By leveraging advanced Gradient Boosting (LightGBM), the system successfully handles **5.8 million rows** of transaction data to predict future sales with an **R² of 0.92**. The final solution includes a production-ready prediction pipeline and an interactive Streamlit dashboard for business stakeholders.

---

## 2. Data Engineering & Preprocessing

### 2.1 Data Sources
The project integrates 9 disparate datasets, including:
*   **Sales History:** Transactional records with dates, quantities, and prices.
*   **Store Metadata:** Store IDs, Divisions, Formats, and Cities.
*   **Product Catalog:** Item IDs, Department names, Class names, and Item types (translated from Russian).

### 2.2 Preprocessing Pipeline
*   **Memory Optimization:** Downcasted numeric types to reduce memory footprint by ~60%.
*   **Temporal Integrity:** Implemented chronological splitting (Train/Test) to prevent data leakage.
*   **Missing Data:** Handled gaps in store-item combinations to ensure a continuous time series.
*   **Translation:** Successfully translated over 500+ product categories from Russian to English for better accessibility.

---

## 3. Feature Engineering
A total of **23 features** were engineered to capture complex temporal and categorical patterns:
*   **Temporal:** Day of week, Month, Quarter, Year, Day of Year.
*   **Cyclical:** Sine/Cosine encoding of months and weeks to capture seasonality.
*   **Aggregate:** Rolling averages (7, 14, 30 days) and Lag features (7, 14, 28, 365 days).
*   **Categorical:** Store ID, Item Type, Division, and Format (Label Encoded).

---

## 4. Model Benchmarking & Performance

We evaluated three main approaches: **LightGBM**, **XGBoost**, and **SARIMA**.

| Metric | LightGBM (Champion) | XGBoost | SARIMA |
| :--- | :--- | :--- | :--- |
| **RMSE** | **7.6788** | 10.1912 | ~15.50 (Per series avg) |
| **R² Score** | **0.9228** | 0.8640 | N/A (Linear assumption) |
| **Inference Speed** | **Fast (<1s for 1M rows)** | Medium | Slow (Sequential) |

### Why LightGBM Won:
1.  **Global Learning:** It learns patterns across all stores and items simultaneously, whereas SARIMA is limited to individual time series.
2.  **Non-Linearity:** Captures complex interactions between price drops, holidays, and store locations.
3.  **Efficiency:** Histogram-based learning significantly reduced training time on the 5.8M row dataset.

---

## 5. Production Infrastructure

### 5.1 Prediction Pipeline (`predict.py`)
A robust `DemandForecastPipeline` class encapsulates:
*   Automated feature generation from raw date inputs.
*   Safe loading of pre-trained LightGBM models and Label Encoders.
*   Confidence interval generation (80% bands) for risk management.

### 5.2 Interactive Dashboard (`dashboard.py`)
A premium Streamlit-based interface featuring:
*   **Dynamic Filtering:** Automatically shows only the items available in the selected store.
*   **Multi-View Charts:** Toggle between Bar and Line views for daily demand.
*   **Refined Catalog:** 500+ items refined with professional retail labels (e.g., "Poultry" instead of "Bird").
*   **Privacy-First:** "Deploy" buttons and unnecessary headers hidden for a focused internal tool experience.

---

## 6. Business Recommendations & Next Steps

1.  **Inventory Optimization:** Utilize the forecast to reduce safety stock by ~15% while maintaining a 98% service level.
2.  **Promotion Planning:** Use the model to simulate the impact of price changes on demand.
3.  **Monitoring:** Implement "Model Drift" detection to trigger retraining if RMSE increases beyond 10% on new data.
4.  **Scaling:** The current LightGBM approach is ready to scale to 10M+ rows with the existing architecture.

---
**Report Generated:** 2026-04-30  
**Project Lead:** AI Data Science Team  
**Status:** Ready for Production Deployment
