# 📊 Model Comparison & Technical Defense

## 1. The Head-to-Head Comparison

This table summarizes the performance of the three models tested on our 5.8M row dataset.

| Feature | **LightGBM (Champion)** | **XGBoost** | **SARIMA** |
| :--- | :--- | :--- | :--- |
| **Accuracy (RMSE)** | **7.67 (Best)** | 10.19 | ~15.50 |
| **R² Score** | **0.9228** | 0.8640 | N/A |
| **Model Type** | Global (Cross-Store Learning) | Global (Cross-Store Learning) | Local (1 model per series) |
| **Categorical Data** | Native Support (Fast) | Requires One-Hot Encoding | No support for metadata |
| **Non-Linearity** | Excellent | Excellent | Poor (Linear only) |
| **Scalability** | Handles 5.8M+ rows easily | Struggles with memory at 5M+ | Impossible at this scale |

---

## 2. Why we chose LightGBM (The "Defense")

### ❌ What was "Wrong" with SARIMA?
1.  **The Scale Nightmare:** SARIMA requires a separate model for every single store-item combination. Managing 50,000+ individual models is a production failure waiting to happen.
2.  **The "Feature Blindness":** SARIMA only looks at sales history. It cannot see that a store is in Moscow, or that a price drop occurred, or that a promotion is active.
3.  **Noisy Data:** Retail data is full of "spikes" and "zeros." SARIMA assumes a smooth, linear trend which leads to massive errors during promotional events.

### ❌ What was "Wrong" with XGBoost?
1.  **Training Speed:** XGBoost is significantly slower on large datasets (5M+ rows) because it evaluates every split point. LightGBM uses a **Histogram-based** approach that is up to 10x faster.
2.  **Accuracy Gap:** LightGBM achieved a significantly lower RMSE (7.67 vs 10.19). In retail, a 2.5 point improvement in RMSE can save millions in wasted inventory and lost sales.

---

## 3. Anticipated Questions & Conditions (FAQ)

### **Q: What if we open a brand-new store with zero sales history?**
*   **The "Cold Start" Condition:** Most models fail here.
*   **The Solution:** LightGBM handles this perfectly. Because it is a **Global Model**, it uses the *metadata* (City, Format, Division) to predict demand based on how similar stores in the same city behave.

### **Q: What if the data size grows to 50 million rows?**
*   **The Scalability Condition:**
*   **The Solution:** LightGBM is built for high-performance large data. It supports distributed training and leaf-wise tree growth, which scales better than almost any other algorithm.

### **Q: Why does the model predict decimals (e.g., 5.4 items) when we sell whole units?**
*   **The Intepretation Condition:**
*   **The Solution:** The model predicts the **Expected Value (Mean)**. A value of 5.4 represents the "statistical requirement." For inventory purposes, you should always **round up** (to 6) to ensure you meet the probability of that demand.

### **Q: When should we NOT use LightGBM?**
*   **The Small Data Condition:**
*   **The Solution:** If you have less than 100 rows of data for a specific item and no other store context, a simple Moving Average or SARIMA would be safer to avoid overfitting.

### **Q: How do we know the model is still accurate next month?**
*   **The Model Drift Condition:**
*   **The Solution:** We implement **Drift Monitoring**. Every month, we compare the current model's predictions against actual sales. If the error increases by >10%, we trigger an automatic retrain on the latest data.

---
**Document Status:** Final  
**Recommendation:** Proceed with LightGBM Production Deployment
