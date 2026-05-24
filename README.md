# ChurnZero 26 — Banking Customer Churn Prediction
### Team: aadipalsingh | Round 2 Submission
 
---
 
## Problem Statement
 
Banks lose 15–25% of customers every year to churn — costing them deposits, cross-sell revenue, and long-term value. This project builds a machine learning model that predicts which customers are likely to churn, so the bank can intervene early and retain them.
 
**Given:** 97 features per customer — profile, account behaviour, digital engagement, complaints, marketing response.  
**Predict:** Will this customer churn (1) or not (0)?
 
---
 
## Our Approach
 
1. **Exploratory Data Analysis** — Identified key signals: churners have 63% lower balances, NPS of 10 vs 38, and log in 2× less frequently
2. **Feature Engineering** — Built 10 new domain-driven features capturing compound risk signals (e.g. inactivity score, complaint severity, loyalty composite)
3. **Model** — LightGBM with class-imbalance weighting (scale_pos_weight = 5.2)
4. **Threshold Optimisation** — Tuned decision threshold using business cost function (FN = ₹40,000, FP = ₹500) instead of default 0.5
5. **Validation** — Stratified 5-Fold Cross Validation, no data leakage

### How to run
1. pip install lightgbm scikit-learn pandas numpy
2. Place ChurnZero_dataset_v1.csv and ChurnZero_test_v1.csv in the same folder
3. python ChurnZero_aadipalsingh_Code.py
4. Output: ChurnZero_aadipalsingh_Predictions.csv
