"""
ChurnZero 26 — Banking Customer Churn Prediction
Team: aadipalsingh 

algo: LightGBM with domain-driven feature engineering and business cost optimisation
  1. Setup & Imports
  2. Load Data
  3. Exploratory Data Analysis
  4. Feature Engineering
  5. Preprocessing
  6. Model Training (LightGBM, 5-Fold CV)
  7. Threshold Optimisation (Business Cost)
  8. Evaluation (PR-AUC, F1, Confusion Matrix, Cost)
  9. Feature Importance
 10. Test Set Prediction
 11. Export Predictions CSV

Requirements:
    pip install lightgbm scikit-learn pandas numpy matplotlib seaborn imbalanced-learn

Usage:
    python ChurnZero_Solution.py
    -> Outputs: ChurnZero_Predictions.csv
"""

# 1. SETUP & IMPORTS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)
import lightgbm as lgb

# Reproducibility
SEED = 42
np.random.seed(SEED)

print("=" * 60)
print("  ChurnZero 26 — Banking Churn Prediction Pipeline")
print("=" * 60)


# 2. LOAD DATA

print("\n[1/10] Loading data...")

TRAIN_PATH = "ChurnZero_dataset_v1.csv"
TEST_PATH  = "ChurnZero_test_v1.csv"

# Update paths if running from a different directory
train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)

print(f"  Train shape : {train.shape}")
print(f"  Test  shape : {test.shape}")
print(f"  Churn rate  : {train['churn'].mean():.3f} ({train['churn'].sum()} churners)")

y        = train['churn'].copy()
test_ids = test['customer_id'].copy()


# 3. EXPLORATORY DATA ANALYSIS (summary prints)

print("\n[2/10] EDA Summary...")

# Class balance
print(f"\n  Class distribution:\n{y.value_counts().to_string()}")

# Key feature comparison
key_cols = [
    'avg_monthly_balance', 'nps_score', 'satisfaction_score',
    'last_login_days', 'total_complaints', 'digital_engagement_index',
    'account_inactive_days', 'customer_lifetime_value'
]
print("\n  Churn vs Non-Churn averages:")
print(train.groupby('churn')[key_cols].mean().round(2).T.to_string())

# Missing values
missing = train.isnull().sum()
missing = missing[missing > 0]
if len(missing):
    print(f"\n  Missing values:\n{missing.to_string()}")

# Single-feature AUC check (leakage test)
print("\n  Leakage check (top single-feature AUCs):")
single_aucs = {}
for col in train.select_dtypes(include='number').columns:
    if col not in ['customer_id', 'churn']:
        try:
            auc = roc_auc_score(y, train[col])
            single_aucs[col] = max(auc, 1 - auc)
        except Exception:
            pass
top_aucs = sorted(single_aucs.items(), key=lambda x: -x[1])[:5]
for feat, auc in top_aucs:
    print(f"    {feat:<45} AUC = {auc:.4f}")


# 4. FEATURE ENGINEERING

print("\n[3/10] Engineering features...")

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Domain-driven feature engineering.
    All features derived from existing columns — no target leakage.
    """
    df = df.copy()

    # -- Account health --
    # Balance stress: simultaneously falling balance AND low credit usage
    df['balance_stress'] = (
        df['balance_decline_percentage'] * (1 - df.get('credit_utilization_ratio', 0))
    )

    # -- Digital & activity recency composite --
    # Weighted recency: login freshness, account dormancy, service gap
    df['inactivity_score'] = (
        df['last_login_days']      * 0.50 +
        df['account_inactive_days'] * 0.30 +
        df['last_contacted_days']   * 0.20
    )

    # -- Complaint severity index --
    # Amplifies impact of unresolved + escalated complaints
    df['complaint_severity'] = (
        df['total_complaints'] *
        (1 + df['unresolved_complaint_count']) *
        (df['escalation_count'] + 1)
    )

    # -- Loyalty composite --
    # Normalised NPS + satisfaction; collapses two correlated signals
    df['loyalty_score'] = (df['nps_score'] / 10) + df['satisfaction_score']

    # -- Campaign effectiveness --
    # Ratio of responses to campaigns received; avoids divide-by-zero
    df['campaign_hit_rate'] = np.where(
        df['campaign_received_count'] > 0,
        df['campaign_response_count'] / (df['campaign_received_count'] + 1),
        0
    )

    # -- Wealth-to-deposit ratio --
    # Relative balance vs income; sudden drop = risk signal
    df['income_balance_ratio'] = np.where(
        df['annual_income'] > 0,
        df['avg_monthly_balance'] / (df['annual_income'] / 12 + 1),
        0
    )

    # -- Transaction velocity change --
    # Product captures customers slowing on BOTH count and value simultaneously
    df['trans_velocity_change'] = (
        df['total_amt_chng_q4_q1'] * df['total_ct_chng_q4_q1']
    )

    # -- Product breadth (cross-sell depth) --
    product_flags = [
        'savings_account_flag', 'current_account_flag', 'credit_card_flag',
        'personal_loan_flag', 'home_loan_flag', 'auto_loan_flag',
        'fixed_deposit_flag', 'investment_product_flag',
        'insurance_product_flag', 'demat_account_flag'
    ]
    df['product_breadth'] = df[product_flags].sum(axis=1)

    # -- Revolving balance risk --
    # Distinct from raw utilisation ratio — measures absolute revolving pressure
    df['revolving_risk'] = np.where(
        df['credit_card_limit'] > 0,
        df['total_revolving_bal'] / (df['credit_card_limit'] + 1),
        0
    )

    # -- Digital gap --
    # Customers who claim engagement but don't transact digitally
    df['digital_gap'] = (
        df['digital_engagement_index'] - df['digital_transaction_ratio'] * 100
    )

    return df

train = engineer_features(train)
test  = engineer_features(test)

engineered = [
    'balance_stress', 'inactivity_score', 'complaint_severity',
    'loyalty_score', 'campaign_hit_rate', 'income_balance_ratio',
    'trans_velocity_change', 'product_breadth', 'revolving_risk', 'digital_gap'
]
print(f"  Engineered {len(engineered)} new features: {engineered}")


# 5. PREPROCESSING

print("\n[4/10] Preprocessing...")

# Categorical columns
CAT_COLS = [
    'gender', 'marital_status', 'education_level', 'occupation_type',
    'income_band', 'income_category', 'city_tier', 'region',
    'customer_segment', 'onboarding_channel', 'relationship_type',
    'primary_account_type', 'card_category',
    'competitor_bank_offer_awareness', 'customer_feedback_sentiment'
]

# Fit encoder on combined corpus to handle all categories at inference time
le = LabelEncoder()
for col in CAT_COLS:
    combined = pd.concat([train[col], test[col]], axis=0).astype(str)
    le.fit(combined)
    train[col] = le.transform(train[col].astype(str))
    test[col]  = le.transform(test[col].astype(str))

# Missing value imputation
# app_rating_given: 56% missing — impute with median (non-parametric, safe)
median_rating = train['app_rating_given'].median()
train['app_rating_given'].fillna(median_rating, inplace=True)
test['app_rating_given'].fillna(median_rating, inplace=True)

# All remaining: fill with 0 (numerical flags/counts)
train.fillna(0, inplace=True)
test.fillna(0, inplace=True)

DROP_COLS = ['customer_id', 'churn']
FEAT_COLS = [c for c in train.columns if c not in DROP_COLS]

X      = train[FEAT_COLS]
X_test = test[FEAT_COLS]

print(f"  Feature matrix: {X.shape}")
print(f"  Test matrix:    {X_test.shape}")

# Class imbalance ratio for scale_pos_weight
scale_pos_weight = (y == 0).sum() / (y == 1).sum()
print(f"  Class ratio (neg/pos): {scale_pos_weight:.2f}")


# 6. MODEL TRAINING — LightGBM, 5-Fold Stratified CV

print("\n[5/10] Training LightGBM (5-Fold CV)...")

lgb_params = {
    # Objective & metric
    'objective':         'binary',
    'metric':            'average_precision',

    # Tree structure
    'n_estimators':      800,
    'learning_rate':     0.05,
    'max_depth':         6,
    'num_leaves':        63,

    # Regularisation
    'min_child_samples': 20,
    'feature_fraction':  0.8,
    'bagging_fraction':  0.8,
    'bagging_freq':      5,
    'lambda_l1':         0.1,
    'lambda_l2':         0.1,

    # Imbalance handling
    'scale_pos_weight':  scale_pos_weight,

    # Reproducibility
    'random_state':      SEED,
    'verbose':           -1,
    'n_jobs':            -1
}

model = lgb.LGBMClassifier(**lgb_params)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof_probs = cross_val_predict(model, X, y, cv=cv, method='predict_proba')[:, 1]

pr_auc = average_precision_score(y, oof_probs)
roc_auc = roc_auc_score(y, oof_probs)
print(f"  5-Fold OOF PR-AUC  : {pr_auc:.4f}")
print(f"  5-Fold OOF ROC-AUC : {roc_auc:.4f}")


# 7. THRESHOLD OPTIMISATION — Business Cost Function

print("\n[6/10] Optimising decision threshold (business cost)...")

FN_COST = 40_000   # ₹ cost of missing a churner (lost revenue)
FP_COST = 500      # ₹ cost of false alarm (wasted retention offer)

best_thresh   = 0.5
best_cost     = float('inf')
best_metrics  = {}

for t in np.arange(0.05, 0.95, 0.02):
    preds = (oof_probs >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, preds).ravel()
    cost = fn * FN_COST + fp * FP_COST
    if cost < best_cost:
        best_cost = cost
        best_thresh = t
        best_metrics = {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}

best_preds = (oof_probs >= best_thresh).astype(int)
best_f1    = f1_score(y, best_preds)

print(f"  Optimal threshold  : {best_thresh:.2f}")
print(f"  Total business cost: ₹{best_cost:,.0f}")
print(f"  F1 at threshold    : {best_f1:.4f}")
print(f"  Confusion matrix   : {best_metrics}")


# 8. EVALUATION REPORT

print("\n[7/10] Evaluation report...")

print("\n  Classification Report:")
print(classification_report(y, best_preds, target_names=['No Churn', 'Churn']))

tn = best_metrics['tn']
fp = best_metrics['fp']
fn = best_metrics['fn']
tp = best_metrics['tp']

cost_no_model = int(y.sum()) * FN_COST
model_savings = cost_no_model - best_cost

print(f"\n  Business Cost Summary:")
print(f"    FN Cost  ({fn} missed × ₹{FN_COST:,})   : ₹{fn * FN_COST:,}")
print(f"    FP Cost  ({fp} false alarms × ₹{FP_COST:,}): ₹{fp * FP_COST:,}")
print(f"    Total model cost                     : ₹{best_cost:,}")
print(f"    Cost without model                   : ₹{cost_no_model:,}")
print(f"    Estimated savings from model         : ₹{model_savings:,}")


# 9. FEATURE IMPORTANCE

print("\n[8/10] Computing feature importance...")

# Refit on full training data for final feature importances
model.fit(X, y)
fi = pd.Series(model.feature_importances_, index=FEAT_COLS).sort_values(ascending=False)

print("\n  Top 15 features by LightGBM gain importance:")
for feat, score in fi.head(15).items():
    flag = " ← ENGINEERED" if feat in engineered else ""
    print(f"    {feat:<50} {int(score):>4}{flag}")


# 10. TEST SET PREDICTION

print("\n[9/10] Generating test predictions...")

test_probs = model.predict_proba(X_test)[:, 1]
test_preds = (test_probs >= best_thresh).astype(int)

print(f"  Test set shape     : {X_test.shape}")
print(f"  Predicted churners : {test_preds.sum()} ({test_preds.mean():.1%})")
print(f"  Predicted non-churn: {(test_preds == 0).sum()}")

# Sanity check: predicted churn rate should be close to training churn rate
print(f"  Training churn rate: {y.mean():.1%}")
print(f"  Test predicted rate: {test_preds.mean():.1%}  ← should be similar")


# 11. EXPORT PREDICTIONS CSV

print("\n[10/10] Exporting predictions...")

pred_df = pd.DataFrame({
    'customer_id':       test_ids,
    'churn_prediction':  test_preds,
    'churn_probability': np.round(test_probs, 4)
})

# Validate required format
assert len(pred_df) == 2026,            f"Expected 2026 rows, got {len(pred_df)}"
assert pred_df.isnull().sum().sum() == 0, "Nulls found in predictions"
assert set(pred_df['churn_prediction'].unique()).issubset({0, 1}), "Predictions must be 0/1"
assert pred_df['churn_probability'].between(0, 1).all(), "Probabilities must be in [0,1]"

OUTPUT_PATH = "ChurnZero_aadipalsingh_Predictions.csv"
pred_df.to_csv(OUTPUT_PATH, index=False)

print(f"\n  ✅ Predictions saved to: {OUTPUT_PATH}")
print(f"  Rows         : {len(pred_df)}")
print(f"  Columns      : {list(pred_df.columns)}")
print(f"  Null values  : {pred_df.isnull().sum().sum()}")
print(f"\n  Sample output:")
print(pred_df.head(5).to_string(index=False))

print("\n" + "=" * 60)
print("  Pipeline complete.")
print(f"  PR-AUC    : {pr_auc:.4f}")
print(f"  F1 Score  : {best_f1:.4f}")
print(f"  Threshold : {best_thresh:.2f}")
print(f"  Business Cost (₹): {best_cost:,}")
print("=" * 60)
