"""
NASA Meteorite Landings — Machine Learning Pipeline
====================================================
Models  : Logistic Regression + Random Forest
Target  : Predict meteorite fall status  (Fell = 0 | Found = 1)

Usage:
    1. Run the kagglehub download snippet to obtain `path`
    2. Run: python meteorite-model-final-EN.py
"""

import os
import glob
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: LOAD DATA
# Use kagglehub to automatically download the NASA Meteorite Landings dataset
# from Kaggle, then locate the CSV file inside the downloaded folder.
# ═══════════════════════════════════════════════════════════════════════════════

import kagglehub
path = kagglehub.dataset_download("nasa/meteorite-landings")
print("Path to dataset files:", path)

csv_files = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)
if not csv_files:
    raise FileNotFoundError(f"No CSV file found under: {path}")

csv_path = csv_files[0]
print(f"Loading file: {csv_path}\n")
df = pd.read_csv(csv_path)

print("=== Raw Data ===")
print(f"Shape: {df.shape}")   # 45,716 rows × 10 columns
print(df.dtypes)
print(df.head(3))


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: BASIC DATA CLEANING
# Address common data quality issues: duplicate rows, inconsistent column names,
# out-of-range values, missing coordinates, and placeholder coordinates (0, 0).
# ═══════════════════════════════════════════════════════════════════════════════

# Remove exact duplicate rows to avoid biasing the model
df = df.drop_duplicates()

# Standardise column names: lowercase and replace whitespace with underscores
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Filter to valid year range: 860 CE – 2016 CE
# Values outside this window are likely parse errors or BCE dates
df["year"] = pd.to_numeric(df["year"], errors="coerce")
df = df[(df["year"] >= 860) & (df["year"] <= 2016)]

# Rename the mass column, coerce to float, then apply a log-transform
# Mass is heavily right-skewed, so log1p(x) = log(x+1) compresses the scale
# and handles edge cases where mass == 0
mass_col = [c for c in df.columns if "mass" in c][0]
df.rename(columns={mass_col: "mass_g"}, inplace=True)
df["mass_g"] = pd.to_numeric(df["mass_g"], errors="coerce")
df["log_mass_g"] = np.log1p(df["mass_g"])

# Coerce coordinates to float
df["reclat"]  = pd.to_numeric(df["reclat"],  errors="coerce")
df["reclong"] = pd.to_numeric(df["reclong"], errors="coerce")

# Drop rows missing BOTH latitude and longitude — location cannot be determined
df = df.dropna(subset=["reclat", "reclong"], how="all")

# Remove placeholder coordinates (0, 0) — this point lies off the west coast
# of Africa and does not represent a real meteorite recovery location
df = df[~((df["reclat"] == 0) & (df["reclong"] == 0))]

print(f"\n=== After Cleaning: {df.shape} ===")
print(f"Missing values:\n{df.isnull().sum()}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: FEATURE DEFINITIONS
# Select the input columns for the models.
# Reduce the high cardinality of recclass by keeping the top 30 classes
# and collapsing the rest into a single "Other" category.
# ═══════════════════════════════════════════════════════════════════════════════

# recclass has 400+ unique values — OneHotEncoding all of them would create
# an extremely sparse feature matrix. Keeping only the top 30 classes
# preserves most of the signal while keeping the feature space manageable.
TOP_N_CLASSES = 30
top_classes = df["recclass"].value_counts().nlargest(TOP_N_CLASSES).index
df["recclass"] = df["recclass"].where(df["recclass"].isin(top_classes), other="Other")

NUMERIC_FEATURES     = ["log_mass_g", "year", "reclat", "reclong"]
CATEGORICAL_FEATURES = ["recclass", "nametype"]   # "fall" is the target — excluded here


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: BUILD SKLEARN PIPELINES
# Pipelines automate preprocessing and prevent data leakage by ensuring
# statistics (e.g. median, category list) are learned only from the training set.
#   Numeric  : impute missing values with median → StandardScaler
#   Categorical: impute with most frequent → OneHotEncoder
# ═══════════════════════════════════════════════════════════════════════════════

# Numeric pipeline:
# - SimpleImputer(median): fills missing values with the column median,
#   which is more robust than the mean when outliers are present
# - StandardScaler: centres and scales each feature to mean=0, std=1,
#   which is required for Logistic Regression to converge properly
numeric_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale",  StandardScaler()),
])

# Categorical pipeline:
# - SimpleImputer(most_frequent): fills missing values with the most common category
# - OneHotEncoder: converts each category into a binary (0/1) column
#   handle_unknown="ignore": silently ignores unseen categories at test time
categorical_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

# Combine both pipelines via ColumnTransformer
# remainder="drop": any column not listed above is discarded
preprocessor = ColumnTransformer([
    ("num", numeric_pipeline,     NUMERIC_FEATURES),
    ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
], remainder="drop")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5: ENCODE LABELS AND SPLIT DATA
# Convert the "fall" column into numeric labels: Fell = 0, Found = 1.
# Use an 80/20 stratified split so each split retains the same class ratio.
# ═══════════════════════════════════════════════════════════════════════════════

label_col = "fall"
df_model  = df.dropna(subset=[label_col])

# LabelEncoder maps "Fell" → 0 and "Found" → 1 alphabetically
le = LabelEncoder()
y  = le.fit_transform(df_model[label_col])
X  = df_model[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

print(f"Class mapping : { {cls: i for i, cls in enumerate(le.classes_)} }")
print(f"Class distribution: {dict(zip(le.classes_, np.bincount(y)))}")
print(f"  → Severe imbalance: Found accounts for {np.bincount(y)[1]/len(y)*100:.1f}% of data\n")

# stratify=y ensures the Fell / Found ratio is preserved in both splits
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6: PREPROCESS THE DATA
# Fit the preprocessor on the training set only, then transform both sets.
# IMPORTANT: never fit on the test set — doing so leaks information and
# produces overly optimistic evaluation results.
# ═══════════════════════════════════════════════════════════════════════════════

# fit_transform on train: learns statistics (median, scale, category list)
X_train_proc = preprocessor.fit_transform(X_train)
# transform on test: applies the statistics learned from train — no re-fitting
X_test_proc  = preprocessor.transform(X_test)

# Recover feature names after OneHotEncoding for later interpretation
ohe_cats = (
    preprocessor.named_transformers_["cat"]["encode"]
    .get_feature_names_out(CATEGORICAL_FEATURES)
    .tolist()
)
feature_names = NUMERIC_FEATURES + ohe_cats

print("=== Preprocessing Output ===")
print(f"X_train: {X_train_proc.shape}   X_test: {X_test_proc.shape}")
print(f"Total features : {len(feature_names)}")
print(f"Sample feature names: {feature_names[:6]} ...\n")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 7: HANDLE CLASS IMBALANCE — SMOTE + RandomUnderSampler
# The dataset is severely imbalanced (~96.6% Found, ~3.4% Fell).
# Training directly on this data causes models to almost always predict Found.
#
# Strategy:
#   SMOTE (Synthetic Minority Oversampling Technique):
#     Creates new synthetic Fell samples by interpolating between existing ones.
#     sampling_strategy=0.5 → grows Fell to 50% of Found count.
#
#   RandomUnderSampler:
#     Randomly removes Found samples to match the oversampled Fell count.
#     sampling_strategy=1.0 → results in a balanced 50/50 training set.
#
# NOTE: Resampling is applied ONLY to the training set.
#       The test set remains untouched to reflect real-world class distribution.
# ═══════════════════════════════════════════════════════════════════════════════

print("=== Handling Class Imbalance ===")
print(f"Before resampling — Fell: {np.bincount(y_train)[0]}, Found: {np.bincount(y_train)[1]}")

smote = SMOTE(sampling_strategy=0.5, random_state=42)
rus   = RandomUnderSampler(sampling_strategy=1.0, random_state=42)

# Apply sequentially: oversample Fell first, then undersample Found
X_train_res, y_train_res = smote.fit_resample(X_train_proc, y_train)
X_train_res, y_train_res = rus.fit_resample(X_train_res, y_train_res)

print(f"After resampling  — Fell: {np.bincount(y_train_res)[0]}, Found: {np.bincount(y_train_res)[1]}")
print(f"→ 50/50 balanced training set achieved\n")


# ═══════════════════════════════════════════════════════════════════════════════
# ██████████████████████████████████████████████████████████████████████████████
# MODEL 1: LOGISTIC REGRESSION
# A linear baseline model for binary classification.
# Strengths : fast, interpretable via coefficients, good probability calibration
# Weaknesses: can only learn linear decision boundaries
# ██████████████████████████████████████████████████████████████████████████████
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("MODEL 1: LOGISTIC REGRESSION")
print("=" * 60)

# max_iter=1000: increase the iteration limit to ensure convergence
# random_state=42: fix the random seed for reproducibility
lr_model = LogisticRegression(max_iter=1000, random_state=42)

# Train on the resampled (50/50) data so the model pays attention to Fell
lr_model.fit(X_train_res, y_train_res)

# Predict on the ORIGINAL (unsampled) test set to evaluate real-world performance
y_pred_lr = lr_model.predict(X_test_proc)
y_prob_lr = lr_model.predict_proba(X_test_proc)[:, 1]


# ── Evaluate Model 1 ──────────────────────────────────────────────────────────

lr_acc  = accuracy_score(y_test, y_pred_lr)
lr_prec = precision_score(y_test, y_pred_lr, zero_division=0)
lr_rec  = recall_score(y_test, y_pred_lr, zero_division=0)
lr_f1   = f1_score(y_test, y_pred_lr, zero_division=0)
lr_auc  = roc_auc_score(y_test, y_prob_lr)

# 5-fold stratified cross-validation to check model stability across folds
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
lr_cv_f1 = cross_val_score(
    lr_model, X_train_res, y_train_res, cv=cv, scoring="f1", n_jobs=-1
).mean()

print(f"\nModel 1 Evaluation Results:")
print(f"  Accuracy       : {lr_acc:.4f}  ({lr_acc*100:.2f}%)")
print(f"  Precision      : {lr_prec:.4f}  (Found class)")
print(f"  Recall         : {lr_rec:.4f}  (Found class)")
print(f"  F1 Score       : {lr_f1:.4f}")
print(f"  ROC-AUC        : {lr_auc:.4f}")
print(f"  CV F1 (5-fold) : {lr_cv_f1:.4f}")

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred_lr, target_names=le.classes_, zero_division=0))

print("--- Confusion Matrix ---")
cm_lr     = confusion_matrix(y_test, y_pred_lr)
cm_lr_df  = pd.DataFrame(cm_lr, index=le.classes_, columns=le.classes_)
print(cm_lr_df)
print(f"\n  Fell  predicted as Fell  : {cm_lr[0][0]}")
print(f"  Fell  predicted as Found : {cm_lr[0][1]}")
print(f"  Found predicted as Fell  : {cm_lr[1][0]}")
print(f"  Found predicted as Found : {cm_lr[1][1]}")


# ── Top Coefficients — Model 1 ────────────────────────────────────────────────
# Positive coefficient (+): feature pushes the prediction toward Found
# Negative coefficient (−): feature pushes the prediction toward Fell
# Larger absolute value → stronger influence on the prediction

print("\n--- Top 10 Most Influential Features (Logistic Regression Coefficients) ---")
coef_df = pd.DataFrame({
    "feature":     feature_names,
    "coefficient": lr_model.coef_[0],
}).sort_values("coefficient", key=abs, ascending=False)
print(coef_df.head(10).to_string(index=False))


# ═══════════════════════════════════════════════════════════════════════════════
# ██████████████████████████████████████████████████████████████████████████████
# MODEL 2: RANDOM FOREST
# An ensemble of decision trees where each tree votes on the final prediction.
# Strengths : captures non-linear feature interactions, no feature scaling needed,
#             naturally produces Gini-based feature importance scores
# Weaknesses: slower to train than LR, less directly interpretable
# ██████████████████████████████████████████████████████████████████████████████
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("MODEL 2: RANDOM FOREST")
print("=" * 60)

# n_estimators=100: build 100 trees — a good balance of performance and speed
# class_weight="balanced": automatically upweights the minority class (Fell)
#   during training, equivalent to resampling but applied inside the algorithm
# n_jobs=-1: use all available CPU cores to train trees in parallel
# random_state=42: fix the seed for reproducibility
rf_model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

# Train on the resampled (50/50) training set
rf_model.fit(X_train_res, y_train_res)

# Predict on the original test set
y_pred_rf = rf_model.predict(X_test_proc)
y_prob_rf = rf_model.predict_proba(X_test_proc)[:, 1]


# ── Evaluate Model 2 ──────────────────────────────────────────────────────────

rf_acc  = accuracy_score(y_test, y_pred_rf)
rf_prec = precision_score(y_test, y_pred_rf, zero_division=0)
rf_rec  = recall_score(y_test, y_pred_rf, zero_division=0)
rf_f1   = f1_score(y_test, y_pred_rf, zero_division=0)
rf_auc  = roc_auc_score(y_test, y_prob_rf)

# 5-fold cross-validation to verify model stability
rf_cv_f1 = cross_val_score(
    rf_model, X_train_res, y_train_res, cv=cv, scoring="f1", n_jobs=-1
).mean()

print(f"\nModel 2 Evaluation Results:")
print(f"  Accuracy       : {rf_acc:.4f}  ({rf_acc*100:.2f}%)")
print(f"  Precision      : {rf_prec:.4f}  (Found class)")
print(f"  Recall         : {rf_rec:.4f}  (Found class)")
print(f"  F1 Score       : {rf_f1:.4f}")
print(f"  ROC-AUC        : {rf_auc:.4f}")
print(f"  CV F1 (5-fold) : {rf_cv_f1:.4f}")

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred_rf, target_names=le.classes_, zero_division=0))

print("--- Confusion Matrix ---")
cm_rf    = confusion_matrix(y_test, y_pred_rf)
cm_rf_df = pd.DataFrame(cm_rf, index=le.classes_, columns=le.classes_)
print(cm_rf_df)
print(f"\n  Fell  predicted as Fell  : {cm_rf[0][0]}")
print(f"  Fell  predicted as Found : {cm_rf[0][1]}")
print(f"  Found predicted as Fell  : {cm_rf[1][0]}")
print(f"  Found predicted as Found : {cm_rf[1][1]}")


# ── Feature Importance — Model 2 ──────────────────────────────────────────────
# Gini importance measures how much each feature reduces impurity across all
# splits in all trees. Values range from 0 to 1 and sum to exactly 1.
# A higher importance score means the feature is used at more influential
# branching nodes and contributes more to the final predictions.

print("\n--- Top 10 Most Influential Features (Random Forest — Gini Importance) ---")
rf_imp_df = pd.DataFrame({
    "feature":    feature_names,
    "importance": rf_model.feature_importances_,
}).sort_values("importance", ascending=False)
print(rf_imp_df.head(10).to_string(index=False))
print(f"\n  → '{rf_imp_df.iloc[0]['feature']}' contributes "
      f"{rf_imp_df.iloc[0]['importance']*100:.1f}% of total importance")
print(f"  → Top 4 features account for "
      f"{rf_imp_df.head(4)['importance'].sum()*100:.1f}% of total importance")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 8: MODEL COMPARISON
# Aggregate all evaluation metrics for both models into a single summary table
# to make it easy to identify which model performs better overall and
# specifically on the minority Fell class.
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("MODEL COMPARISON SUMMARY")
print("=" * 60)

lr_fell_recall = recall_score(y_test, y_pred_lr, pos_label=0, zero_division=0)
rf_fell_recall = recall_score(y_test, y_pred_rf, pos_label=0, zero_division=0)

comparison = pd.DataFrame({
    "Logistic Regression": {
        "Accuracy":         lr_acc,
        "Precision (Found)": lr_prec,
        "Recall (Found)":    lr_rec,
        "F1 Score":         lr_f1,
        "ROC-AUC":          lr_auc,
        "CV F1 (5-fold)":   lr_cv_f1,
        "Fell Recall":      lr_fell_recall,
        "Fell Precision":   precision_score(y_test, y_pred_lr, pos_label=0, zero_division=0),
    },
    "Random Forest": {
        "Accuracy":         rf_acc,
        "Precision (Found)": rf_prec,
        "Recall (Found)":    rf_rec,
        "F1 Score":         rf_f1,
        "ROC-AUC":          rf_auc,
        "CV F1 (5-fold)":   rf_cv_f1,
        "Fell Recall":      rf_fell_recall,
        "Fell Precision":   precision_score(y_test, y_pred_rf, pos_label=0, zero_division=0),
    },
}).T

print(comparison.to_string(float_format="{:.4f}".format))

# Determine the better model based on overall accuracy
best = "Random Forest" if rf_acc > lr_acc else "Logistic Regression"
print(f"\n→ Best model by Accuracy    : {best}")
print(f"→ Random Forest improved Fell Recall from "
      f"{lr_fell_recall*100:.1f}% to {rf_fell_recall*100:.1f}%")
print(f"  (Fell is the minority class — higher recall means fewer missed detections)")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 9: SAVE RESULTS
# Export all processed arrays and summary tables to CSV files
# for use in visualisation, reporting, or further analysis.
# ═══════════════════════════════════════════════════════════════════════════════

out_dir = "preprocessed"
os.makedirs(out_dir, exist_ok=True)

# Save preprocessed feature arrays
pd.DataFrame(X_train_proc, columns=feature_names).to_csv(f"{out_dir}/X_train.csv", index=False)
pd.DataFrame(X_test_proc,  columns=feature_names).to_csv(f"{out_dir}/X_test.csv",  index=False)
pd.DataFrame({"y": y_train}).to_csv(f"{out_dir}/y_train.csv", index=False)
pd.DataFrame({"y": y_test}).to_csv(f"{out_dir}/y_test.csv",   index=False)

# Save model comparison table
comparison.to_csv(f"{out_dir}/model_comparison.csv")

# Save Random Forest feature importances
rf_imp_df.to_csv(f"{out_dir}/rf_feature_importance.csv", index=False)

# Save Logistic Regression coefficients
coef_df.to_csv(f"{out_dir}/lr_coefficients.csv", index=False)

print(f"\n✓ All results saved to '{out_dir}/'")
print(f"  - X_train.csv, X_test.csv, y_train.csv, y_test.csv")
print(f"  - model_comparison.csv")
print(f"  - rf_feature_importance.csv")
print(f"  - lr_coefficients.csv")
