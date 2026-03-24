"""
NASA Meteorite Landings — Data Preprocessing Pipeline
======================================================
Usage:
    1. Run the kagglehub download snippet to get `path`
    2. Pass that path below (or let it auto-detect the CSV)
    3. Run: python meteorite_preprocessing.py
"""

import os
import glob
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    LabelEncoder,
)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ── 1. LOAD ───────────────────────────────────────────────────────────────────

import kagglehub
path = kagglehub.dataset_download("nasa/meteorite-landings")
print("Path to dataset files:", path)

# Auto-find the CSV inside the downloaded folder
csv_files = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)
if not csv_files:
    raise FileNotFoundError(f"No CSV found under {path}")

csv_path = csv_files[0]
print(f"Loading: {csv_path}\n")
df = pd.read_csv(csv_path)

print("=== Raw Data ===")
print(df.shape) #45716 rows and 10 columns
print(df.dtypes)
print(df.head(3))

# ── 2. BASIC CLEANING ─────────────────────────────────────────────────────────

# Drop exact duplicate rows
df = df.drop_duplicates()

# Standardise column names (lowercase, strip spaces)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Keep only rows with a valid year and sensible range
df["year"] = pd.to_numeric(df["year"], errors="coerce")
df = df[(df["year"] >= 860) & (df["year"] <= 2016)]

# Mass: rename for clarity, coerce to float
mass_col = [c for c in df.columns if "mass" in c][0]
df.rename(columns={mass_col: "mass_g"}, inplace=True)
df["mass_g"] = pd.to_numeric(df["mass_g"], errors="coerce")

# Log-transform mass (heavy right skew) — add 1 to handle zeros
df["log_mass_g"] = np.log1p(df["mass_g"])

# Coordinates
df["reclat"] = pd.to_numeric(df["reclat"], errors="coerce")
df["reclong"] = pd.to_numeric(df["reclong"], errors="coerce")

# Drop rows where both lat AND long are missing (can't place them)
df = df.dropna(subset=["reclat", "reclong"], how="all")

print(f"\n=== After Basic Cleaning: {df.shape} ===")
print(f"Missing values:\n{df.isnull().sum()}\n")

#Drops rows where both lat and long are zero (invalid coordinates)
df = df[~((df["reclat"] == 0) & (df["reclong"] == 0))]

# ── 3. FEATURE DEFINITIONS ────────────────────────────────────────────────────

NUMERIC_FEATURES = ["log_mass_g", "year", "reclat", "reclong"]
CATEGORICAL_FEATURES = ["recclass", "fall", "nametype"]

# Reduce recclass cardinality — keep top N, group rest as "Other"
TOP_N_CLASSES = 30
top_classes = df["recclass"].value_counts().nlargest(TOP_N_CLASSES).index
df["recclass"] = df["recclass"].where(df["recclass"].isin(top_classes), other="Other")

# ── 4. SKLEARN PIPELINES ──────────────────────────────────────────────────────

numeric_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),   # fill missing with median
    ("scale",  StandardScaler()),                    # zero-mean, unit-variance
])

categorical_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline,         NUMERIC_FEATURES),
    ("cat", categorical_pipeline,     CATEGORICAL_FEATURES),
], remainder="drop")

# ── 5. TRAIN / TEST SPLIT ─────────────────────────────────────────────────────

# Use `fall` (Fell vs Found) as an example target for classification
label_col = "fall"
df_model = df.dropna(subset=[label_col])

MODEL_CATEGORICAL = [f for f in CATEGORICAL_FEATURES if f != label_col]
le = LabelEncoder()
y = le.fit_transform(df_model[label_col])
print(f"Class mapping: { {cls: i for i, cls in enumerate(le.classes_)} }")
X = df_model[NUMERIC_FEATURES + MODEL_CATEGORICAL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 6. FIT & TRANSFORM ────────────────────────────────────────────────────────

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline,     NUMERIC_FEATURES),
    ("cat", categorical_pipeline, MODEL_CATEGORICAL),
], remainder="drop")

X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc  = preprocessor.transform(X_test)

# Recover feature names for inspection
ohe_cats = (
    preprocessor
    .named_transformers_["cat"]["encode"]
    .get_feature_names_out(MODEL_CATEGORICAL)
    .tolist()
)
feature_names = NUMERIC_FEATURES + ohe_cats

print("=== Preprocessed Output ===")
print(f"X_train shape : {X_train_proc.shape}")
print(f"X_test  shape : {X_test_proc.shape}")
print(f"Total features: {len(feature_names)}")
print(f"\nSample feature names: {feature_names[:8]} ...")

# ── 7. LOGISTIC REGRESSION MODEL ─────────────────────────────────────────────

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_proc, y_train)

y_pred = model.predict(X_test_proc)

# ── 8. EVALUATION ─────────────────────────────────────────────────────────────

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall    = recall_score(y_test, y_pred, zero_division=0)
f1        = f1_score(y_test, y_pred, zero_division=0)

print("=== Model Evaluation ===")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

print("=== Confusion Matrix ===")
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
print(cm_df)

# ── 9. TOP COEFFICIENTS ───────────────────────────────────────────────────────

coef_df = pd.DataFrame({
    "feature":     feature_names,
    "coefficient": model.coef_[0],
}).sort_values("coefficient", key=abs, ascending=False)

print("\n=== Top 10 Most Influential Features ===")
print(coef_df.head(10).to_string(index=False))

# ── 10. OPTIONAL — SAVE PROCESSED ARRAYS ─────────────────────────────────────

out_dir = "preprocessed"
os.makedirs(out_dir, exist_ok=True)
pd.DataFrame(X_train_proc, columns=feature_names).to_csv(f"{out_dir}/X_train.csv", index=False)
pd.DataFrame(X_test_proc,  columns=feature_names).to_csv(f"{out_dir}/X_test.csv",  index=False)
pd.DataFrame({"y": y_train}).to_csv(f"{out_dir}/y_train.csv", index=False)
pd.DataFrame({"y": y_test}).to_csv(f"{out_dir}/y_test.csv",   index=False)