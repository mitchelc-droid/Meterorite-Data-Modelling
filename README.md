# NASA Meteorite Landings — Preprocessing & Classification Pipeline

**Course:** COMP 3523-001: Applied Machine Learning — Mount Royal University  
**Group 6:** Mitchel Chanthaseng, John David, Gia Nguyen, Marco Serpico

---

## Overview

This pipeline downloads the NASA Meteorite Landings dataset from Kaggle, cleans and preprocesses it, and trains a logistic regression model to classify whether a meteorite was **Fell** (witnessed falling) or **Found** (discovered after the fact).

---

## Requirements

Install dependencies before running:

```bash
pip install kagglehub pandas numpy scikit-learn imbalanced-learn
```

You will also need a Kaggle account with an API key configured. See [Kaggle API setup](https://www.kaggle.com/docs/api) for instructions.

---

## Usage

```bash
python meteorite_preprocessing.py
```

The script will automatically download the dataset, preprocess it, train the model, and print evaluation results to the console.

---

## Dataset

- **Source:** [NASA Meteorite Landings on Kaggle](https://www.kaggle.com/datasets/nasa/meteorite-landings)
- **Origin:** The Meteoritical Society's Meteoritical Bulletin Database via NASA's Open Data Portal
- **Size:** ~45,700 records, 10 columns

### Key Columns

| Column | Description |
|---|---|
| `name` | Meteorite name |
| `recclass` | Meteorite classification |
| `mass (g)` | Mass in grams |
| `fall` | Whether it was `Fell` (observed) or `Found` (discovered later) |
| `year` | Year of fall or discovery |
| `reclat` / `reclong` | Geographic coordinates |
| `nametype` | Whether the entry is `Valid` or `Relict` |

---

## Pipeline Steps

### 1. Load
Downloads the dataset via `kagglehub` and auto-detects the CSV file.

### 2. Basic Cleaning
- Removes duplicate rows
- Standardises column names to lowercase with underscores
- Filters years to valid range: **860–2016**
- Coerces mass to float and applies a **log transformation** (`log1p`) to handle right skew
- Drops rows where both latitude and longitude are missing
- Drops rows with placeholder coordinates **(0, 0)**

### 3. Feature Definitions
- **Numeric features:** `log_mass_g`, `year`, `reclat`, `reclong`
- **Categorical features:** `recclass`, `nametype`
- **Target:** `fall` (Fell = 0, Found = 1)
- Reduces `recclass` cardinality by keeping the top 30 classes and grouping the rest as `"Other"`

### 4. Sklearn Pipelines
- **Numeric pipeline:** median imputation → standard scaling
- **Categorical pipeline:** most-frequent imputation → one-hot encoding
- Combined via `ColumnTransformer`

### 5. Train / Test Split
- 80% train, 20% test
- Stratified on the target variable to preserve class balance

### 6. Fit & Transform
- Fits the preprocessor on training data only (prevents data leakage)
- Transforms both train and test sets

### 7. Logistic Regression
- Trained with `max_iter=1000` to ensure convergence

### 8. Evaluation
Prints the following metrics to the console:
- **Accuracy** — overall proportion of correct predictions
- **Precision** — of predicted "Fell", how many were actually "Fell"
- **Recall** — of actual "Fell", how many were correctly predicted
- **F1 Score** — harmonic mean of precision and recall
- **Classification report** — per-class breakdown
- **Confusion matrix** — predicted vs actual counts

### 9. Top Coefficients
Displays the 10 features with the largest absolute logistic regression coefficients — these are the features most influential in predicting Fell vs Found.

### 10. Save Output
Saves the processed arrays to a `preprocessed/` folder:
- `X_train.csv`, `X_test.csv` — feature matrices with column names
- `y_train.csv`, `y_test.csv` — encoded target labels

---

## Output Example

```
=== Model Evaluation ===
Accuracy  : 0.XXXX
Precision : 0.XXXX
Recall    : 0.XXXX
F1 Score  : 0.XXXX

=== Classification Report ===
              precision    recall  f1-score   support
        Fell       ...
       Found       ...

=== Confusion Matrix ===
       Fell  Found
Fell    ...    ...
Found   ...    ...

=== Top 10 Most Influential Features ===
feature   coefficient
...
```

---

## Notes

- The `fall` column is the **target variable** and is explicitly excluded from the input features to prevent data leakage.
- Rows missing coordinates are excluded from modelling since location is a key feature.
- The dataset is imbalanced — the vast majority of records are `Found`. Precision, recall, and F1 are more informative than accuracy alone in this case.
