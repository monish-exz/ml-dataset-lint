import pandas as pd
import numpy as np

# =============================
# Missing Value Analysis
# =============================

def check_missing_values(df, threshold=30):
    warnings = []
    total_rows = len(df)

    for col, missing in df.isna().sum().items():
        if missing == 0:
            continue

        percent = (missing / total_rows) * 100
        print(f"{col}: {percent:.2f}% missing")

        if percent > threshold:
            warnings.append(
                f"[WARN] Column '{col}' has {percent:.2f}% missing values"
            )

    return warnings


# =============================
# Duplicate Rows
# =============================

def check_duplicates(df, threshold=5):
    warnings = []

    dup_count = df.duplicated().sum()
    percent = (dup_count / len(df)) * 100

    print(f"\nDuplicate rows: {dup_count} ({percent:.2f}%)")

    if percent > threshold:
        warnings.append(
            f"[WARN] Dataset contains {dup_count} duplicate rows"
        )

    return warnings


# =============================
# Constant Columns
# =============================

def check_constant_columns(df):
    warnings = []

    for col, unique in df.nunique().items():
        if unique == 1:
            warnings.append(
                f"[INFO] Column '{col}' is constant"
            )

    return warnings


# =============================
# Target Validation
# =============================

def validate_target_column(df, col):
    if not col:
        print("[INFO] No target provided — skipping target-based checks")
        return False

    if col not in df.columns:
        print(f"[ERROR] Target column '{col}' not found")
        return False

    return True


def extract_column(df, col):
    return df.drop(columns=[col]), df[col]


# =============================
# Label Distribution
# =============================

def class_frequencies(df, col):
    print("\nClass counts:")
    print(df[col].value_counts())


def class_percents(df, col):
    print("\nClass percentages:")
    print((df[col].value_counts(normalize=True) * 100).round(2))


def class_imbalance(df, col, dominance_threshold=90, rare_count_threshold=20):
    warnings = []

    counts = df[col].value_counts()
    percents = counts / counts.sum() * 100

    if percents.max() > dominance_threshold:
        warnings.append(
            f"[WARN] Label '{percents.idxmax()}' dominates dataset ({percents.max():.2f}%)"
        )

    for label, count in counts.items():
        if count < rare_count_threshold:
            warnings.append(
                f"[WARN] Label '{label}' has only {count} samples"
            )

    return warnings


# =============================
# Leakage Checks
# =============================

def leakage_checks(df, target, corr_threshold=0.9, id_ratio=0.95):
    warnings = []

    numeric_df = df.select_dtypes(include='number')

    if target not in numeric_df.columns:
        print("[INFO] Target is non-numeric — skipping leakage checks")
        return warnings

    X = numeric_df.drop(columns=[target])
    y = numeric_df[target]

    high_corr = set()
    id_like = set()

    for col in X.columns:
        corr = X[col].corr(y)
        if pd.isna(corr):
            continue

        if abs(corr) > corr_threshold:
            warnings.append(
                f"[WARN] Feature '{col}' highly correlated with target ({corr:.2f})"
            )
            high_corr.add(col)

        unique_ratio = X[col].nunique() / len(df)
        if unique_ratio > id_ratio or any(k in col.lower() for k in ['id', 'uuid', 'index']):
            warnings.append(
                f"[WARN] Feature '{col}' appears ID-like"
            )
            id_like.add(col)

    for col in high_corr & id_like:
        warnings.append(
            f"[CRITICAL] Feature '{col}' shows strong leakage risk"
        )

    return warnings


# =============================
# Bias Signals
# =============================

def bias_signals(df, target, min_group_pct=10):
    warnings = []

    if target not in df.columns:
        return warnings

    cat_df = df.select_dtypes(exclude='number')
    if cat_df.empty:
        return warnings

    sensitive_keys = ['gender', 'sex', 'race', 'age', 'region', 'income']

    for col in cat_df.columns:
        if col == target:
            continue

        if not any(k in col.lower() for k in sensitive_keys):
            continue

        dist = cat_df[col].value_counts(normalize=True) * 100

        for group, pct in dist.items():
            if pct < min_group_pct:
                warnings.append(
                    f"[WARN] Group '{group}' in '{col}' under-represented ({pct:.2f}%)"
                )

    return warnings


# =============================
# Dataset Health
# =============================

def dataset_health(df):
    warnings = []

    if len(df) < 100:
        warnings.append("[INFO] Dataset is very small (<100 rows)")

    if df.shape[1] > df.shape[0]:
        warnings.append("[WARN] More features than samples")

    return warnings


# =============================
# Trainability Checks
# =============================

def trainability_checks(df):
    warnings = []

    if df.isna().sum().sum() > 0:
        warnings.append("[INFO] Missing values must be handled before training")

    if df.select_dtypes(include='number').empty:
        warnings.append("[WARN] No numeric features for ML models")

    return warnings
