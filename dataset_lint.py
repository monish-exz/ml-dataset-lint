#####################
## ML-DATASET-LINT ##
#####################

# v0.1 - Core Dataset Linting

import numpy as np
import pandas as pd

# =============================
# Load Dataset
# =============================
user_input = input("ENTER THE PATH (OR) DROP THE FILE: ")
data = pd.read_csv(user_input)

# container to store warnings
warnings = []

# =============================
# Dataset Summary
# =============================
print("\nDataset Summary")
print("---------------")

rows, cols = data.shape
print(f"Rows: {rows}")
print(f"Columns: {cols}")

print("\nColumns & DataTypes:")
for col, dtype in data.dtypes.items():
    print(f"{col} ---> {dtype}")

# =============================
# Missing Value Analysis
# =============================
print("\nMissing Value Report")
print("--------------------")

total_rows = rows
missing_counts = data.isna().sum()

for column, missing in missing_counts.items():
    if missing > 0:
        percent_missing = (missing / total_rows) * 100
        print(f"{column} ---> {percent_missing:.2f}% missing")

        # WARNING RULE
        if percent_missing > 30:
            warnings.append(
                f"[WARN] Column '{column}' has {percent_missing:.2f}% missing values"
            )

# =============================
# Duplicate Row Analysis
# =============================
duplicate_rows = data.duplicated().sum()
percent_dup_rows = (duplicate_rows / total_rows) * 100

print("\nDuplicate Rows Report")
print("---------------------")
print(f"Total Duplicate Rows: {duplicate_rows}")
print(f"{percent_dup_rows:.2f}% of duplicate rows")

# WARNING RULE
if percent_dup_rows > 5:
    warnings.append(
        f"[WARN] Dataset contains {duplicate_rows} duplicate rows ({percent_dup_rows:.2f}%)"
    )

# =============================
# Constant Column Analysis
# =============================
unique_values = data.nunique()

print("\nUnique Values Report")
print("--------------------")
print(unique_values)

for column, unique_count in unique_values.items():
    if unique_count == 1:
        warnings.append(
            f"[INFO] Column '{column}' is constant (only 1 unique value)"
        )

# =============================
# Lint Warnings Output
# =============================
print("\nLint Warnings")
print("-------------")

if len(warnings) == 0:
    print("No major issues detected")
else:
    for warn in warnings:
        print(warn)
