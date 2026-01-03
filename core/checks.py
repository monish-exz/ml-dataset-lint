# =============================#
#    Missing Value Analysis    #
# =============================#

def check_missing_values(df, threshold=30):
    warnings = []

    total_rows = len(df)
    missing_counts = df.isna().sum()

    for column, missing in missing_counts.items():
        if missing > 0:
            percent = (missing / total_rows) * 100
            print(f"{column} ---> {percent:.2f}% missing")

            if percent > threshold:
                warnings.append(
                    f"[WARN] Column '{column}' has {percent:.2f}% missing values"
                )

    return warnings


# =============================#
#    Duplicate Row Analysis    #
# =============================#

def check_duplicates(df, threshold=5):
    warnings = []

    total_rows = len(df)
    dup_rows = df.duplicated().sum()
    percent = (dup_rows / total_rows) * 100

    print("\nDuplicate Rows Report")
    print("---------------------")
    print(f"Total Duplicate Rows: {dup_rows}")
    print(f"{percent:.2f}% of duplicate rows")

    if percent > threshold:
        warnings.append(
            f"[WARN] Dataset contains {dup_rows} duplicate rows ({percent:.2f}%)"
        )

    return warnings


# =============================#
#   Constant Column Analysis   #
# =============================#

def check_constant_columns(df):
    warnings = []

    unique_values = df.nunique()
    print("\nUnique Values Report")
    print("--------------------")
    print(unique_values)

    for column, count in unique_values.items():
        if count == 1:
            warnings.append(
                f"[INFO] Column '{column}' is constant (only 1 unique value)"
            )

    return warnings


# =============================#
#     Accept Target Column     #
# =============================#

def validate_target_column(df, column_name):
    if column_name.strip() == "":
        print("No target column provided — skipping label analysis")
        return False

    if column_name not in df.columns:
        print(f"[ERROR] Target column '{column_name}' not found")
        return False

    print(f"Target column '{column_name}' validated")
    return True


# =============================#
#     Extract Target Column    #
# =============================#

def extract_column(df, column_name):
    y = df[column_name]
    X = df.drop(columns=[column_name])
    return X, y


# =============================#
#     Count Class Frequencies  #
# =============================#


def class_frequencies(df, column_name):
    counts = df[column_name].value_counts()
    print(f"\nClass Frequencies for '{column_name}':")
    print(counts)


# =============================#
#  Calculate Class Percentages #
# =============================#

def class_percents(df, column_name):
    percents = df[column_name].value_counts(normalize=True) * 100
    print(f"\nClass Percentages for '{column_name}':")
    print(percents.round(2))


# =============================#
#    Detect Imbalance & Rarity #
# =============================#

def class_imbalance(
    df,
    column_name,
    dominance_threshold=90,
    rare_count_threshold=20
):
    warnings = []

    counts = df[column_name].value_counts()
    percents = counts / counts.sum() * 100

    majority_class = percents.idxmax()
    majority_percent = percents.max()

    print("\nLabel Distribution Analysis")
    print("---------------------------")
    print(f"Majority class: {majority_class} ({majority_percent:.2f}%)")

    if majority_percent > dominance_threshold:
        warnings.append(
            f"[WARN] Label '{majority_class}' dominates dataset ({majority_percent:.2f}%)"
        )

    for label, count in counts.items():
        if count < rare_count_threshold:
            warnings.append(
                f"[WARN] Label '{label}' has only {count} samples"
            )

    return warnings
