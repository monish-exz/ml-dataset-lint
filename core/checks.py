# =============================#
#    Missing Value Analysis    #
# =============================#

def check_missing_values(df, threshold = 30):
    warnings = []

    total_rows = len(df)
    missing_counts = df.isna().sum()

    for column, missing in missing_counts.items():
        if missing > 0:
            percent_missing = (missing / total_rows) * 100
            print(f"{column} ---> {percent_missing:.2f}% missing")

            # WARNING RULE
            if percent_missing > threshold:
                warnings.append(
                    f"[WARN] Column '{column}' has {percent_missing:.2f}% missing values"
                )
    return warnings


# =============================#
#    Duplicate Row Analysis    #
# =============================#

def check_duplicates(df, threshold = 5):
    warnings = []

    total_rows = len(df)
    dup_rows = df.duplicated().sum()
    percent = (dup_rows / total_rows) * 100
        
    print("\nDuplicate Rows Report")
    print("---------------------")
    print(f"Total Duplicate Rows: {dup_rows}")
    print(f"{percent:.2f}% of duplicate rows")

    # WARNING RULE
    if percent > threshold:
        warnings.append(
            f"[WARN] Dataset contains {duplicate_rows} duplicate rows ({percent_dup_rows:.2f}%)"
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


