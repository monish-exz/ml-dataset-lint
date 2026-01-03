#####################
## ML-DATASET-LINT ##
#####################

from core.loader import load_dataset
from core.summary import print_dataset_summary
from core.checks import (
    check_missing_values,
    check_duplicates,
    check_constant_columns,
    validate_target_column,
    extract_column,
    class_frequencies,
    class_percents,
    class_imbalance
)
from core.report import print_warnings


def main():
    user_input = input("ENTER THE PATH (OR) DROP THE FILE: ")
    df = load_dataset(user_input)

    print_dataset_summary(df)

    warnings = []

    # v0.1 checks
    warnings.extend(check_missing_values(df))
    warnings.extend(check_duplicates(df))
    warnings.extend(check_constant_columns(df))

    print("===========================================================")

    # v0.2 — Target analysis
    target_column = input("\nENTER THE TARGET COLUMN (OR LEAVE BLANK): ")

    if not validate_target_column(df, target_column):
        print_warnings(warnings)
        return

    X, y = extract_column(df, target_column)

    print("\nTarget Preview:")
    print(y.head())

    class_frequencies(df, target_column)
    class_percents(df, target_column)

    label_warnings = class_imbalance(
        df,
        target_column,
        dominance_threshold=90,
        rare_count_threshold=20
    )

    warnings.extend(label_warnings)

    print("\n================= LINT WARNINGS =================")
    print_warnings(warnings)


if __name__ == "__main__":
    main()
