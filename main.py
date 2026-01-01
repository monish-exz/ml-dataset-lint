#####################
## ML-DATASET-LINT ##
#####################

from core.loader import load_dataset
from core.summary import print_dataset_summary
from core.checks import (
    check_missing_values,
    check_duplicates,
    check_constant_columns
)
from core.report import print_warnings

def main():
    user_input = input("ENTER THE PATH (OR DROP THE FILE: ")

    df = load_dataset(user_input)

    print_dataset_summary(df)

    warnings = []

    warnings.extend(check_missing_values(df))
    warnings.extend(check_duplicates(df))
    warnings.extend(check_constant_columns(df))

    print_warnings(warnings)

if __name__ == "__main__":
    main()