# =============================#
#       Dataset Summary        #
# =============================#

def print_dataset_summary(df):

    """
    Prints basic dataset information.
    """
    rows, cols = df.shape

    print("\nDataset Summary")
    print("---------------")
    print(f"Rows: {rows}")
    print(f"Columns: {cols}")

    print("\nColumns & DataTypes:")
    for col, dtype in df.dtypes.items():
        print(f"{col} ---> {dtype}")
