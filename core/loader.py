# =============================#
#        Load Dataset          #
# =============================#

import pandas as pd

def load_dataset(path):
     """
     Loads dataset from a CSV file.
     """

     return pd.read_csv(path)