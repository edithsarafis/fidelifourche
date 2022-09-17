from fidelifourche.data import clean_data,merge_data,load_data
from fidelifourche.params import (LOCAL_DATA_PATH,DTYPES_RAW)

import pandas as pd
import os

def preprocess():
    """
    Preprocess the dataset
    """

    print("\n⭐️ use case: preprocess")

    # Load data
    orders,details,sav = load_data()

    # Merge the dataframes
    df_merge = merge_data(orders,details,sav)

    # Clean data using ml_logic.data.clean_data
    df = clean_data(df_merge)

    # Create X, y
    #X = df.drop("bool_churn", axis=1)
    #y = df[["bool_churn"]]

    print("✅ data preprocessed")


if __name__ == '__main__':
    try:
        preprocess()
    except:
        import ipdb, traceback, sys
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)
