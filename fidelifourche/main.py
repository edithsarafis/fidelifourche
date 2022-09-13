from fidelifourche.data import clean_data,merge_data
from fidelifourche.params import (LOCAL_DATA_PATH,DTYPES_RAW)

import pandas as pd
import os

def preprocess():
    """
    Preprocess the dataset
    """

    print("\n⭐️ use case: preprocess")

    # Retrieve raw data
    orders_raw_path = os.path.join(LOCAL_DATA_PATH, "orders.csv")
    orders = pd.read_csv(
        orders_raw_path,
        dtype=DTYPES_RAW
        )

    #details_raw_path = os.path.join(LOCAL_DATA_PATH, "orders.csv")
    #orders = pd.read_csv(data_raw_path, dtype=DTYPES_RAW_OPTIMIZED)


    # Merge the dataframes
    #data = merge_data(orders, details, catalog)

    # Clean data using ml_logic.data.clean_data
    df = clean_data(orders)

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
