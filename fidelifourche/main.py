from fidelifourche.data import clean_data,merge_data,load_data,merge_zip
from fidelifourche.params import (LOCAL_DATA_PATH,DTYPES_RAW)

import os
import pandas as pd

from fidelifourche.preproc import preprocess_features

def preprocess():

    # Load data
    orders,details,sav,nb_epicerie_bio,zip_invalid = load_data()

    # Merge the dataframes
    df_merge = merge_data(orders,details,sav)

    # Clean data using ml_logic.data.clean_data
    df_clean = clean_data(df_merge)

    # Merge zip data
    df = merge_zip(df_clean,nb_epicerie_bio,zip_invalid)

    # Create X, y
    X = df.drop("bool_churn", axis=1)
    y = df[["bool_churn"]]

    print("✅ data preprocessed")

    return df


if __name__ == '__main__':
    try:
        preprocess()
    except:
        import ipdb, traceback, sys
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)
