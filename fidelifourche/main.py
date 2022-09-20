from fidelifourche.data import clean_data,merge_data,load_data
from fidelifourche.params import (LOCAL_DATA_PATH,DTYPES_RAW)

import pandas as pd
import os

# sklearn
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.compose import make_column_selector
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split

def preprocess():

    # NUM PIPE
    num_transformer = make_pipeline(SimpleImputer(strategy ='constant',fill_value= 0),
                                    RobustScaler())
    num_col = make_column_selector(dtype_include=['float64','int64'])

    # CAT PIPE
    cat_transformer = OneHotEncoder(handle_unknown='ignore')
    cat_col = make_column_selector(dtype_include=['object'])

    # FINAL PIPE
    preprocessor = make_column_transformer(
        (num_transformer, num_col),
        (cat_transformer, cat_col),
        remainder='passthrough'
    )

    # Load data
    orders,details,sav = load_data()

    # Merge the dataframes
    df_merge = merge_data(orders,details,sav)

    # Clean data using ml_logic.data.clean_data
    df = clean_data(df_merge)

    # Create X, y
    X = df.drop("bool_churn", axis=1)
    y = df[["bool_churn"]]

    print("✅ data preprocessed")


if __name__ == '__main__':
    try:
        preprocess()
    except:
        import ipdb, traceback, sys
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)
