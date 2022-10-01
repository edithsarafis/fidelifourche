# sklearn
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.compose import make_column_selector
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

def preproc_pipe():

    print("\nPreprocess features...")
    # NUM PIPE
    num_transformer = make_pipeline(SimpleImputer(strategy ='constant',fill_value= 0),
                                RobustScaler())
    num_col = make_column_selector(dtype_include=['float32','float64','int8','int16'])

    # CAT PIPE
    cat_transformer = OneHotEncoder(handle_unknown='ignore')
    cat_col = ['delivery_type','carrier','acquisition_channel','department']

    # FINAL PIPE
    preprocessor = make_column_transformer(
        (num_transformer, num_col),
        (cat_transformer, cat_col),
        remainder='drop'
    )

    return preprocessor

def pipeline_fit_transform(preprocessor,X_train) -> np.ndarray:

    X_train_preprocessed = preprocessor.fit_transform(X_train)

    return X_train_preprocessed

def preproc_transform(preprocessor,X_test) :

    X_test_preprocessed = preprocessor.transform(X_test)

    return X_test_preprocessed
