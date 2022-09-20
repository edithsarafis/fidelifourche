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

def preprocess_features(X: pd.DataFrame) -> np.ndarray:

    print("\nPreprocess features...")

    # NUM PIPE
    num_transformer = make_pipeline(SimpleImputer(strategy ='constant',fill_value= 0),
                                RobustScaler())
    num_col = make_column_selector(dtype_include=['float32','int8'])

    # CAT PIPE
    cat_transformer = OneHotEncoder(handle_unknown='ignore')
    cat_col = make_column_selector(dtype_include=['object'])

    # FINAL PIPE
    preprocessor = make_column_transformer(
        (num_transformer, num_col),
        (cat_transformer, cat_col),
        remainder='passthrough'
    )

    X_processed = preprocessor.fit_transform(X)

    print("\n✅ X_processed, with shape", X_processed.shape)

    return X_processed
