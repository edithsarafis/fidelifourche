from fidelifourche.data import clean_data,merge_data,load_data,merge_zip,compress
from fidelifourche.params import (LOCAL_DATA_PATH,DTYPES_RAW)
from fidelifourche.registry import load_model

import os
import pandas as pd

from sklearn.model_selection import train_test_split

from fidelifourche.preproc import preproc_pipe,preproc_transform,pipeline_fit_transform

def clean_merge():

    # Load data
    orders,details,sav,nb_epicerie_bio,zip_invalid = load_data()

    # Merge the dataframes
    df_merge = merge_data(orders,details,sav)

    # Clean data using ml_logic.data.clean_data
    df_clean = clean_data(df_merge)

    # Merge zip data
    df = merge_zip(df_clean,nb_epicerie_bio,zip_invalid)

    # Compressing df

    df.zip_valid=df.zip_valid.astype('object')

    df = compress(df)

    print("✅ data cleaned, merged and compressed")

    return df

def preprocess(df:pd.DataFrame, stratify=False):

    # Create final test dataset and save it in raw_data
    df[-5000:].to_csv(os.path.join(LOCAL_DATA_PATH, "test_data.csv"))

    # New train/val dataset
    df = df[:-5000]

    # Create X, y
    X = df.drop("bool_churn", axis=1)
    y = df[["bool_churn"]]

    # Train/val split
    X_train,X_val,y_train,y_val = train_test_split(X, y, test_size=0.3)

    # Preprocess
    model=load_model()

    preprocessor = model.named_steps["columntransformer"]

    X_train_preproc = preprocessor.fit_transform(X_train)
    X_val_preproc = preprocessor.transform(X_val)

    #preprocessor =  preproc_pipe()
    #X_train_preproc = pipeline_fit_transform(preprocessor,X_train)
    #X_val_preproc = preproc_transform(preprocessor,X_val)

    X_train_preproc = pd.DataFrame(X_train_preproc.toarray(),
             columns=preprocessor.get_feature_names_out())

    X_val_preproc = pd.DataFrame(X_val_preproc.toarray(),
             columns=preprocessor.get_feature_names_out())

    print("\n✅ X_train processed, with shape", X_train_preproc.shape)
    print("\n✅ X_val processed, with shape", X_val_preproc.shape)

    print("✅ data preprocessed")

    return X_train_preproc,y_train,X_val_preproc,y_val,preprocessor

def pipe_predict(pipe,start_date,end_date):

    test_data = os.path.join(LOCAL_DATA_PATH, "test_data.csv")
    df_test = pd.read_csv(
        test_data,
        dtype={"department": "O","zip_valid":"O"})


    df_test['created_at'] = pd.to_datetime(df_test['created_at'])
    df_test_X=df_test.loc[df_test['created_at']>=start_date,:].loc[df_test['created_at']<=end_date,:]
    df_test_X.loc[:,'predictions']=pipe.predict(df_test_X)

    return df_test_X.to_json()


if __name__ == '__main__':
    try:
        df = clean_merge()
    except:
        import ipdb, traceback, sys
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)
