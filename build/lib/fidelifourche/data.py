from fidelifourche.params import (LOCAL_DATA_PATH,DTYPES_RAW)

import os
import datetime as dt
import pandas as pd

def load_data():

    print("\n✅ data loading...")

    orders_raw_path = os.path.join(LOCAL_DATA_PATH, "orders.csv")
    orders = pd.read_csv(
        orders_raw_path,
        dtype=DTYPES_RAW)

    details_raw_path = os.path.join(LOCAL_DATA_PATH, "order_details.json")
    chunks = pd.read_json(
       details_raw_path,
       lines=True,
       chunksize = 100)
    details = pd.concat([c for c in chunks])
    details.drop_duplicates(inplace=True)
    details = details.groupby(['order_id']).sum()


    sav = orders[['customer_id','ticket_at']]
    sav.drop_duplicates(inplace=True)
    sav = sav.groupby(['customer_id']).nunique().reset_index()


    epicerie_bio = os.path.join(LOCAL_DATA_PATH, "nb_epicerie_bio_1372.csv")
    nb_epicerie_bio = pd.read_csv(
        epicerie_bio,
        dtype={"zip": "O"})


    zip_error = os.path.join(LOCAL_DATA_PATH, "zipcode_invalide_875.csv")
    zip_invalid = pd.read_csv(
        zip_error,
        dtype={"zip": "O"})


    print("\n✅ data loaded")

    return orders,details,sav,nb_epicerie_bio,zip_invalid


def merge_data(orders: pd.DataFrame,details: pd.DataFrame,sav: pd.DataFrame) -> pd.DataFrame:

    df = orders.drop(columns=['ticket_at', 'raw_subject', 'value'])
    df = pd.merge(df,details,on='order_id',how='left')
    df = pd.merge(df,sav,on='customer_id',how='left')

    print("\n✅ data merged")

    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:

    print(f"\n Shape before cleaning : {df.shape}")

    # Filter created_at : starting from 01/01/2020
    df['created_at']= pd.to_datetime(df['created_at'])
    df=df.loc[df['created_at']>='2020',:]
    df['month']=df['created_at'].dt.month

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Drop missing values for ZIP code
    df.dropna(axis=0, subset='zip', inplace=True)

    # Replace outliers for DELAY
    df = df[df["delay"].between(left=0, right=90)]

    ### Transform weight (from g en Kg)
    df['weight']=df['weight']/1000

    ### Create a new column : department
    df['zip']=df['zip'].replace(' ','')
    df['zip']=df['zip'].replace('\r\n94170','94170')
    df['zip'] = df['zip'].str[:5]
    df['department'] = df['zip'].str[:-3]
    df['department']=df['department'].replace(['01','02','03','04','05','06','07','08','09'],['1','2','3','4','5','6','7','8','9'])
    df.dropna(axis=0, subset='department', inplace=True)

    print(f"\n Shape after cleaning : {df.shape}")
    print("\n✅ data cleaned")

    return df

def merge_zip(df:pd.DataFrame,nb_epicerie_bio:pd.DataFrame,zip_invalid:pd.DataFrame) -> pd.DataFrame:

    df = pd.merge(df,nb_epicerie_bio,on='zip',how='left')
    df = pd.merge(df,zip_invalid[['zip','lat']],on='zip',how='left')

    df.loc[df['lat'].isna(),'zip_valid']=0
    df.loc[df['nb_epiceries_bio_1km'].notna(),'zip_valid']=1
    df.loc[df['zip_valid'].isna(),'zip_valid']=1
    df.drop(columns={'lat'},inplace=True)

    print("\n✅ ZIP data merged")

    return df


def compress(df, **kwargs):
    """
    Reduces size of dataframe by downcasting numerical columns
    """

    input_size = df.memory_usage(index=True).sum()/ 1024**2
    print("old dataframe size: ", round(input_size,2), 'MB')

    in_size = df.memory_usage(index=True).sum()
    for t in ["float", "integer"]:
        l_cols = list(df.select_dtypes(include=t))
        for col in l_cols:
            df[col] = pd.to_numeric(df[col], downcast=t)
    out_size = df.memory_usage(index=True).sum()
    ratio = (1 - round(out_size / in_size, 2)) * 100

    print("optimized size by {} %".format(round(ratio,2)))
    print("new dataframe size: ", round(out_size / 1024**2,2), " MB")

    return df
