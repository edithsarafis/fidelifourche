import pandas as pd

def merge_data(orders: pd.DataFrame,details: pd.DataFrame,catalog: pd.DataFrame) -> pd.DataFrame:
    """
    merge raw data
    """
    pass
    #print("\n✅ data merged")

    #return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    clean raw data by removing buggy or irrelevant transactions
    or columns for the training set
    """

    print(f"\n Shape before cleaning : {df.shape}")
    #drop duplicates
    df.drop_duplicates(inplace=True)

    #drop irrelevant values

    #replace missing values

    print(f"\n Shape after cleaning : {df.shape}")
    print("\n✅ data cleaned")

    return df
