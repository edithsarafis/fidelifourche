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

    ## Filter data
    #Filter created_at : starting from 01/01/2020

    df=df.loc[df['created_at']>='2020',:]

    ## Clean data
    ### Drop duplicates

    df.drop_duplicates(inplace=True)

    ### Drop missing values for ZIP code

    df.dropna(axis=0, subset='zip', inplace=True)

    ### Replace outliers for DELAY
    # Edith -> peut etre plutot -1 pour qu'on différencie avec des vraies delay = 0 ?

    df['delay'].replace([-3367,-3366,-3247,-3244,-709,-703],0,inplace=True)

    ## Feature Engineering
    ### Convert to datetime

    df.loc[:,'created_at']= pd.to_datetime(df['created_at'])
    df.loc[:,'ticket_at']= pd.to_datetime(df['ticket_at'])

    ### Transform weight (from g en Kg)

    df.loc[:,'weight']=df['weight']/1000

    ### Create a new column : department

    df.loc[:,'department'] = df['zip'].str[:2]
    

    print(f"\n Shape after cleaning : {df.shape}")
    print("\n✅ data cleaned")

    return df
