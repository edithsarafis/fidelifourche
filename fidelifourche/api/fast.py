from datetime import datetime
from socket import if_nameindex, if_nametoindex
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from fidelifourche.registry import load_model


import pandas as pd

app = FastAPI()

app.state.model = load_model()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# http://127.0.0.1:8000/predict?pickup_datetime=2012-10-06 12:10:20&pickup_longitude=40.7614327&pickup_latitude=-73.9798156&dropoff_longitude=40.6513111&dropoff_latitude=-73.8803331&passenger_count=2
@app.get("/predict")
def predict(customer_id: str,
            zip: str,
            aov: float,
            weight: float,
            delivery_type: str,
            carrier: str,
            created_at: str,
            order_id: str,
            share_refunds: float,
            share_discount: float,
            Baby: float,
            Vrac: float,
            Sale: float,
            Bois: float,
            Mais: float,
            Sucr: float,
            Sant: float,
            Alco: float,
            Beau: float,
            acquisition_channel: str,
            delay: float,
            quantity: str,
            ticket_at: str,
            month: str,
            department:str,
            nb_epiceries_bio_1km: float,
            zip_valid: float):
    """
    we use type hinting to indicate the data types expected
    for the parameters of the function
    FastAPI uses this information in order to hand errors
    to the developpers providing incompatible parameters
    FastAPI also provides variables of the expected data type to use
    without type hinting we need to manually convert
    the parameters of the functions which are all received as strs
    """

    X_new = pd.DataFrame.from_dict(dict(
            key=[customer_id],  # useless but the pipeline requires it
            customer_id=[customer_id],
            zip=[zip],
            aov=[aov],
            weight=[weight],
            delivery_type=[delivery_type],
            carrier=[carrier],
            created_at=[created_at],
            order_id=[order_id],
            share_refunds=[share_refunds],
            share_discount=[share_discount],
            Baby=[Baby],
            Vrac=[Vrac],
            Sale=[Sale],
            Bois=[Bois],
            Mais=[Mais],
            Sucr=[Sucr],
            Sant=[Sant],
            Alco=[Alco],
            Beau=[Beau],
            acquisition_channel=[acquisition_channel],
            delay=[delay],
            quantity= [quantity],
            ticket_at= [ticket_at],
            month= [month],
            department=[department],
            nb_epiceries_bio_1km= [nb_epiceries_bio_1km],
            zip_valid=[zip_valid]
            ))

    y_pred = app.state.model.predict(X_new)

    return {'bool_churn': float(y_pred)}


@app.get("/")
def root():
    return {'Welcome to Fidelifourche'}
